import logging
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from models.wider_resnet import wresnet
from models.basic_modules import ConvBnRelu, ConvTransposeBnRelu, initialize_weights
from models.mem_module import MemModule

logger = logging.getLogger(__name__)


class EfficientProjectionMLP(nn.Module):
    """Memory-efficient MLP projection head with gradient checkpointing"""
    def __init__(self, input_dim, hidden_dim, output_dim, use_checkpoint=True):
        super(EfficientProjectionMLP, self).__init__()
        self.use_checkpoint = use_checkpoint
        
        # Use smaller intermediate dimension to reduce memory
        efficient_hidden = min(hidden_dim, input_dim // 2)
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, efficient_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),  # Add dropout for regularization
            nn.Linear(efficient_hidden, output_dim)
        )
    
    def _forward_impl(self, x_flat):
        return self.projection(x_flat)
    
    def forward(self, x):
        # x shape: [N, C, H, W]
        N, C, H, W = x.shape
        
        # Process in chunks to reduce memory usage
        chunk_size = min(H * W, 4096)  # Process max 4096 pixels at once
        
        if H * W <= chunk_size:
            # Small enough to process at once
            x_flat = x.permute(0, 2, 3, 1).contiguous().view(-1, C)
            
            if self.use_checkpoint and self.training:
                projected = checkpoint(self._forward_impl, x_flat, use_reentrant=False)
            else:
                projected = self._forward_impl(x_flat)
            
            projected = projected.view(N, H, W, -1).permute(0, 3, 1, 2)
        else:
            # Process in chunks
            x_flat = x.permute(0, 2, 3, 1).contiguous().view(N, -1, C)  # [N, H*W, C]
            chunks = []
            
            for i in range(0, H * W, chunk_size):
                chunk = x_flat[:, i:i+chunk_size, :]  # [N, chunk_size, C]
                chunk_flat = chunk.view(-1, C)
                
                if self.use_checkpoint and self.training:
                    chunk_proj = checkpoint(self._forward_impl, chunk_flat, use_reentrant=False)
                else:
                    chunk_proj = self._forward_impl(chunk_flat)
                
                chunk_proj = chunk_proj.view(N, -1, chunk_proj.shape[-1])
                chunks.append(chunk_proj)
            
            projected_flat = torch.cat(chunks, dim=1)  # [N, H*W, output_dim]
            projected = projected_flat.view(N, H, W, -1).permute(0, 3, 1, 2)
        
        return projected


class ASTNet(nn.Module):
    def get_name(self):
        return self.model_name

    def __init__(self, config, pretrained=False):
        super(ASTNet, self).__init__()
        frames = config.MODEL.ENCODED_FRAMES
        final_conv_kernel = config.MODEL.EXTRA.FINAL_CONV_KERNEL
        self.model_name = config.MODEL.NAME
        
        # Memory optimization flags
        self.use_gradient_checkpointing = getattr(config.MODEL, 'USE_CHECKPOINT', True)
        self.memory_efficient_attention = getattr(config.MODEL, 'MEMORY_EFFICIENT_ATTENTION', True)
        
        logger.info('=> ' + self.model_name + '_1024: (CATTN + TSM) - Ped2 [Memory Optimized]')

        self.wrn38 = wresnet(config, self.model_name, pretrained=pretrained)

        channels = [4096, 2048, 1024, 512, 256, 128]

        # Reduce intermediate channels to save memory
        efficient_channels = [c // 2 for c in channels]  # Halve channel dimensions
        
        self.conv_x8 = nn.Conv2d(channels[0] * frames, efficient_channels[1], kernel_size=1, bias=False)
        self.conv_x2 = nn.Conv2d(channels[4] * frames, efficient_channels[4], kernel_size=1, bias=False)
        self.conv_x1 = nn.Conv2d(channels[5] * frames, efficient_channels[5], kernel_size=1, bias=False)

        self.up8 = ConvTransposeBnRelu(efficient_channels[1], efficient_channels[2], kernel_size=2)
        self.up4 = ConvTransposeBnRelu(efficient_channels[2], efficient_channels[3], kernel_size=2)
        self.up2 = ConvTransposeBnRelu(efficient_channels[3], efficient_channels[4], kernel_size=2)

        self.tsm_left = TemporalShift(n_segment=4, n_div=16, direction='left')

        # Memory module with reduced dimensions
        self.mem_rep8 = MemModule(mem_dim=200, fea_dim=efficient_channels[1], shrink_thres=0.0025) 
        
        # Efficient projection MLPs
        projection_dim = 256  # Reduced from 512
        self.query_projection = EfficientProjectionMLP(
            input_dim=efficient_channels[1], 
            hidden_dim=512, 
            output_dim=projection_dim,
            use_checkpoint=self.use_gradient_checkpointing
        )
        self.memory_projection = nn.Linear(efficient_channels[1], projection_dim)
        
        self.final = nn.Sequential(
            ConvBnRelu(efficient_channels[4], efficient_channels[5], kernel_size=1, padding=0),
            ConvBnRelu(efficient_channels[5], efficient_channels[5], kernel_size=3, padding=1),
            nn.Conv2d(efficient_channels[5], 3,
                      kernel_size=final_conv_kernel,
                      padding=1 if final_conv_kernel == 3 else 0,
                      bias=False)
        )

        initialize_weights(self.conv_x1, self.conv_x2, self.conv_x8)
        initialize_weights(self.up2, self.up4, self.up8)
        initialize_weights(self.final)
        initialize_weights(self.query_projection, self.memory_projection)

    def compute_contrastive_in_projection_space(self, query_features, memory_features, temperature=0.07):
        """
        Memory-efficient contrastive computation with chunked processing
        """
        N, C, H, W = query_features.shape
        M = memory_features.shape[0]
        
        # Project features efficiently
        projected_query = self.query_projection(query_features)
        projected_memory = self.memory_projection(memory_features)
        
        # Normalize
        projected_query_norm = nn.functional.normalize(projected_query, dim=1)
        projected_memory_norm = nn.functional.normalize(projected_memory, dim=1)
        
        proj_dim = projected_query_norm.shape[1]
        
        # Memory-efficient similarity computation using chunked processing
        query_flat = projected_query_norm.permute(0, 2, 3, 1).contiguous().view(-1, proj_dim)
        
        # Process similarity in chunks to avoid OOM
        chunk_size = min(query_flat.shape[0], 1024)  # Smaller chunk size without autocast
        similarities = []
        
        for i in range(0, query_flat.shape[0], chunk_size):
            chunk_query = query_flat[i:i+chunk_size]
            # Use more memory-efficient matrix multiplication
            chunk_sim = torch.matmul(chunk_query, projected_memory_norm.t())
            similarities.append(chunk_sim)  # Detach to save memory in backward pass
            
            # Clear intermediate tensors
            del chunk_query, chunk_sim
        
        similarity_matrix = torch.cat(similarities, dim=0)
        del similarities  # Free memory
        
        # Compute attention with memory efficiency
        # Use temperature scaling before softmax to avoid overflow
        scaled_similarity = similarity_matrix / temperature
        attention_weights = torch.softmax(scaled_similarity, dim=1)
        attention_weights = attention_weights.view(N, H, W, M).permute(0, 3, 1, 2)
        
        # Simplified loss computation to save memory
        most_similar_idx = torch.argmax(scaled_similarity, dim=1)
        batch_indices = torch.arange(scaled_similarity.shape[0], device=scaled_similarity.device)
        positive_sim = scaled_similarity[batch_indices, most_similar_idx]
        
        # Use logsumexp for numerical stability and memory efficiency
        log_sum_exp = torch.logsumexp(scaled_similarity, dim=1)
        contrastive_loss = (-positive_sim + log_sum_exp).mean()
        
        # Memory-efficient feature combination
        max_attention = torch.max(attention_weights, dim=1)[0].unsqueeze(1)
        
        # Clear intermediate tensors
        del similarity_matrix, scaled_similarity
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            'contrastive_loss': contrastive_loss,
            'attention_weights': attention_weights,
            'max_attention': max_attention,
            'most_similar_idx': most_similar_idx.view(N, H, W)
        }

    @staticmethod
    def process_memory_with_pull_push(mem_module_output, temperature=0.1):
        """Efficient memory processing"""
        f8 = mem_module_output['output']
        att_8 = mem_module_output['att']
        sim_mem_8 = mem_module_output['mem_fea_align']
        
        N, C, H, W = f8.shape
        M = att_8.shape[1]
        
        # Process in chunks if necessary
        if N * H * W > 8192:  # If too large, process in chunks
            chunk_size = 8192 // (H * W)
            results = []
            
            for i in range(0, N, chunk_size):
                chunk_att = att_8[i:i+chunk_size]
                chunk_reshaped = chunk_att.permute(0, 2, 3, 1).contiguous().view(-1, M)
                chunk_most_sim = torch.argmax(chunk_reshaped, dim=1)
                results.append(chunk_most_sim) 
            
            x8_mem_most_sim = torch.cat(results, dim=0)
        else:
            att_reshaped = att_8.permute(0, 2, 3, 1).contiguous().view(-1, M)
            x8_mem_most_sim = torch.argmax(att_reshaped, dim=1)
        
        return {
            'features': f8,
            'attention': att_8,
            'most_similar_idx': x8_mem_most_sim.view(N, H, W),
        }

    def forward(self, x):
        # Process frames efficiently without autocast
        x1s, x2s, x8s = [], [], []
        
        # Process frames one by one to save memory
        for i, xi in enumerate(x):
            if self.use_gradient_checkpointing and self.training:
                x1, x2, x8 = checkpoint(self.wrn38, xi, use_reentrant=False)
            else:
                x1, x2, x8 = self.wrn38(xi)
            
            x8s.append(x8)
            x2s.append(x2)
            x1s.append(x1)
            
            # Clear cache periodically
            if (i + 1) % 2 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Concatenate and reduce dimensions
        x8 = self.conv_x8(torch.cat(x8s, dim=1))
        x2 = self.conv_x2(torch.cat(x2s, dim=1))
        x1 = self.conv_x1(torch.cat(x1s, dim=1))
        
        # Clear intermediate lists
        del x1s, x2s, x8s
        
        # Temporal shift
        left = self.tsm_left(x8)
        x8 = x8 + left
        del left
        
        # Memory processing
        x8_original = x8.clone()
        res_mem8 = self.mem_rep8(x8)
        
        # Process memory output
        processed_output = self.process_memory_with_pull_push(res_mem8, temperature=0.1)
        
        # Contrastive learning with memory efficiency
        memory_features = self.mem_rep8.memory.weight
        contrastive_output = self.compute_contrastive_in_projection_space(
            query_features=x8_original,
            memory_features=memory_features,
            temperature=0.2
        )
        
        # Use attention-weighted features
        f8_for_decoder = x8_original * contrastive_output['max_attention']
        
        # Decoder with gradient checkpointing
        if self.use_gradient_checkpointing and self.training:
            x = checkpoint(self.up8, f8_for_decoder, use_reentrant=False)
            x = checkpoint(self.up4, x, use_reentrant=False)
            x = checkpoint(self.up2, x, use_reentrant=False)
            output = checkpoint(self.final, x, use_reentrant=False)
        else:
            x = self.up8(f8_for_decoder)
            x = self.up4(x)
            x = self.up2(x)
            output = self.final(x)
        
        return (output, [res_mem8['att']], res_mem8['mem_fea_align'], 
               processed_output, f8_for_decoder, contrastive_output)


class ChannelAttention(nn.Module):
    def __init__(self, input_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Reduce intermediate channels for memory efficiency
        reduced_channels = max(input_channels // reduction, 8)
        
        self.layer = nn.Sequential(
            nn.Conv2d(input_channels, reduced_channels, 1, bias=False),  # Remove bias to save memory
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, input_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.layer(y)
        return x * y


class TemporalShift(nn.Module):
    def __init__(self, n_segment=4, n_div=8, direction='left'):
        super(TemporalShift, self).__init__()
        self.n_segment = n_segment
        self.fold_div = n_div
        self.direction = direction
        print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        return self.shift(x, self.n_segment, fold_div=self.fold_div, direction=self.direction)

    @staticmethod
    def shift(x, n_segment=4, fold_div=8, direction='left'):
        bz, nt, h, w = x.size()
        c = nt // n_segment
        x = x.view(bz, n_segment, c, h, w)

        fold = c // fold_div
        
        # Use in-place operations to save memory
        if direction == 'left':
            x_shifted = x.clone()
            x_shifted[:, :-1, :fold] = x[:, 1:, :fold]
        elif direction == 'right':
            x_shifted = x.clone()
            x_shifted[:, 1:, :fold] = x[:, :-1, :fold]
        else:
            x_shifted = x.clone()
            x_shifted[:, :-1, :fold] = x[:, 1:, :fold]
            x_shifted[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]

        return x_shifted.view(bz, nt, h, w)


# Additional memory optimization utilities
class MemoryOptimizer:
    """Utility class for memory optimization during training"""
    
    @staticmethod
    def enable_memory_efficient_training(model):
        """Enable various memory optimization techniques"""
        
        # Enable gradient checkpointing
        if hasattr(model, 'use_gradient_checkpointing'):
            model.use_gradient_checkpointing = True
        
        # Enable memory efficient attention
        if hasattr(model, 'memory_efficient_attention'):
            model.memory_efficient_attention = True
    
    @staticmethod
    def clear_cache_periodically():
        """Clear GPU cache periodically during training"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @staticmethod
    def optimize_dataloader_settings():
        """Recommended dataloader settings for memory efficiency"""
        return {
            'batch_size': 2,  # Reduce batch size
            'num_workers': 4,  # Reduce number of workers
            'pin_memory': True,
            'prefetch_factor': 2,
            'persistent_workers': True
        }
    
    @staticmethod
    def get_memory_efficient_optimizer_settings():
        """Recommended optimizer settings for memory efficiency"""
        return {
            'lr': 1e-4,
            'weight_decay': 1e-5,
            'eps': 1e-8,
            'foreach': False,  # Disable foreach for memory efficiency
            'fused': False     # Disable fused for compatibility
        }