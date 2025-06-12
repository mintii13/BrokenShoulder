import os
import pprint
import argparse
import torch.backends.cudnn as cudnn

import torch
import torch.nn as nn

from config.defaults import _C as config, update_config
from utils import train_util, log_util, loss_util, optimizer_util, anomaly_util
import models as models
from models.wresnet1024_cattn_tsm import ASTNet as get_net1
from models.wresnet2048_multiscale_cattn_tsmplus_layer6 import ASTNet as get_net2
from utils.loss_util import EntropyLossEncap
import datasets


def parse_args():

    parser = argparse.ArgumentParser(description='ASTNet')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        default='config/shanghaitech_wresnet.yaml', type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)
    return args

def print_mem_dim(model, logger):
    """Print memory dimensions of MemModules"""
    # Get the actual model (unwrap DataParallel if needed)
    actual_model = model.module if hasattr(model, 'module') else model
    
    logger.info("=== Memory Dimensions ===")
    
    if hasattr(actual_model, 'mem_rep8'):
        logger.info(f"mem_rep8 dim: {actual_model.mem_rep8.mem_dim}")
    
    if hasattr(actual_model, 'mem_rep2'):
        logger.info(f"mem_rep2 dim: {actual_model.mem_rep2.mem_dim}")
    
    if hasattr(actual_model, 'mem_rep1'):
        logger.info(f"mem_rep1 dim: {actual_model.mem_rep1.mem_dim}")
    
    logger.info("=" * 25)


def main():
    args = parse_args()

    logger, final_output_dir, tb_log_dir = \
        log_util.create_logger(config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    if config.DATASET.DATASET == "ped2":
        model = get_net1(config)
    else:
        model = get_net2(config)

    logger.info('Model: {}'.format(model.get_name()))
    print_mem_dim(model, logger)

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    losses = loss_util.MultiLossFunction(config=config).cuda()

    optimizer = optimizer_util.get_optimizer(config, model)

    scheduler = optimizer_util.get_scheduler(config, optimizer)

    train_dataset = eval('datasets.get_data')(config)
    

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True,
        drop_last=True
    )
    logger.info('Number videos: {}'.format(len(train_dataset)))

    # Khởi tạo biến để theo dõi model có loss thấp nhất
    best_loss = float('inf')
    best_epoch = -1
    best_model_path = None

    last_epoch = config.TRAIN.BEGIN_EPOCH
    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        # Train và nhận về average loss của epoch
        avg_loss = train(config, train_loader, model, losses, optimizer, epoch, logger)

        scheduler.step()

        # Kiểm tra và lưu model có loss thấp nhất
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            
            # Xóa model tốt nhất cũ (nếu có)
            if best_model_path is not None and os.path.exists(best_model_path):
                os.remove(best_model_path)
                logger.info('=> Removed previous best model: {}'.format(best_model_path))
            
            # Lưu model tốt nhất mới
            best_model_path = os.path.join(final_output_dir, 'best_model_epoch_{}_loss_{:.6f}.pth'.format(epoch + 1, avg_loss))
            torch.save(model.module.state_dict(), best_model_path)
            logger.info('=> New best model saved at epoch {} with loss {:.6f}: {}'.format(epoch + 1, avg_loss, best_model_path))

        # Lưu checkpoint mỗi 5 epoch
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(final_output_dir, 'checkpoint_epoch_{}.pth'.format(epoch + 1))
            torch.save(model.module.state_dict(), checkpoint_path)
            logger.info('=> Checkpoint saved at epoch {}: {}'.format(epoch + 1, checkpoint_path))

    # Lưu model cuối cùng
    final_model_state_file = os.path.join(final_output_dir, 'final_state_epoch_{}.pth'.format(config.TRAIN.END_EPOCH))
    logger.info('=> Saving final model state to {}'.format(final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    
    # In thông tin tổng kết về model tốt nhất
    logger.info('=' * 80)
    logger.info('TRAINING COMPLETED!')
    logger.info('Best model was saved at epoch {} with loss {:.6f}'.format(best_epoch + 1, best_loss))
    logger.info('Best model path: {}'.format(best_model_path))
    logger.info('=' * 80)

def compute_contrastive_loss(processed_output, att_weights, mem_fea_align, temperature=0.2):
    """
    Simple contrastive loss:
    - Positive: similarity between query and most_similar_idx memory
    - Negative: similarity between most_similar_idx memory and other memories
    
    Args:
        processed_output: output from process_memory_with_pull_push
        att_weights: [N, M, H, W] - similarity between query and memory
        mem_fea_align: [M, M] - similarity between memory features
        temperature: temperature for contrastive loss
    
    Returns:
        contrastive_loss: scalar tensor
    """
    
    most_similar_idx = processed_output['most_similar_idx']  # [N, H, W]
    N, M, H, W = att_weights.shape
    
    # Get positive similarities: query vs most_similar_idx memory
    # att_weights[n, most_similar_idx[n,h,w], h, w] for each pixel
    batch_idx = torch.arange(N, device=att_weights.device).view(N, 1, 1).expand(N, H, W)
    height_idx = torch.arange(H, device=att_weights.device).view(1, H, 1).expand(N, H, W)
    width_idx = torch.arange(W, device=att_weights.device).view(1, 1, W).expand(N, H, W)
    
    positive_sim = att_weights[batch_idx, most_similar_idx, height_idx, width_idx]  # [N, H, W]
    
    # Get negative similarities: most_similar_idx memory vs other memories
    flat_most_sim = most_similar_idx.view(-1)  # [N*H*W]
    
    # For each selected memory, get its similarity with all other memories
    selected_mem_sim = mem_fea_align[flat_most_sim]  # [N*H*W, M]
    
    # Remove self-similarity (set to very negative value so it doesn't affect softmax)
    mask = torch.ones_like(selected_mem_sim)
    batch_indices = torch.arange(N*H*W, device=mem_fea_align.device)
    mask[batch_indices, flat_most_sim] = 0
    selected_mem_sim = selected_mem_sim * mask - 1e9 * (1 - mask)  # Mask self-similarity
    
    # Reshape back to [N, H, W, M]
    negative_sim = selected_mem_sim.view(N, H, W, M)
    
    # Contrastive loss: InfoNCE style
    positive_exp = torch.exp(positive_sim / temperature)  # [N, H, W]
    negative_exp = torch.exp(negative_sim / temperature).sum(dim=3)  # [N, H, W]
    
    # Total denominator = positive + sum of negatives
    denominator = positive_exp + negative_exp
    
    contrastive_loss = -torch.log(positive_exp / denominator).mean()
    
    return contrastive_loss

def compute_contrastive_loss_isolated(processed_output, att_weights, mem_fea_align, temperature=0.2):
    """
        Contrastive loss with isolated gradient flow
        
    Chỉ memory parameters sẽ nhận gradient từ loss này
    Main model parameters bị detach khỏi loss này
    """

    most_similar_idx = processed_output['most_similar_idx']  # [N, H, W]
    N, M, H, W = att_weights.shape

    # DETACH query features để cắt gradient về main model
    att_weights_detached = att_weights.detach()  # Cắt gradient từ main model

    # Memory similarity vẫn giữ gradient (cho memory parameters)
    # mem_fea_align giữ nguyên gradient

    # Get positive similarities
    batch_idx = torch.arange(N, device=att_weights.device).view(N, 1, 1).expand(N, H, W)
    height_idx = torch.arange(H, device=att_weights.device).view(1, H, 1).expand(N, H, W)
    width_idx = torch.arange(W, device=att_weights.device).view(1, 1, W).expand(N, H, W)

    positive_sim = att_weights_detached[batch_idx, most_similar_idx, height_idx, width_idx]  # [N, H, W]

    # Get negative similarities từ memory-to-memory similarity
    flat_most_sim = most_similar_idx.view(-1)  # [NHW]
    selected_mem_sim = mem_fea_align[flat_most_sim]  # [NHW, M] - có gradient cho memory

    # Remove self-similarity
    mask = torch.ones_like(selected_mem_sim)
    batch_indices = torch.arange(N*H*W, device=mem_fea_align.device)
    mask[batch_indices, flat_most_sim] = 0
    selected_mem_sim = selected_mem_sim * mask - 1e9 * (1 - mask)

    negative_sim = selected_mem_sim.view(N, H, W, M)

    # Contrastive loss
    positive_exp = torch.exp(positive_sim / temperature)
    negative_exp = torch.exp(negative_sim / temperature).sum(dim=3)

    denominator = positive_exp + negative_exp
    contrastive_loss = -torch.log(positive_exp / denominator).mean()

    return contrastive_loss
def spatial_flatten_kl_loss(x):
    """
    Flatten tensor theo chiều spatial và tính KL loss với phân phối normal
    """
    batch_size, channels, height, width = x.shape
    spatial_dim = height * width
    flattened = x.view(batch_size, channels, spatial_dim)
    
    reshaped = flattened.contiguous().view(-1, spatial_dim)
    
    empirical_mean = torch.mean(reshaped, dim=1, keepdim=True)
    empirical_var = torch.var(reshaped, dim=1, keepdim=True, unbiased=False)
    empirical_log_var = torch.log(empirical_var + 1e-8)
    
    target_mean = torch.zeros_like(empirical_mean)
    target_log_var = torch.zeros_like(empirical_log_var)
    
    kl_loss = 0.5 * (
        target_log_var - empirical_log_var +
        torch.exp(empirical_log_var) +
        empirical_mean.pow(2) - 1
    )
    
    kl_loss = kl_loss.mean()
    return kl_loss


def train(config, train_loader, model, loss_functions, optimizer, epoch, logger):
    loss_func_mse = nn.MSELoss(reduction='none')
    entropy_loss_func = EntropyLossEncap().to("cuda")
    model.train()

    total_loss = 0.0
    num_batches = 0

    for i, data in enumerate(train_loader):
        # Decode input
        inputs, target = train_util.decode_input(input=data, train=True)
        
        # Forward pass
        # Returns: output, att, sim_mem_8, processed_output, f8_for_decoder, contrastive_output
        output, att, sim_mem_8, processed_output, f8_for_decoder, contrastive_output = model(inputs)
        
        # Extract contrastive loss from the projection space
        contrastive_loss = contrastive_output['contrastive_loss']
        
        # Other losses
        kl_loss = spatial_flatten_kl_loss(f8_for_decoder)  # KL loss on decoder input features
        entropy_loss = 0
        for att_step in att:
            entropy_loss += entropy_loss_func(att_step)

        # Compute reconstruction losses
        target = target.cuda(non_blocking=True)
        inte_loss, grad_loss, msssim_loss, l2_loss = loss_functions(output, target)
        
        # Combined loss
        # f8_for_decoder đã được weight bởi attention từ projection space
        # contrastive_loss được tính trong projection space  
        # decoder sử dụng original features * projection attention
        # loss = l2_loss + contrastive_loss + kl_loss + 0.0002 * entropy_loss
        loss = inte_loss + l2_loss + contrastive_loss + kl_loss 
        
        total_loss += loss.item()
        num_batches += 1

        # Compute PSNR
        mse_imgs = torch.mean(loss_func_mse((output + 1) / 2, (target + 1) / 2)).item()
        psnr = anomaly_util.psnr_park(mse_imgs)

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cur_lr = optimizer.param_groups[0]['lr']
        if (i + 1) % config.PRINT_FREQ == 0:
            # Get additional info about projection space
            max_attention = torch.max(contrastive_output['attention_weights']).item()
            mean_attention = torch.mean(contrastive_output['attention_weights']).item()
            
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Lr {lr:.6f}\t' \
                  '[Inte {inte:.4f} + L2 {l2:.4f} + Contrastive_Proj {contrastive:.4f} + KL {kl:.4f}]\t' \
                  'PSNR {psnr:.2f}\t' \
                  'Att [max:{max_att:.3f}, mean:{mean_att:.3f}]'.format(
                      epoch+1, i+1, len(train_loader),
                      lr=cur_lr,
                      inte = inte_loss,
                      l2=l2_loss, 
                      contrastive=contrastive_loss, 
                      kl=kl_loss, 
                      psnr=psnr,
                      max_att=max_attention,
                      mean_att=mean_attention
                  )
            logger.info(msg)

    avg_loss = total_loss / num_batches
    logger.info('Epoch [{0}] completed. Average Loss: {1:.6f}'.format(epoch + 1, avg_loss))
    return avg_loss 



if __name__ == '__main__':
    main()