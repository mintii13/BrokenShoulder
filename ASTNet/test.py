import os
import pprint
import argparse
import tqdm
import glob
import time
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import matplotlib
matplotlib.use('Agg')  # Không dùng GUI backend
import matplotlib.pyplot as plt

import datasets
from utils import train_util, log_util, anomaly_util
from config.defaults import _C as config, update_config
from models.wresnet1024_cattn_tsm import ASTNet as get_net1
from models.wresnet2048_multiscale_cattn_tsmplus_layer6 import ASTNet as get_net2

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def parse_args():
    parser = argparse.ArgumentParser(description='Batch Test Anomaly Detection')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        default='config/ped2_wresnet.yaml', type=str)
    parser.add_argument('--base-path', help='base path for batch testing',
                        default='D:/FPTU-sourse/Term4/ResFes_FE/References/Anomaly Detection/Reconstruction/astnet/ASTNet/output/ped2/1mem_newloss_200/', type=str)
    parser.add_argument('--epoch-range', help='epoch range for batch testing (e.g., 32-36 or 32 for single epoch)', 
                        default='10-60', type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args

def inference(config, data_loader, model, save_frames_info=None, epoch_info=None):
    """Enhanced inference with frame information saving"""
    loss_func_mse = nn.MSELoss(reduction='none')

    model.eval()
    psnr_list = []
    ef = config.MODEL.ENCODED_FRAMES
    df = config.MODEL.DECODED_FRAMES
    fp = ef + df  # number of frames to process
    
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            print('[{}/{}]'.format(i+1, len(data_loader)))
            psnr_video = []

            # compute the output
            video, video_name = train_util.decode_input(input=data, train=False)
            video = [frame.to(device=config.GPUS[0]) for frame in video]
            
            for f in tqdm.tqdm(range(len(video) - fp)):
                inputs = video[f:f + fp]
                output = model(inputs)
                target = video[f + fp:f + fp + 1][0]

                # compute PSNR for each frame
                mse_imgs = torch.mean(loss_func_mse((output[0] + 1) / 2, (target[0] + 1) / 2)).item()
                psnr = anomaly_util.psnr_park(mse_imgs)
                psnr_video.append(psnr)
                
                # Store frame information for visualization if requested
                # Only save every 10th frame (f % 10 == 0) and only for first video
                if save_frames_info is not None and i == 0 and f % 10 == 0:
                    frame_key = f"video_{i}_frame_{f + fp}"
                    save_frames_info[frame_key] = {
                        'video_idx': i,
                        'frame_idx': f + fp,
                        'psnr': psnr,
                        'target': (target[0] + 1) / 2,  # Convert back to [0,1]
                        'predicted': (output[0] + 1) / 2,  # Convert back to [0,1]
                        'video_name': video_name[0] if isinstance(video_name, list) else video_name
                    }

            psnr_list.append(psnr_video)
    return psnr_list

def save_frame_comparison(target_frame, predicted_frame, video_idx, frame_idx, psnr_value, base_path, epoch_info, logger):
    """Save comparison of target and predicted frames"""
    try:
        # Create epoch-specific folder for comparison images
        epoch_folder_name = f"compare_image_{epoch_info}"
        epoch_folder_path = os.path.join(base_path, epoch_folder_name)
        
        # Create folder if it doesn't exist
        if not os.path.exists(epoch_folder_path):
            os.makedirs(epoch_folder_path)
            logger.info(f'Created folder: {epoch_folder_path}')
        
        # Convert tensor to numpy array for visualization
        target_np = target_frame.cpu().detach().squeeze().numpy()
        predicted_np = predicted_frame.cpu().detach().squeeze().numpy()
        
        # Handle different tensor shapes
        if len(target_np.shape) == 3:  # [C, H, W]
            if target_np.shape[0] == 3:  # RGB
                target_np = np.transpose(target_np, (1, 2, 0))
                predicted_np = np.transpose(predicted_np, (1, 2, 0))
            elif target_np.shape[0] == 1:  # Grayscale with channel dimension
                target_np = target_np.squeeze(0)
                predicted_np = predicted_np.squeeze(0)
        
        # Ensure values are in [0, 1] range
        target_np = np.clip(target_np, 0, 1)
        predicted_np = np.clip(predicted_np, 0, 1)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot original frame
        if len(target_np.shape) == 3:  # Color image
            axes[0].imshow(target_np)
        else:  # Grayscale image
            axes[0].imshow(target_np, cmap='gray')
        axes[0].set_title(f'Original Frame\nVideo {video_idx+1}, Frame {frame_idx}, {epoch_info}')
        axes[0].axis('off')
        
        # Plot predicted frame
        if len(predicted_np.shape) == 3:  # Color image
            axes[1].imshow(predicted_np)
        else:  # Grayscale image
            axes[1].imshow(predicted_np, cmap='gray')
        axes[1].set_title(f'Predicted Frame\nPSNR: {psnr_value:.2f}, {epoch_info}')
        axes[1].axis('off')
        
        # Add main title
        plt.suptitle(f'Frame Comparison Video {video_idx+1} Frame {frame_idx} - {epoch_info}', fontsize=16)
        plt.tight_layout()
        
        # Save figure in epoch-specific folder
        filename = f'FRAME_COMPARISON_video{video_idx+1}_frame{frame_idx}_psnr{psnr_value:.2f}.png'
        save_path = os.path.join(epoch_folder_path, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f'Saved frame comparison to: {save_path}')
        
    except Exception as e:
        logger.error(f'Error saving frame comparison: {str(e)}')

def save_all_frames_video1(frames_info, base_path, epoch_info, logger):
    """Save frame comparisons for selected frames of video 1 (every 10th frame)"""
    if not frames_info:
        logger.warning("No frame information available for visualization")
        return
    
    # Filter frames for video 1 (video_idx = 0)
    video1_frames = {}
    for frame_key, frame_data in frames_info.items():
        if frame_data['video_idx'] == 0:  # Video 1 has index 0
            video1_frames[frame_data['frame_idx']] = frame_data
    
    if not video1_frames:
        logger.warning("No frames found for video 1")
        return
    
    logger.info(f"Found {len(video1_frames)} frames for video 1")
    
    # Sort frames by frame index
    sorted_frame_indices = sorted(video1_frames.keys())
    
    # Save comparison for each frame
    for frame_idx in sorted_frame_indices:
        frame_data = video1_frames[frame_idx]
        logger.info(f"Saving frame comparison for Video 1, Frame {frame_idx}, PSNR: {frame_data['psnr']:.2f}")
        
        save_frame_comparison(
            frame_data['target'],
            frame_data['predicted'],
            frame_data['video_idx'],
            frame_data['frame_idx'],
            frame_data['psnr'],
            base_path,
            epoch_info,
            logger
        )

def plot_regularity_score(psnr_values, gt_labels, video_idx, epoch_name, base_path, logger):
    """Plot and save regularity score visualization for a specific video"""
    plt.figure(figsize=(12, 6))
    
    # Plot PSNR values (regularity score)
    plt.plot(psnr_values, 'b-', label='Regularity Score (PSNR)')
    
    # Create second y-axis for ground truth
    ax2 = plt.twinx()
    ax2.plot(gt_labels, 'r-', label='Ground Truth')
    ax2.set_ylim([-0.1, 1.1])
    ax2.set_ylabel('Anomaly (1 = anomalous)')
    
    plt.title(f'Regularity Score - Video {video_idx + 1} - {epoch_name}')
    plt.xlabel('Frame Number')
    plt.ylabel('Regularity Score (PSNR)')
    
    # Add separate legends for both axes
    plt.legend(loc='upper left')      # Legend for primary axis (PSNR)
    ax2.legend(loc='upper right')     # Legend for secondary axis (Ground Truth)
    
    plt.grid(True)
    plt.tight_layout()
    
    # Save figure to base path with clear naming
    filename = f'PSNR_PLOT_regularity_video{video_idx + 1}_{epoch_name}.png'
    save_path = os.path.join(base_path, filename)
    plt.savefig(save_path)
    logger.info(f'Saved regularity score plot to: {save_path}')
    plt.close()  # Close the figure to free memory

def test_single_model(model_path, config, logger, save_visualizations=False, base_path=None, epoch_info=None):
    """Test a single model and return its metrics"""
    logger.info(f'Testing model: {os.path.basename(model_path)}')
    
    config.defrost()
    config.MODEL.INIT_WEIGHTS = False
    config.freeze()

    gpus = [(config.GPUS[0])]
    
    # Create model based on dataset
    if config.DATASET.DATASET == "ped2":
        model = get_net1(config, pretrained=False)
    else:
        model = get_net2(config, pretrained=False)
        
    logger.info('Model: {}'.format(model.get_name()))
    model = nn.DataParallel(model, device_ids=gpus).cuda(device=gpus[0])

    # Load model with error handling for size mismatch
    try:
        state_dict = torch.load(model_path, map_location='cuda')
        if 'state_dict' in state_dict.keys():
            state_dict = state_dict['state_dict']
            model.load_state_dict(state_dict)
        else:
            model.module.load_state_dict(state_dict)
        logger.info(f"Successfully loaded model: {os.path.basename(model_path)}")
    except RuntimeError as e:
        if "size mismatch" in str(e):
            logger.error(f"Model architecture mismatch for {os.path.basename(model_path)}")
            logger.error(f"Error details: {str(e)}")
            logger.error("Please check if the model was trained with the same configuration")
            raise e
        else:
            raise e

    # Get test data
    test_dataset = eval('datasets.get_test_data')(config)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    # Load ground truth
    mat_loader = datasets.get_label(config)
    mat = mat_loader()

    # Dictionary to store frame information for visualization
    frames_info = {} if save_visualizations else None

    # Run inference
    psnr_list = inference(config, test_loader, model, frames_info, epoch_info)
    assert len(psnr_list) == len(mat), f'Ground truth has {len(mat)} videos, BUT got {len(psnr_list)} detected videos!'

    # Calculate AUC
    auc, fpr, tpr = anomaly_util.calculate_auc(config, psnr_list, mat)
    logger.info(f'AUC: {auc * 100:.1f}%')

    # Save visualizations if requested
    if save_visualizations and base_path and epoch_info:
        # Save frame comparisons for video 1 (every 10th frame)
        save_all_frames_video1(frames_info, base_path, epoch_info, logger)
        
        # Plot regularity scores for first 3 videos
        for video_idx in range(min(3, len(psnr_list))):
            plot_regularity_score(
                psnr_list[video_idx], 
                mat[video_idx], 
                video_idx,
                epoch_info,
                base_path,
                logger
            )

    return auc, psnr_list, mat

def parse_epoch_range(epoch_range):
    """Parse epoch range string to return start and end epochs"""
    if '-' in epoch_range:
        # Range format: "32-36"
        start_epoch, end_epoch = map(int, epoch_range.split('-'))
        return start_epoch, end_epoch
    else:
        # Single epoch format: "32"
        epoch = int(epoch_range)
        return epoch, epoch

def batch_test_models(base_path, epoch_range, config, logger):
    """Test multiple models in batch mode"""
    logger.info(f"Starting batch testing in: {base_path}")
    logger.info(f"Epoch range: {epoch_range}")
    
    # Parse epoch range
    start_epoch, end_epoch = parse_epoch_range(epoch_range)
    
    # Get list of model files
    model_files = []
    for epoch in range(start_epoch, end_epoch + 1):
        model_path = os.path.join(base_path, f"epoch_{epoch}.pth")
        if os.path.exists(model_path):
            model_files.append(model_path)
        else:
            logger.warning(f"Model file not found: {model_path}")
    
    if not model_files:
        logger.error("No model files found!")
        return
    
    logger.info(f"Found {len(model_files)} model files: {[os.path.basename(f) for f in model_files]}")
    logger.info(f"All output files will be saved to: {base_path}")
    
    # Dictionary to store results
    results = {}
    psnr_results = {}
    gt_labels_store = None
    
    # Test each model
    for model_path in model_files:
        model_name = os.path.basename(model_path)
        epoch_info = model_name.replace('.pth', '').replace('_', '')
        
        try:
            start_time = time.time()
            auc, psnr_list, gt_labels = test_single_model(
                model_path, config, logger, 
                save_visualizations=True, 
                base_path=base_path, 
                epoch_info=epoch_info
            )
            end_time = time.time()
            
            results[model_name] = {"auc": auc}
            psnr_results[model_name] = psnr_list
            
            # Store ground truth labels (should be the same for all models)
            if gt_labels_store is None:
                gt_labels_store = gt_labels
            
            logger.info(f"Completed {model_name} in {end_time - start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error testing model {model_name}: {str(e)}")
            results[model_name] = {"auc": None}
    
    # Print and save final results
    logger.info("\n=== FINAL RESULTS ===")
    sorted_results = sorted(results.items(), key=lambda x: -1 if x[1]["auc"] is None else x[1]["auc"], reverse=True)
    
    for model_name, metrics in sorted_results:
        if metrics["auc"] is not None:
            logger.info(f"{model_name}: AUC = {metrics['auc'] * 100:.1f}%")
        else:
            logger.info(f"{model_name}: Failed to test")
    
    # Save results to file
    results_filename = f"TEST_RESULTS_model_metrics_results_epoch{epoch_range}.txt"
    results_file = os.path.join(base_path, results_filename)
    with open(results_file, 'w') as f:
        f.write("=== MODEL METRICS RESULTS ===\n")
        f.write("Model Name, AUC (%)\n")
        for model_name, metrics in sorted_results:
            if metrics["auc"] is not None:
                f.write(f"{model_name}: AUC = {metrics['auc'] * 100:.1f}%\n")
            else:
                f.write(f"{model_name}: Failed to test\n")
    
    logger.info(f"Test results saved to: {results_file}")
    logger.info(f"All files saved in base path: {base_path}")

def main():
    args = parse_args()
    update_config(config, args)

    logger, final_output_dir, tb_log_dir = \
        log_util.create_logger(config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # Only batch testing mode
    batch_test_models(args.base_path, args.epoch_range, config, logger)

if __name__ == '__main__':
    main()