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


def train(config, train_loader, model, loss_functions, optimizer, epoch, logger):
    loss_func_mse = nn.MSELoss(reduction='none')
    entropy_loss_func = EntropyLossEncap().to("cuda")
    model.train()

    # Khởi tạo biến để tính average loss
    total_loss = 0.0
    num_batches = 0

    for i, data in enumerate(train_loader):
        # decode input
        inputs, target = train_util.decode_input(input=data, train=True)
        output, att = model(inputs)
        entropy_loss = 0
        for att_step in att:
            entropy_loss += entropy_loss_func(att_step)

        # compute loss
        target = target.cuda(non_blocking=True)
        inte_loss, grad_loss, msssim_loss, l2_loss = loss_functions(output, target)
        loss = inte_loss + grad_loss + msssim_loss + l2_loss + 0.0002*entropy_loss

        # Cộng dồn loss để tính average
        total_loss += loss.item()
        num_batches += 1

        # compute PSNR
        mse_imgs = torch.mean(loss_func_mse((output + 1) / 2, (target + 1) / 2)).item()
        psnr = anomaly_util.psnr_park(mse_imgs)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cur_lr = optimizer.param_groups[0]['lr']
        if (i + 1) % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Lr {lr:.6f}\t' \
                  '[inte {inte:.5f} + grad {grad:.4f} + msssim {msssim:.4f} + L2 {l2:.4f} + EL {entropy:.4f}]\t' \
                  'PSNR {psnr:.2f}'.format(epoch+1, i+1, len(train_loader),
                                             lr=cur_lr,
                                             inte=inte_loss, grad=grad_loss, msssim=msssim_loss, l2=l2_loss, entropy = entropy_loss,
                                             psnr=psnr)
            logger.info(msg)

    # Tính và trả về average loss của epoch
    avg_loss = total_loss / num_batches
    logger.info('Epoch [{0}] completed. Average Loss: {1:.6f}'.format(epoch + 1, avg_loss))
    return avg_loss


if __name__ == '__main__':
    main()