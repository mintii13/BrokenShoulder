import pprint
import argparse
import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import datasets
from utils import train_util, log_util, anomaly_util
from config.defaults import _C as config, update_config
from models.wresnet1024_cattn_tsm import ASTNet as get_net1
from models.wresnet2048_multiscale_cattn_tsmplus_layer6 import ASTNet as get_net2

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# --cfg experiments/sha/sha_wresnet.yaml --model-file output/shanghai/sha_wresnet/shanghai.pth GPUS [3]
# --cfg experiments/ped2/ped2_wresnet.yaml --model-file output/ped2/ped2_wresnet/ped2.pth GPUS [3]
def parse_args():
    parser = argparse.ArgumentParser(description='Test Anomaly Detection')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        default='config/shanghaitech_wresnet.yaml', type=str)
    parser.add_argument('--model-file', help='model parameters',
                        default='D:/FPTU-sourse/Term4/ResFes_FE/References/Anomaly Detection/Reconstruction/astnet/ASTNet/output/ped2/ped2_wresnet/epoch_8.pth', type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args


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

    config.defrost()
    config.MODEL.INIT_WEIGHTS = False       # TODO ? False
    config.freeze()

    gpus = [(config.GPUS[0])]
    # model = models.get_net(config)
    if config.DATASET.DATASET == "ped2":
        model = get_net1(config, pretrained=False)
    else:
        model = get_net2(config, pretrained=False)
    logger.info('Model: {}'.format(model.get_name()))
    model = nn.DataParallel(model, device_ids=gpus).cuda(device=gpus[0])
    logger.info('Epoch: '.format(args.model_file))

    # load model
    state_dict = torch.load(args.model_file)
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
    else:
        model.module.load_state_dict(state_dict)

    test_dataset = eval('datasets.get_test_data')(config)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    mat_loader = datasets.get_label(config)
    mat = mat_loader()

    psnr_list = inference(config, test_loader, model)
    assert len(psnr_list) == len(mat), f'Ground truth has {len(mat)} videos, BUT got {len(psnr_list)} detected videos!'

    auc, fpr, tpr = anomaly_util.calculate_auc(config, psnr_list, mat)
    print(psnr_list)
    logger.info(f'AUC: {auc * 100:.1f}%')


def inference(config, data_loader, model):
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
                # https://github.com/cvlab-yonsei/MNAD/blob/d6d1e446e0ed80765b100d92e24f5ab472d27cc3/utils.py#L20
                mse_imgs = torch.mean(loss_func_mse((output[0] + 1) / 2, (target[0] + 1) / 2)).item()
                psnr = anomaly_util.psnr_park(mse_imgs)
                psnr_video.append(psnr)

            psnr_list.append(psnr_video)
    return psnr_list




import matplotlib.pyplot as plt
def plot_regularity_score(psnr_values, gt_labels, video_idx, epoch_name):
    plt.figure(figsize=(12, 6))
    
    # Plot PSNR values (regularity score)
    plt.plot(psnr_values, 'b-', label='Regularity Score (PSNR)')
    
    # Tạo trục y thứ hai cho ground truth
    ax2 = plt.twinx()
    ax2.plot(gt_labels, 'r-', label='Ground Truth')
    ax2.set_ylim([-0.1, 1.1])
    ax2.set_ylabel('Anomaly (1 = anomalous)')
    
    plt.title(f'Regularity Score - Video {video_idx + 1} - {epoch_name}')
    plt.xlabel('Frame Number')
    plt.ylabel('Regularity Score (PSNR)')
    
    # Kết hợp legend
    lines1, labels1 = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'regularity_video{video_idx + 1}_{epoch_name}.png')
    plt.show()

if __name__ == '__main__':
    main()
    # mat_loader = datasets.get_label(config)
    # mat = mat_loader()
    # psnr = [34.87209515742202, 35.088338116535304, 34.95834550697249, 35.12151408551995, 35.334627897744625, 35.13821265467967, 35.064983921991754, 35.17008454073597, 34.66563941669558, 35.035528116467766, 35.088825192786814, 35.103113789866555, 34.88422235538026, 35.218789018013986, 35.10881954843298, 35.133066555134576, 35.25958012623931, 35.20126169726089, 35.0622591095916, 35.04499241457887, 35.02634376596514, 35.174079666441386, 35.06232074170264, 34.89109146107284, 35.04058366510755, 34.97299655817999, 35.17127932479283, 34.91456627448549, 35.077538986158615, 35.068996782736214, 35.300564385502234, 34.993390912929925, 34.94340677748702, 35.18215723795143, 35.13587205023977, 35.089443295314965, 35.283863349552874, 34.90932327596842, 34.78099188343299, 34.774041126935586, 35.17410171699014, 35.10698466381612, 35.257347085226975, 35.08969586549283, 34.881870754744504, 34.96907892877382, 35.46609375124123, 35.26118349934799, 35.27531476307908, 35.43609137916598, 35.04322708585237, 34.90507312804657, 34.96929601913278, 34.97282535854712, 34.79458983381778, 35.08498065339568, 34.4282185256782, 34.94170433890869, 34.91370414864116, 34.78462928527812, 34.97668474819717, 35.034027262260764, 35.22482316089695, 34.98354728621009, 35.09463964957165, 34.83880954395831, 34.691733331412635, 34.736586946929165, 34.956763047201946, 34.90461678670518, 35.10817901269057, 34.987375568721426, 35.016531974933926, 34.620175267113204, 34.440731926413896, 34.5881992151171, 34.52123925984792, 34.38612227474148, 34.32033956162022, 34.39967763800061, 33.9778006294564, 34.082916691396925, 34.503468326251415, 34.413212589501796, 34.48244055271607, 34.481112754888166, 34.4743201313313, 34.40326314045021, 34.35027730001175, 34.37694028764659, 34.63532947128941, 34.341857641864806, 34.1840647854408, 34.324493699363366, 34.66478979067138, 34.552581876876786, 34.47233920868408, 34.24712482377052, 34.48831994972985, 34.41590144872325, 34.334145820184865, 34.360060085326495, 34.3449814767273, 34.16597316927734, 33.86656467540371, 33.87336135586785, 33.89106507976676, 34.00451780838318, 34.21042542706938, 34.23239141716, 34.27331461753715, 34.13362074585319, 34.17406990850588, 34.18427612081008, 34.346075057731326, 34.24743638417394, 33.923823473041374, 34.19189569704457, 34.405310739598164, 34.370883559053816, 34.44782167900425, 34.29322654782352, 34.26298292901464, 34.18718783188921, 34.2303109017903, 34.078036435946984, 33.855958317185774, 33.88834646347042, 33.570512155247144, 33.710366532069315, 33.69263415723028, 33.80265170385407, 33.96233109600829, 34.03075058643973, 34.038646933678066, 33.7521509750501, 33.680074632054804, 33.85264466608676, 33.86363765945702, 33.76429724831861, 33.46126354510832, 33.527089854169034, 33.63816487218602, 33.558153556681226, 33.38949911450075, 33.48898350015552, 33.594382687765304, 33.53053824535367, 33.52222529044115, 33.435951644320355, 33.476154480881874, 33.48144101481948, 33.14888774823697, 33.24857437992747, 33.40190888639847, 33.42680221032118, 33.459231574244384, 33.27097985329872, 33.22581340639031, 33.17320634174791, 33.2383602709479, 33.002477305550876, 33.1690232471548, 33.175381064604906, 32.81703038453513, 33.17379323111404, 33.01218953370687, 32.873511556613465, 32.94777328501803, 32.8167566874051, 33.10981666862473, 33.04003824050855, 33.23313062151535, 33.008920227215434, 33.20595266262652, 33.22719236778477]
    # plot_regularity_score(psnr,mat[0],0,"epoch8")