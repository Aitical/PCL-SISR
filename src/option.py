import argparse
import template

parser = argparse.ArgumentParser(description='EDSR and MDSR')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--template', default='.',
                    help='You can set various templates in option.py')

parser.add_argument('--proj_name', default='EDSR',
                    help='You can set various templates in option.py')
# Log board
parser.add_argument('--use_wandb', action='store_true')
parser.add_argument('--wandb_key', default='')
parser.add_argument('--wandb_offline', action='store_true')
parser.add_argument('--wandb_dir', default='../experiment/wandb')
parser.add_argument('--swinir_arch', default='standard', choices=['standard', 'lightweight'])

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=6,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--exp_name',
                    help='number of residual groups')
parser.add_argument('--lpips_spatial', action='store_true')
parser.add_argument('--mcl_neg', type=int, default=3)
parser.add_argument('--shuffle_p', type=float, default=0.)
parser.add_argument('--shuffle_pos', action='store_true')
parser.add_argument('--only_blur', action='store_true')
parser.add_argument('--neg_sr', action='store_true')
parser.add_argument('--ema_iter', default=10, type=int)
parser.add_argument('--use_ema', action='store_true')
parser.add_argument('--random_neg', action='store_true')
parser.add_argument('--layer_weight', action='store_true')
parser.add_argument('--cl_layer', type=str, default='0,2,4,6')
parser.add_argument('--sharp_hr', action='store_true')
parser.add_argument('--cl_loss_type', type=str, default='cosine')
parser.add_argument('--before_relu', action='store_true')
parser.add_argument('--multi_neg_D', action='store_true')
parser.add_argument('--vgg_like_relu', action='store_true')

parser.add_argument('--WaveD_path', type=str, default='loss.pt')
parser.add_argument('--CLD_path', type=str, default='loss_d.pt')
parser.add_argument('--sharp_range', default=5, type=float)
parser.add_argument('--sharp_value', default=0.5, type=float)
# Data specifications
parser.add_argument('--dir_data', type=str, default='../../../dataset',
                    help='dataset directory')
parser.add_argument('--dir_demo', type=str, default='../test',
                    help='demo image directory')
parser.add_argument('--multi_neg', action='store_true')
parser.add_argument('--data_train', type=str, default='DIV2K',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='DIV2K',
                    help='test dataset name')
parser.add_argument('--data_range', type=str, default='1-800/801-810',
                    help='train/test data range')
parser.add_argument('--ext', type=str, default='sep',
                    help='dataset file extension')
parser.add_argument('--scale', type=str, default='4',
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=192,
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--chop', action='store_true',
                    help='enable memory-efficient forward')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')

# Model specifications
parser.add_argument('--model', default='EDSR',
                    help='model name')

parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--pre_train', type=str, default='',
                    help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=16,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--dilation', action='store_true',
                    help='use dilated convolution')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')

# Option for Residual dense network (RDN)
parser.add_argument('--G0', type=int, default=64,
                    help='default number of filters. (Use in RDN)')
parser.add_argument('--RDNkSize', type=int, default=3,
                    help='default kernel size. (Use in RDN)')
parser.add_argument('--RDNconfig', type=str, default='B',
                    help='parameters config of RDN. (Use in RDN)')

# Option for Residual channel attention network (RCAN)
parser.add_argument('--n_resgroups', type=int, default=10,
                    help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16,
                    help='number of feature maps reduction')

# Option for CARN
parser.add_argument('--carn_group', default=1, type=int)


# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')
parser.add_argument('--scheduler', choices=['Cosine', 'MileStone', 'CosineRestart'], default='MileStone')
parser.add_argument('--cosine_tmax', default=150, type=int, help='Epochs for Cosine scheduler which will *1000 for iterations')
parser.add_argument('--cosine_etamin', default=1e-7, type=float, help='Min value for cosine scheduler')
parser.add_argument('--cosine_restart_weight', default=0.5, type=float)
parser.add_argument('--gpu_blur', action='store_true', help='use kornia\'s filter with GPU')
parser.add_argument('--gpu_sharp', action='store_true', help='use kornia\'s sharpness with GPU')
parser.add_argument('--use_noise_pos', action='store_true', help='random noise to build pos samples')
parser.add_argument('--pos_noise_value', type=int, default=2, help='random noise scale ([0, 255])')
parser.add_argument('--use_aug', action='store_true', help='build positive and negative samples')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--decay', type=str, default='200',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--recursive', type=int, default=4)
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--gclip', type=float, default=0,
                    help='gradient clipping threshold (0 = no clipping)')
parser.add_argument('--use_wavelet', action='store_true')
parser.add_argument('--use_fft', action='store_true')
parser.add_argument('--contras_D_train', action='store_true')
parser.add_argument('--shuffle_neg', action='store_true', help='Used in CLD in loss/discriminator.py, shuffle idx in batch')
parser.add_argument('--pos_id', type=int, default=-1)
parser.add_argument('--neg_id', type=int, default=-1)
parser.add_argument('--only_aug', action='store_true')
parser.add_argument('--shuffle_neg_num', type=int, default=1)
parser.add_argument('--cl_gan_cl_weight', type=float, default=1)
# Loss specifications
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e8',
                    help='skipping batch that has large error')
parser.add_argument('--no_ad_loss', action='store_true')
# Log specifications
parser.add_argument('--save', type=str, default='test',
                    help='file name to save')
parser.add_argument('--load', type=str, default='',
                    help='file name to load')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true',
                    help='save output results')
parser.add_argument('--save_gt', action='store_true',
                    help='save low-resolution and high-resolution images together')
parser.add_argument('--nlsn_n_hashes', type=int, default=4)
parser.add_argument('--nlsn_chunk_size', type=int, default=144)
parser.add_argument('--nlsn_dilation', action='store_true')

args = parser.parse_args()
template.set_template(args)

args.scale = list(map(lambda x: int(x), args.scale.split('+')))
args.data_train = args.data_train.split('+')
args.data_test = args.data_test.split('+')

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

