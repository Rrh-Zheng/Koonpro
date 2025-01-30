import argparse
import torch
from exp.exp_main import Exp_Main

parser = argparse.ArgumentParser(description='Koonpro for Time Series Forecasting')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_name', type=str, required=True, default='Koonpro', help='deployed model')
parser.add_argument('--data_name', type=str, required=True, default='etth1', help='select dataset')
parser.add_argument('--dims', type=int, default=7, required=True, help='dimension of data')
parser.add_argument('--time_shifts', type=int, default=30, help='time length in training stage')
parser.add_argument('--window', type=int, default=10, help='delayed embedding number')
parser.add_argument('--enc_width', type=list, default=[32, 128, 256, 128, 32], help='number of layers of encoder and decoder')
parser.add_argument('--aux_width', type=list, default=[32, 128, 256, 128, 32], help='number of layers of aux')
parser.add_argument('--dmd_width', type=list, default=[32, 128, 256], help='number of layers of dmd')
parser.add_argument('--delta_t', type=int, default=3, help='delta_t in linear shift')
parser.add_argument('--var_weight', type=float, default=0.01, help='weight of variance of angle shift in linear shift')
parser.add_argument('--var_weight_decoder', type=float, default=0.1, help='weight of variance of Decoder')
parser.add_argument('--use_prob', type=bool, default=True, help='use PDF or MSE')
parser.add_argument('--get_back', type=bool, default=True, help='predict full of delayed embedding or tail')
parser.add_argument('--predict_length', type=int, default=24, help='predict length of test set')
parser.add_argument('--seed', type=int, default=7, help='random seed')
parser.add_argument('--scale', type=bool, default=True, help='scale data or not')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--check_q', type=str, default='./q_context/', help='location of learned DMD-embedding')

# optimization
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--train_size', type=list, default=[0, 0.7], help='size of train dataset')
parser.add_argument('--val_size', type=list, default=[0.7, 0.9], help='size of validate dataset')
parser.add_argument('--test_size', type=list, default=[0.9, 1], help='size of test dataset')

#GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu:
    if args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    else:
        torch.cuda.set_device(args.gpu)
print('Args in experiment:')
print(args)

Exp = Exp_Main
dims = dims_dict[args.data_name]

if args.is_training:
    crps_sum_results = []
    nrmse_sum_results = []
    setting = '{}_{}_ts{}_ew{}_dt{}_vw{}_vwd{}_up_{}'.format(
        args.model_name,
        args.data_name,
        args.time_shifts,
        args.window,
        args.delta_t,
        args.var_weight,
        args.var_weight_decoder,
        args.use_prob)
    exp = Exp(args)
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)
    crps, nrmse = exp.metric(setting, 'test')
    print('crps_sum:', crps, 'nrmse_sum', nrmse)

else:
    ii = 0
    setting = '{}_{}_ts{}_ew{}_dt{}_vw{}_vwd{}_up_{}'.format(
        args.model_name,
        args.data_name,
        args.time_shifts,
        args.window,
        args.delta_t,
        args.var_weight,
        args.var_weight_decoder,
        args.use_prob
    )
    exp = Exp(args)
    crps, nrmse = exp.metric(setting, 'test')
    print('crps_sum:', crps, 'nrmse_sum', nrmse)
    exp.plot_interval(setting, 'test')
