from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import sys

from torch.backends import cudnn
import copy
import torch.nn as nn
import random

from reid import datasets
from reid.evaluators import Evaluator
from reid.utils.data import IterLoader
from reid.utils.data.sampler import RandomMultipleGallerySampler
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from reid.utils.lr_scheduler import WarmupMultiStepLR
from reid.utils.my_tools import *
from reid.models.resnet import build_resnet_backbone
from reid.models.layers import DataParallel
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def get_data(name, data_dir, height, width, batch_size, workers, num_instances):
    if name == "cuhk_sysu":
        root = osp.join(data_dir, "cuhksysu4reid")
    elif name == "msmt17":
        root = osp.join(data_dir, "MSMT17")
    else:
        root = osp.join(data_dir, name)

    dataset = datasets.create(name, root)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = sorted(dataset.train)

    iters = int(len(train_set) / batch_size)
    num_classes = dataset.num_train_pids

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None

    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir,transform=train_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True)

    init_loader = DataLoader(Preprocessor(train_set, root=dataset.images_dir,transform=test_transformer),
                             batch_size=128, num_workers=workers,shuffle=False, pin_memory=True, drop_last=False)

    return dataset, num_classes, train_loader, test_loader, init_loader

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):

    cudnn.benchmark = True
    log_name = 'step_1.txt'
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, log_name))
    else:
        log_dir = osp.dirname(args.resume)
        sys.stdout = Logger(osp.join(log_dir, log_name))
    print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    dataset_viper, num_classes_viper, train_loader_viper, test_loader_viper, _ = \
        get_data('viper', args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances)
    dataset_Grid, num_classes_Grid, train_loader_Grid, test_loader_Grid, _ = \
        get_data('Grid', args.data_dir, args.height, args.width, args.batch_size, args.workers,
                 args.num_instances)
    dataset_market, num_classes_market, train_loader_market, test_loader_market, init_loader_market = \
        get_data('market1501', args.data_dir, args.height, args.width, args.batch_size, args.workers,
                 args.num_instances)

    dataset_dukemtmc, num_classes_dukemtmc, train_loader_dukemtmc, test_loader_dukemtmc, init_loader_dukemtmc = \
        get_data('dukemtmc', args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances)

    dataset_cuhksysu, num_classes_cuhksysu, train_loader_cuhksysu, test_loader_cuhksysu, init_loader_chuksysu = \
        get_data('cuhk_sysu', args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances)

    dataset_msmt17, num_classes_msmt17, train_loader_msmt17, test_loader_msmt17, init_loader_msmt17 = \
        get_data('msmt17', args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances)

    dataset_cuhk03, num_classes_cuhk03, _, test_loader_cuhk03, _ = \
        get_data('CUHK03', args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances)

    dataset_cuhk02, num_classes_cuhk02, _, test_loader_cuhk02, _ = \
        get_data('CUHK02', args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances)

    dataset_prid, num_classes_prid, train_loader_prid, test_loader_prid, init_loader_prid = \
        get_data('prid2011', args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances)

    dataset_Duke, num_classes_Duke, _, test_loader_Duke, _ = \
        get_data('Occluded_Duke', args.data_dir, args.height, args.width, args.batch_size, args.workers,
                 args.num_instances)

    dataset_Reid, num_classes_Reid, _, test_loader_Reid, _ = \
        get_data('Occluded_REID', args.data_dir, args.height, args.width, args.batch_size, args.workers,
                 args.num_instances)

    # Create model
    num_classes_total = num_classes_market + num_classes_dukemtmc + num_classes_cuhksysu + num_classes_msmt17 + num_classes_cuhk03
    model = build_resnet_backbone(num_class=num_classes_total, depth='50x')
    model.cuda()
    model = DataParallel(model)
    evaluator = Evaluator(model)

    # Load checkpoints
    if args.resume_working:
        working_checkpoint = load_checkpoint(args.resume_working)
        copy_state_dict(working_checkpoint['state_dict'], model)


    epoch = working_checkpoint['epoch']
    
    # Setup evaluators
    _, mAP_market = evaluator.evaluate(test_loader_market, dataset_market.query, dataset_market.gallery,
                                       cmc_flag=True)
    print('Finished epoch {:3d}  Market-1501 mAP: {:5.1%}'.format(epoch, mAP_market))

    '''_, mAP_cuhksysu = evaluator.evaluate(test_loader_cuhksysu, dataset_cuhksysu.query,  dataset_cuhksysu.gallery, cmc_flag=True)

    print('Finished epoch {:3d}  CUHKSYSU mAP: {:5.1%}'.format(epoch, mAP_cuhksysu))

    _, mAP_duke = evaluator.evaluate(test_loader_dukemtmc, dataset_dukemtmc.query, dataset_dukemtmc.gallery, cmc_flag=True)

    print('Finished epoch {:3d}  DukeMTMC mAP: {:5.1%}'.format(epoch, mAP_duke))

    _, mAP_msmt = evaluator.evaluate(test_loader_msmt17, dataset_msmt17.query, dataset_msmt17.gallery, cmc_flag=True)
    print('Finished epoch {:3d}  MSMT17 mAP: {:5.1%}'.format(epoch, mAP_msmt))

    _, mAP_cuhk03 = evaluator.evaluate(test_loader_cuhk03, dataset_cuhk03.query, dataset_cuhk03.gallery,
                                       cmc_flag=True)
    print('Finished epoch {:3d}  CUHK03 mAP: {:5.1%}'.format(epoch, mAP_cuhk03))'''

    _, mAP_viper = evaluator.evaluate(test_loader_viper, dataset_viper.query, dataset_viper.gallery, cmc_flag=True)
    print('Finished epoch {:3d}  viper mAP: {:5.1%}'.format(epoch, mAP_viper))

    _, mAP_Grid = evaluator.evaluate(test_loader_Grid, dataset_Grid.query, dataset_Grid.gallery, cmc_flag=True)
    print('Finished epoch {:3d}  Grid mAP: {:5.1%}'.format(epoch, mAP_Grid))

    _, mAP_cuhk02 = evaluator.evaluate(test_loader_cuhk02, dataset_cuhk02.query, dataset_cuhk02.gallery, cmc_flag=True)
    print('Finished epoch {:3d}  cuhk02 mAP: {:5.1%}'.format(epoch, mAP_cuhk02))

    _, mAP_prid = evaluator.evaluate(test_loader_prid, dataset_prid.query, dataset_prid.gallery, cmc_flag=True)
    print('Finished epoch {:3d}  prid mAP: {:5.1%}'.format(epoch, mAP_prid))

    _, mAP_Duke = evaluator.evaluate(test_loader_Duke, dataset_Duke.query, dataset_Duke.gallery, cmc_flag=True)
    print('Finished epoch {:3d}  Duke mAP: {:5.1%}'.format(epoch, mAP_Duke))

    _, mAP_Reid = evaluator.evaluate(test_loader_Reid, dataset_Reid.query, dataset_Reid.gallery, cmc_flag=True)
    print('Finished epoch {:3d}  Reid mAP: {:5.1%}'.format(epoch, mAP_Reid))

    print('finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Continual training for lifelong person re-identification")
    # data
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-br', '--replay-batch-size', type=int, default=128)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # model
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate of new parameters, for pretrained ")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int, default=[40, 70],
                        help='milestones for the learning rate decay')
    # training configs
    parser.add_argument('--resume-working', type=str, default='***', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=200)
    parser.add_argument('--margin', type=float, default=0.3, help='margin for the triplet loss with batch hard')
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='***')
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'order1'))
    parser.add_argument('--rr-gpu', action='store_true',
                        help="use GPU for accelerating clustering")
    main()