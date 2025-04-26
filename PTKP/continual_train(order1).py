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
from reid.trainer import Trainer
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

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


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


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
    log_name = 'log.txt'
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, log_name))
    else:
        log_dir = osp.dirname(args.resume)
        sys.stdout = Logger(osp.join(log_dir, log_name))
    print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders  order1
    dataset_market, num_classes_market, train_loader_market, test_loader_market, _ = \
        get_data('market1501', args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances)

    dataset_cuhksysu, num_classes_cuhksysu, train_loader_cuhksysu, test_loader_cuhksysu, init_loader_chuksysu = \
        get_data('cuhk_sysu', args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances)

    dataset_duke, num_classes_duke, train_loader_duke, test_loader_duke, init_loader_duke = \
        get_data('dukemtmc', args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances)

    dataset_msmt17, num_classes_msmt17, train_loader_msmt17, test_loader_msmt17, init_loader_msmt17 = \
        get_data('msmt17', args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances)

    # Data loaders for test only
    dataset_cuhk03, num_classes_cuhk03, train_loader_cuhk03, test_loader_cuhk03, init_loader_cuhk03 = \
        get_data('CUHK03', args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances)

    # Create model
    model = build_resnet_backbone(num_class=num_classes_market, depth='50x')
    model.cuda()
    model = DataParallel(model)

    # Load from checkpoint
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        copy_state_dict(checkpoint['state_dict'], model)
        start_epoch = checkpoint['epoch']
        best_mAP = checkpoint['best_mAP']
        print("=> Start epoch {}  best mAP {:.1%}".format(start_epoch, best_mAP))

    # Evaluator
    start_epoch = 0
    evaluator = Evaluator(model)

    # Opitimizer initialize
    params = []
    for key, value in model.named_params(model):
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
    optimizer = torch.optim.Adam(params)
    lr_scheduler = WarmupMultiStepLR(optimizer, [40, 70], gamma=0.1, warmup_factor=0.01, warmup_iters=args.warmup_step)

    # Start training
    print('Continual training starts!')

    # Train Market-1501
    trainer = Trainer(model, num_classes_market, margin=args.margin)
    for epoch in range(start_epoch, 2):

        train_loader_market.new_epoch()
        trainer.train(epoch, train_loader_market, None, optimizer, training_phase=1,
                      train_iters=len(train_loader_market), add_num=0, old_model=None, replay=False)
        lr_scheduler.step()

        if ((epoch + 1) % 2 == 0):
            _, mAP_market = evaluator.evaluate(test_loader_market, dataset_market.query, dataset_market.gallery, cmc_flag=True)

            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
            }, True, fpath=osp.join(args.logs_dir, 'market_checkpoint.pth.tar'))

            print('Finished epoch {:3d}  Market-1501 mAP: {:5.1%} '.format(epoch, mAP_market))


    # Select replay data of market-1501
    replay_dataloader, market_replay_dataset = select_replay_samples(model, dataset_market, training_phase=1)

    # Expand the dimension of classifier
    org_classifier_params = model.module.classifier.weight.data
    model.module.classifier = nn.Linear(2048, num_classes_cuhksysu + num_classes_market, bias=False)
    model.cuda()
    model.module.classifier.weight.data[:num_classes_market].copy_(org_classifier_params)
    add_num = num_classes_market

    # Create old frozen model
    old_model = copy.deepcopy(model)
    old_model = old_model.cuda()
    old_model.eval()

    # Initialize classifer with class centers
    class_centers = initial_classifier(model, init_loader_chuksysu)
    model.module.classifier.weight.data[num_classes_market:].copy_(class_centers)

    # Re-initialize optimizer
    params = []
    for key, value in model.named_params(model):
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": args.lr * 0.1, "weight_decay": args.weight_decay}]
    optimizer = torch.optim.Adam(params)
    lr_scheduler = WarmupMultiStepLR(optimizer, [30], gamma=0.1, warmup_factor=0.01, warmup_iters=args.warmup_step)

    trainer = Trainer(model, num_classes_cuhksysu + num_classes_market, margin=args.margin)

    for epoch in range(start_epoch, args.epochs):

        train_loader_cuhksysu.new_epoch()
        trainer.train(epoch, train_loader_cuhksysu, replay_dataloader, optimizer, training_phase=2,
                      train_iters=len(train_loader_cuhksysu), add_num=add_num, old_model=old_model, replay=True)
        lr_scheduler.step()

        if ((epoch + 1) % args.epochs == 0):

            _, mAP_cuhksysu = evaluator.evaluate(test_loader_cuhksysu, dataset_cuhksysu.query, dataset_cuhksysu.gallery, cmc_flag=True)

            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
            }, True, fpath=osp.join(args.logs_dir, 'cuhksysu_checkpoint.pth.tar'))

            print('Finished epoch {:3d}  CUHKSYSU mAP: {:5.1%}'.format(epoch, mAP_cuhksysu))

    # Select replay data of CUHKSYSU
    replay_dataloader, cuhksysu_replay_dataset = select_replay_samples(model, dataset_cuhksysu, training_phase=2,
                                                  add_num=add_num, old_datas=market_replay_dataset)

    # Expand the dimension of classifier
    org_classifier_params = model.module.classifier.weight.data
    model.module.classifier = nn.Linear(2048, num_classes_cuhksysu + num_classes_market + num_classes_duke, bias=False)
    model.module.classifier.weight.data[:(num_classes_market + num_classes_cuhksysu)].copy_(org_classifier_params)
    model.cuda()
    add_num = num_classes_market + num_classes_cuhksysu

    # Initialize classifer with class centers
    class_centers = initial_classifier(model, init_loader_duke)
    model.module.classifier.weight.data[(num_classes_market + num_classes_cuhksysu):].copy_(class_centers)
    model.cuda()

    # Create old frozen model
    old_model = copy.deepcopy(model)
    old_model = old_model.cuda()
    old_model.eval()

    # Re-initialize optimizer
    params = []
    for key, value in model.named_params(model):
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": args.lr * 0.1, "weight_decay": args.weight_decay}]
    optimizer = torch.optim.Adam(params)
    lr_scheduler = WarmupMultiStepLR(optimizer, [30], gamma=0.1, warmup_factor=0.01, warmup_iters=args.warmup_step)

    trainer = Trainer(model, num_classes_duke + add_num, margin=args.margin)

    for epoch in range(start_epoch, args.epochs):

        train_loader_duke.new_epoch()
        trainer.train(epoch, train_loader_duke, replay_dataloader, optimizer, training_phase=3,
                      train_iters=len(train_loader_duke), add_num=add_num, old_model=old_model, replay=True)
        lr_scheduler.step()

        if ((epoch + 1) % args.epochs == 0):

            _, mAP_duke = evaluator.evaluate(test_loader_duke, dataset_duke.query, dataset_duke.gallery, cmc_flag=True)

            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
            }, True, fpath=osp.join(args.logs_dir, 'dukemtmc_checkpoint.pth.tar'))

            print('Finished epoch {:3d}  DukeMTMC mAP: {:5.1%}'.format(epoch, mAP_duke))


    # Select replay data of DukeMTMC
    replay_dataloader, duke_replay_dataset = select_replay_samples(model, dataset_duke, training_phase=3,
                                                     add_num=add_num,old_datas=cuhksysu_replay_dataset)

    # Expand the dimension of classifier
    org_classifier_params = model.module.classifier.weight.data
    model.module.classifier = nn.Linear(2048, num_classes_cuhksysu + num_classes_market +
                                        num_classes_duke + num_classes_msmt17, bias=False)
    model.module.classifier.weight.data[:num_classes_market + num_classes_cuhksysu
                                         + num_classes_duke].copy_(org_classifier_params)
    add_num = num_classes_market + num_classes_cuhksysu + num_classes_duke
    model.cuda()

    # Initialize classifer with class centers
    class_centers = initial_classifier(model, init_loader_msmt17)
    model.module.classifier.weight.data[(num_classes_market + num_classes_cuhksysu
                                         + num_classes_duke):].copy_(class_centers)
    model.cuda()

    # Create old frozen model
    old_model = copy.deepcopy(model)
    old_model = old_model.cuda()
    old_model.eval()

    # Re-initialize optimizer
    params = []
    for key, value in model.named_params(model):
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": args.lr * 0.1, "weight_decay": args.weight_decay}]
    optimizer = torch.optim.Adam(params)
    lr_scheduler = WarmupMultiStepLR(optimizer, [20, 30], gamma=0.1, warmup_factor=0.01, warmup_iters=args.warmup_step)

    trainer = Trainer(model, num_classes_msmt17 + add_num, margin=args.margin)

    for epoch in range(start_epoch, args.epochs):

        train_loader_msmt17.new_epoch()
        trainer.train(epoch, train_loader_msmt17, replay_dataloader, optimizer, training_phase=4,
                      train_iters=len(train_loader_msmt17), add_num=add_num, old_model=old_model, replay=True)
        lr_scheduler.step()

        if ((epoch + 1) % args.epochs == 0):

            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                #'mAP': mAP_msmt,
            }, True, fpath=osp.join(args.logs_dir, 'msmt17_checkpoint.pth.tar'))

            #print('Finished epoch {:3d}  MSMT17 mAP: {:5.1%}'.format(epoch, mAP_msmt))


        # Select replay data of MSMT17
        replay_dataloader, msmt17_replay_dataset = select_replay_samples(model, dataset_msmt17, training_phase=3,
                                                                       add_num=add_num,
                                                                       old_datas=duke_replay_dataset)

        # Expand the dimension of classifier
        org_classifier_params = model.module.classifier.weight.data
        model.module.classifier = nn.Linear(2048, num_classes_cuhksysu + num_classes_market +
                                            num_classes_duke + num_classes_msmt17  + num_classes_cuhk03, bias=False)
        model.module.classifier.weight.data[:num_classes_market + num_classes_cuhksysu
                                             + num_classes_duke + num_classes_msmt17].copy_(org_classifier_params)
        add_num = num_classes_market + num_classes_cuhksysu + num_classes_duke + num_classes_msmt17
        model.cuda()

        # Initialize classifer with class centers
        class_centers = initial_classifier(model, init_loader_cuhk03)
        model.module.classifier.weight.data[(num_classes_market + num_classes_cuhksysu
                                             + num_classes_duke + num_classes_msmt17):].copy_(class_centers)
        model.cuda()

        # Create old frozen model
        old_model = copy.deepcopy(model)
        old_model = old_model.cuda()
        old_model.eval()

        # Re-initialize optimizer
        params = []
        for key, value in model.named_params(model):
            if not value.requires_grad:
                continue
            params += [{"params": [value], "lr": args.lr * 0.1, "weight_decay": args.weight_decay}]
        optimizer = torch.optim.Adam(params)
        lr_scheduler = WarmupMultiStepLR(optimizer, [20, 30], gamma=0.1, warmup_factor=0.01,
                                         warmup_iters=args.warmup_step)

        trainer = Trainer(model, num_classes_cuhk03 + add_num, margin=args.margin)

        for epoch in range(start_epoch, args.epochs):

            train_loader_cuhk03.new_epoch()
            trainer.train(epoch, train_loader_cuhk03, replay_dataloader, optimizer, training_phase=4,
                          train_iters=len(train_loader_cuhk03), add_num=add_num, old_model=old_model, replay=True)
            lr_scheduler.step()

            if ((epoch + 1) % args.epochs == 0):
                _, mAP_cuhk03= evaluator.evaluate(test_loader_cuhk03, dataset_cuhk03.query, dataset_cuhk03.gallery, cmc_flag=True)

                save_checkpoint({
                    'state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                }, True, fpath=osp.join(args.logs_dir, 'cuhk03_checkpoint.pth.tar'))

                print('Finished epoch {:3d}  CUHK03 mAP: {:5.1%}'.format(epoch, mAP_cuhk03))

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
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
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
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--rr-gpu', action='store_true',
                        help="use GPU for accelerating clustering")
    main()