import ast
import torch
import copy
from torch.cuda.amp import autocast
from pkd.utils import set_random_seed, time_now
from pkd.core import BasePatchKD
from pkd.data_loader import IncrementalReIDLoaders
from pkd.visualization import visualize, Logger, VisdomPlotLogger, VisdomFeatureMapsLogger
from pkd.operation import train_p_s_an_epoch, fast_test_p_s

def main(config):
    set_random_seed(config.seed)
    # init loaders and base
    loaders = IncrementalReIDLoaders(config)
    base = BasePatchKD(config, loaders)

    # init logger
    print(base.output_dirs_dict['logs'])
    logger = Logger(os.path.join(base.output_dirs_dict['logs'], 'log.txt'))
    logger(config)

    assert config.mode in ['train', 'test', 'visualize']
    if config.mode == 'train':  # train mode
        # initialize visdom logger under training mode
        if config.visdom:
            visdom_dict = {
                'feature_maps': VisdomFeatureMapsLogger('image', pad_value=1, nrow=8, port=config.visdom_port,
                                                        env=config.running_time, opts={'title': f'featuremaps'})
            }

        # automatically resume model from the latest one
        if config.auto_resume_training_from_lastest_steps:
            start_train_step, start_train_epoch = base.resume_last_model()
        # continual loop
        for current_step in range(start_train_step, loaders.total_step):
            current_total_train_epochs = config.total_continual_train_epochs if current_step > 0 else config.total_train_epochs
            if current_step > 0:
                logger(f'save_and_frozen old model in {current_step}')
                old_model = base.copy_model_and_frozen(model_name='tasknet')
            else:
                old_model = None
            for current_epoch in range(start_train_epoch, current_total_train_epochs):
                result_dict = {}
                # save model
                base.save_model(current_step, current_epoch)
                # train
                str_lr, dict_lr = base.get_current_learning_rate()
                logger(str_lr)
                results = train_p_s_an_epoch(config, base, loaders, current_step, old_model, current_epoch, output_featuremaps=config.output_featuremaps)

                if config.output_featuremaps and len(results) == 3:
                    results_dict, results_str, heatmaps = results
                    if config.visdom:
                        visdom_dict['feature_maps'].images(heatmaps)
                else:
                    results_dict, results_str = results
                logger('Time: {};  Step: {}; Epoch: {};  {}'.format(time_now(), current_step, current_epoch, results_str))

                if config.test_frequency > 0 and current_epoch % config.test_frequency == 0:
                    with autocast(config.fp_16):
                        rank_map_dict, rank_map_str = fast_test_p_s(config, base, loaders, current_step, if_test_forget=config.if_test_forget)
                    logger(
                        f'Time: {time_now()}; Test Dataset: {config.test_dataset}: {rank_map_str}')
                    result_dict.update(rank_map_dict)

                if current_epoch == config.total_train_epochs - 1:
                    # test
                    # base.save_model(current_step, config.total_train_epochs)
                    with autocast(config.fp_16):
                        rank_map_dict, rank_map_str = fast_test_p_s(config, base, loaders, current_step, if_test_forget=config.if_test_forget)
                    logger(
                        f'Time: {time_now()}; Step: {current_step}; Epoch: {current_epoch} Test Dataset: {config.test_dataset}, {rank_map_str}')
                    print(f'Current step {current_step} is finished.')
                    start_train_epoch = 0
                    result_dict.update(rank_map_dict)

                if config.visdom:
                    result_dict.update(results_dict)
                    result_dict.update(dict_lr)
                    if current_step > 0:
                        global_current_epoch = current_epoch + (current_step-1) * current_total_train_epochs + config.total_train_epochs
                    else:
                        global_current_epoch = current_epoch
                    for name, value in result_dict.items():
                        if name in visdom_dict.keys():
                            visdom_dict[name].log(global_current_epoch, value, name=str(current_step))
                        else:
                            visdom_dict[name] = VisdomPlotLogger('line', port=config.visdom_port, env=config.running_time,
                                                                 opts={'title': f'train {name}'})
                            visdom_dict[name].log(global_current_epoch, value, name=str(current_step))

            if current_step > 0:
                del old_model
            base.save_model(current_step, current_epoch + 1)

    elif config.mode == 'test':	# test mode
        base.resume_from_model(config.resume_test_model)
        current_step = loaders.total_step - 1
        with autocast(config.fp_16):
            rank_map_dict, rank_map_str = fast_test_p_s(config, base, loaders, current_step, if_test_forget=config.if_test_forget)
            logger(
                f'Time: {time_now()}; Step: {current_step}; Test Dataset: {config.test_dataset}, {rank_map_str}')
            print(f'Current step {current_step} is finished.')

    elif config.mode == 'visualize':  # visualization mode
        base.resume_from_model(config.resume_visualize_model)
        visualize(config, base, loaders)


if __name__ == '__main__':
    import time
    import argparse
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

    running_time = time.strftime('%Y-%m-%d-%H-%M-%S')
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fp_16', type=bool, default=True)
    parser.add_argument('--running_time', type=str, default=running_time)
    parser.add_argument('--visdom', type=bool, default=False)
    parser.add_argument('--visdom_port', type=int, default=8097)
    parser.add_argument('--cuda', type=str, default='cuda')
    parser.add_argument('--mode', type=str, default='test', help='trian_10, train_5, train, test or visualize')
    parser.add_argument('--output_path', type=str, default=f'results/{running_time}', help='path to save related informations')
    parser.add_argument('--continual_step', type=str, default='5',
                        help='10 or 5 or task')
    parser.add_argument('--num_identities_per_domain', type=int, default=500,
                        help='250 for 10 steps, 500 for 5 steps, -1 for all aviliable identities')
    parser.add_argument('--joint_train', type=bool, default=False,
                        help='joint all dataset')
    parser.add_argument('--re_init_lr_scheduler_per_step', type=bool, default=False,
                        help='after_previous_step if re_init_optimizers')
    parser.add_argument('--warmup_lr', type=bool, default=False,
                        help='0-10 epoch warmup')

    # dataset configuration
    machine_dataset_path = '***'
    parser.add_argument('--datasets_root', type=str, default=machine_dataset_path, help='mix/market/duke/')
    parser.add_argument('--combine_all', type=ast.literal_eval, default=False, help='train+query+gallery as train')
    parser.add_argument('--train_dataset', nargs='+', type=str,
                        default=['market', 'cuhksysu', 'duke', 'msmt17', 'cuhk03'])
    '''parser.add_argument('--test_dataset', nargs='+', type=str,
                        default=['market', 'cuhksysu', 'duke', 'cuhk03', 'allgeneralizable'])'''
    parser.add_argument('--test_dataset', nargs='+', type=str,
                        default=['market', 'cuhksysu', 'duke', 'msmt17', 'cuhk03'])

    parser.add_argument('--image_size', type=int, nargs='+', default=[256, 128])
    parser.add_argument('--test_batch_size', type=int, default=64, help='test batch size')
    parser.add_argument('--p', type=int, default=32, help='person count in a batch')
    parser.add_argument('--k', type=int, default=4, help='images count of a person in a batch')
    parser.add_argument('--use_local_label4validation', type=bool, default=True,
                        help='validation use global pid label or not')

    # data augmentation
    parser.add_argument('--use_rea', type=ast.literal_eval, default=True)
    parser.add_argument('--use_colorjitor', type=ast.literal_eval, default=False)

    # model configuration
    parser.add_argument('--pid_num', type=int, default=2500, help='#domain times #pid per domain')

    # train configuration
    parser.add_argument('--steps', type=int, default=150, help='150 for 5s32p4k, 75 for 10s32p4k')

    parser.add_argument('--task_base_learning_rate', type=float, default=3.5e-4)
    parser.add_argument('--task_milestones', nargs='+', type=int, default=[25, 35],
                        help='task_milestones for the task learning rate decay')
    parser.add_argument('--task_gamma', type=float, default=0.1,
                        help='task_gamma for the task learning rate decay')

    parser.add_argument('--new_module_learning_rate', type=float, default=3.5e-4)
    parser.add_argument('--new_module_milestones', nargs='+', type=int, default=[75, 85],
                        help='new_milestones for the new module learning rate decay')
    parser.add_argument('--new_module_gamma', type=float, default=0.1,
                        help='new_gamma for the new module learning rate decay')

    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--total_train_epochs', type=int, default=50)
    parser.add_argument('--total_continual_train_epochs', type=int, default=50)

    # resume and save
    parser.add_argument('--auto_resume_training_from_lastest_steps', type=ast.literal_eval, default=True)
    parser.add_argument('--max_save_model_num', type=int, default=1, help='0 for max num is infinit')
    parser.add_argument('--resume_train_dir', type=str, default='',
                        help='directory to resume training. "" stands for output_path')

    # test
    parser.add_argument('--fast_test', type=bool,
                        default=True,
                        help='test during train using Cython')
    parser.add_argument('--test_frequency', type=int,
                        default=25,
                        help='test during train, i <= 0 means do not test during train')
    parser.add_argument('--if_test_forget', type=bool,
                        default=True,
                        help='test during train for forgeting')

    parser.add_argument('--resume_test_model', type=str, default='***',
                        help='only available under test model')
    parser.add_argument('--test_mode', type=str, default='all', help='inter-camera, intra-camera, all')
    parser.add_argument('--test_metric', type=str, default='euclidean', help='cosine, euclidean')

    # visualization configuration
    parser.add_argument('--resume_visualize_model', type=str, default='/path/to/pretrained/model',
                        help='only available under visualize model')
    parser.add_argument('--visualize_dataset', type=str, default='',
                        help='market, duke, only available under visualize model')
    parser.add_argument('--visualize_mode', type=str, default='inter-camera',
                        help='inter-camera, intra-camera, all, only available under visualize model')
    parser.add_argument('--visualize_mode_onlyshow', type=str, default='pos', help='pos, neg, none')
    parser.add_argument('--visualize_output_path', type=str, default='results/visualization/',
                        help='path to save visualization results, only available under visualize model')
    parser.add_argument('--output_featuremaps', type=bool, default=True,
                        help='During training visualize featuremaps')
    parser.add_argument('--output_featuremaps_frequency', type=int, default=10,
                        help='Frequency of visualize featuremaps')
    parser.add_argument('--save_heatmaps', type=bool, default=False,
                        help='During training visualize featuremaps and save')

    # losses configuration
    parser.add_argument('--weight_x', type=float, default=1, help='weight for cross entropy loss')

    # for triplet loss
    parser.add_argument('--weight_t', type=float, default=1, help='weight for triplet loss')
    parser.add_argument('--t_margin', type=float, default=0.3, help='margin for the triplet loss with batch hard')
    parser.add_argument('--t_metric', type=str, default='euclidean', help='euclidean, cosine')
    parser.add_argument('--t_l2', type=bool, default=False, help='if l2 normal for the triplet loss with batch hard')

    # for logit distillation loss
    parser.add_argument('--weight_kd', type=float, default=1, help='weight for cross entropy loss')
    parser.add_argument('--kd_T', type=float, default=2, help='weight for cross entropy loss')

    # for features disstilation
    parser.add_argument('--weight_fkd', type=float, default=0, help='weight for cross entropy loss')
    parser.add_argument('--fkd_l2', type=bool, default=False, help='weight for cross entropy loss')

    # for patch-based losses
    parser.add_argument('--weight_pd', type=float, default=0.1, help='weight for patch distillation loss')
    parser.add_argument('--weight_rd', type=float, default=100, help='weight for relation distillation loss')
    parser.add_argument('--weight_div', type=float, default=0.5, help='weight for patch diversity loss')
    parser.add_argument('--weight_conf', type=float, default=1, help='weight for confidence loss')

    # the number of patches per image
    parser.add_argument('--K', type=int, default=3, help='the number of patches per image')

    # main
    config = parser.parse_args()
    main(config)



