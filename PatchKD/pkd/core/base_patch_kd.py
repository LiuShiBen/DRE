import sys
sys.path.append('..')
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import os
import os.path as osp
import numpy as np
from collections import OrderedDict
from pkd.models import LwFNet, PatchNet
from pkd.losses import CrossEntropyLabelSmooth, TripletLoss
from pkd.utils import os_walk, make_dirs
from .lr_schedulers import WarmupMultiStepLR, torch16_MultiStepLR


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
GRID_SPACING = 10


class BasePatchKD(object):
    '''
    a base module includes model, optimizer, loss, and save/resume operations.
    '''

    def __init__(self, config, loader):

        self.config = config
        self.loader = loader
        # Model Config
        self.mode = config.mode
        self.pid_num = config.pid_num
        self.t_margin = config.t_margin
        # Logger Configuration
        self.max_save_model_num = config.max_save_model_num
        self.output_path = config.output_path
        self.output_dirs_dict = {'logs': os.path.join(self.output_path, 'logs/'),
                                 'models': os.path.join(self.output_path, 'models/'),
                                 'images': os.path.join(self.output_path, 'images/'),
                                 'features': os.path.join(self.output_path, 'features/')}

        # make directions
        for current_dir in self.output_dirs_dict.values():
            make_dirs(current_dir)
        # Train Configuration
        # resume_train_dir
        self.resume_train_dir = config.resume_train_dir

        # init
        self._init_device()
        self._init_model()
        self._init_criterion()
        self._init_optimizer()

    def _init_device(self):
        self.device = torch.device('cuda')


    def _init_model(self):
        pretrained = False if self.mode != 'train' else True

        self.model_dict = nn.ModuleDict()

        num_class_list = self.loader.continual_num_pid_per_step

        self.model_dict['tasknet'] = LwFNet(class_num_list=num_class_list, pretrained=pretrained)
        self.model_dict['patchnet'] = PatchNet(K=self.config.K)

        self.model_dict.to(self.device)
        for name, module in self.model_dict.items():
            module = module.to(self.device)

    def _init_criterion(self):
        self.ide_criterion = CrossEntropyLabelSmooth(self.pid_num)
        self.triplet_criterion = TripletLoss(self.t_margin, self.config.t_metric, self.config.t_l2)
        self.reconstruction_criterion = nn.L1Loss()

    def _init_optimizer(self):
        self.lr_scheduler_dict = {}
        self.optimizer_dict = {}
        for name, module in self.model_dict.items():
            if 'task' in name:
                self.optimizer_dict[name] = optim.Adam(module.parameters(), lr=self.config.task_base_learning_rate,
                                                       weight_decay=self.config.weight_decay)
                if self.config.warmup_lr:
                    self.lr_scheduler_dict[name] = WarmupMultiStepLR(self.optimizer_dict[name],
                                                                     self.config.task_milestones,
                                                                     gamma=self.config.task_gamma,
                                                                     warmup_factor=0.01,
                                                                     warmup_iters=10)

                else:
                    self.lr_scheduler_dict[name] = torch16_MultiStepLR(self.optimizer_dict[name],
                                                                       self.config.task_milestones,
                                                                       gamma=self.config.task_gamma)

            else:
                self.optimizer_dict[name] = optim.Adam(module.parameters(), lr=self.config.new_module_learning_rate,
                                                       weight_decay=self.config.weight_decay)
                if self.config.warmup_lr:
                    self.lr_scheduler_dict[name] = WarmupMultiStepLR(self.optimizer_dict[name],
                                                                     self.config.new_module_milestones,
                                                                     gamma=self.config.new_module_gamma,
                                                                     warmup_factor=0.01,
                                                                     warmup_iters=10)
                else:
                    self.lr_scheduler_dict[name] = torch16_MultiStepLR(self.optimizer_dict[name],
                                                                       self.config.new_module_milestones,
                                                                       gamma=self.config.new_module_gamma)

    def save_model(self, save_step, save_epoch):
        '''save model as save_epoch'''
        # save model
        models_steps_path = os.path.join(self.output_dirs_dict['models'], str(save_step))
        if not osp.exists(models_steps_path):
            make_dirs(models_steps_path)
        for module_name, module in self.model_dict.items():
            torch.save(module.state_dict(),
                       os.path.join(models_steps_path, f'model_{module_name}_{save_epoch}.pkl'))
        for optimizer_name, optimizer in self.optimizer_dict.items():
            torch.save(optimizer.state_dict(),
                       os.path.join(models_steps_path, f'optimizer_{optimizer_name}_{save_epoch}.pkl'))


        # if saved model is more than max num, delete the model with smallest epoch
        if self.max_save_model_num > 0:
            root, _, files = os_walk(models_steps_path)

            # get indexes of saved models
            indexes = []
            for file in files:
                indexes.append(int(file.replace('.pkl', '').split('_')[-1]))

            # remove the bad-case and get available indexes
            model_num = len(self.model_dict)
            optimizer_num = len(self.optimizer_dict)
            available_indexes = copy.deepcopy(indexes)
            for element in indexes:
                if indexes.count(element) < model_num + optimizer_num:
                    available_indexes.remove(element)

            available_indexes = sorted(list(set(available_indexes)), reverse=True)
            unavailable_indexes = list(set(indexes).difference(set(available_indexes)))

            # delete all unavailable models
            for unavailable_index in unavailable_indexes:
                try:
                    # os.system('find . -name "{}*_{}.pth" | xargs rm  -rf'.format(self.config.save_models_path, unavailable_index))
                    for module_name in self.model_dict.keys():
                        os.remove(os.path.join(root, f'model_{module_name}_{unavailable_index}.pkl'))
                    for optimizer_name in self.optimizer_dict.keys():
                        os.remove(os.path.join(root, f'optimizer_{optimizer_name}_{unavailable_index}.pkl'))
                except:
                    pass

            # delete extra models
            if len(available_indexes) >= self.max_save_model_num:
                for extra_available_index in available_indexes[self.max_save_model_num:]:
                    # os.system('find . -name "{}*_{}.pth" | xargs rm  -rf'.format(self.config.save_models_path, extra_available_index))
                    for mudule_name, mudule in self.model_dict.items():
                        os.remove(os.path.join(root, f'model_{mudule_name}_{extra_available_index}.pkl'))
                    for optimizer_name, optimizer in self.optimizer_dict.items():
                        os.remove(os.path.join(root, f'optimizer_{optimizer_name}_{extra_available_index}.pkl'))

    def resume_last_model(self):
        '''resume model from the last one in path self.output_path'''
        # find all files in format of *.pkl

        if self.resume_train_dir == '':
            root, dir, files = os_walk(self.output_dirs_dict['models'])
        else:
            root, dir, files = os_walk(os.path.join(self.resume_train_dir, 'models'))
        if len(dir) > 0:
            resume_step = max(dir)
        else:
            return 0, 0
        _, _, files = os_walk(os.path.join(root, resume_step))
        for file in files:
            if '.pkl' not in file:
                files.remove(file)
        # find the last one
        if len(files) > 0:
            # get indexes of saved models
            indexes = []
            for file in files:
                indexes.append(int(file.replace('.pkl', '').split('_')[-1]))
            indexes = sorted(list(set(indexes)), reverse=False)
            # resume model from the latest model
            self.resume_model(resume_step, indexes[-1])
            #
            start_train_epoch = indexes[-1]
            start_train_step = resume_step
            return int(start_train_step), start_train_epoch
        else:
            return 0, 0

    def resume_model(self, resume_step, resume_epoch):
        '''resume model from resume_epoch'''
        for module_name, module in self.model_dict.items():
            if self.resume_train_dir == '':
                model_path = os.path.join(self.output_dirs_dict['models'], resume_step, f'model_{module_name}_{resume_epoch}.pkl')
            else:
                model_path = os.path.join(self.resume_train_dir, 'models', resume_step, f'model_{module_name}_{resume_epoch}.pkl')
            try:
                module.load_state_dict(torch.load(model_path), strict=False)
            except:
                print(('fail resume model from {}'.format(model_path)))
                pass
            else:
                print(('successfully resume model from {}'.format(model_path)))

        for optimizer_name, optimizer in self.optimizer_dict.items():
            if self.resume_train_dir == '':
                model_path = os.path.join(self.output_dirs_dict['models'], resume_step, f'optimizer_{optimizer_name}_{resume_epoch}.pkl')
            else:
                model_path = os.path.join(self.resume_train_dir, 'models', resume_step, f'optimizer_{optimizer_name}_{resume_epoch}.pkl')
            try:
                optimizer.load_state_dict(torch.load(model_path))
            except:
                print(('fail resume optimizer from {}'.format(model_path)))
                pass
            else:
                print(('successfully resume optimizer from {}'.format(model_path)))

    def resume_from_model(self, models_dir):
        '''resume from model. model_path shoule be like /path/to/model.pkl'''
        # self.model.load_state_dict(torch.load(model_path), strict=False)
        # print(('successfully resume model from {}'.format(model_path)))
        '''resume model from resume_epoch'''
        for module_name, module in self.model_dict.items():
            model_path = os.path.join(models_dir, f'model_{module_name}_50.pkl')
            state_dict = torch.load(model_path)
            model_dict = module.state_dict()
            new_state_dict = OrderedDict()
            matched_layers, discarded_layers = [], []
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    k = k[7:]  # discard module.
                if k in model_dict and model_dict[k].size() == v.size():
                    new_state_dict[k] = v
                    matched_layers.append(k)
                else:
                    discarded_layers.append(k)
            model_dict.update(new_state_dict)
            module.load_state_dict(model_dict)
            if len(discarded_layers) > 0:
                print('discarded layers: {}'.format(discarded_layers))

            print(('successfully resume model from {}'.format(model_path)))

    # set model as train mode
    def set_all_model_train(self):
        for name, module in self.model_dict.items():
            module = module.train()
            module.training = True

    # set model as eval mode
    def set_all_model_eval(self):
        for name, module in self.model_dict.items():
            module = module.eval()
            module.training = False

    def set_specific_models_train(self, models_list):
        copy_list = copy.deepcopy(list(self.model_dict.keys()))
        print(f'****** open following modules for training! ******')
        for specific_name in models_list:
            if specific_name in copy_list:
                self.model_dict[specific_name] = self.model_dict[specific_name].train()
                self.model_dict[specific_name].training = True
                copy_list.remove(specific_name)
                print(f'open < {specific_name} > modules !')
        print(f'**************************************************\n')
        print(f'****** close the other modules for training! ******')
        for non_specific_name in copy_list:
            self.model_dict[non_specific_name] = self.model_dict[non_specific_name].eval()
            self.model_dict[non_specific_name].training = False
            print(f'close < {non_specific_name} > modules !')
        print(f'**************************************************\n')

    def close_all_layers(self, model):
        r"""Opens all layers in model for training.

        Examples::
            >>> from torchreid.utils import open_all_layers
            >>> open_all_layers(model)
        """
        model.train()
        for p in model.parameters():
            p.requires_grad = False

    def open_specified_layers(self, model, open_layers):
        r"""Opens specified layers in model for training while keeping
        other layers frozen.

        Args:
            model (nn.Module): neural net model.
            open_layers (str or list): layers open for training.

        Examples::
            >>> from torchreid.utils import open_specified_layers
            >>> # Only model.classifier will be updated.
            >>> open_layers = 'classifier'
            >>> open_specified_layers(model, open_layers)
            >>> # Only model.fc and model.classifier will be updated.
            >>> open_layers = ['fc', 'classifier']
            >>> open_specified_layers(model, open_layers)
        """
        if isinstance(model, nn.DataParallel):
            model = model.module

        if isinstance(open_layers, str):
            open_layers = [open_layers]

        for layer in open_layers:
            assert hasattr(
                model, layer
            ), '"{}" is not an attribute of the model, please provide the correct name'.format(
                layer
            )

        for name, module in model.named_children():
            if name in open_layers:
                module.train()
                for p in module.parameters():
                    p.requires_grad = True
            else:
                module.eval()
                for p in module.parameters():
                    p.requires_grad = False

    # set model as eval mode
    def close_specific_layers(self, model_name, layers_list):
        if isinstance(self.model_dict[model_name], nn.DataParallel):
            model = self.model_dict[model_name].module
        else:
            model = self.model_dict[model_name]
        if isinstance(layers_list, str):
            layers_list = [layers_list]

        for layer in layers_list:
            assert hasattr(
                model, layer
            ), '"{}" is not an attribute of the model, please provide the correct name'.format(
                layer
            )
        for name, module in model.named_children():
            if name in layers_list:
                module.eval()
                for p in module.parameters():
                    p.requires_grad = False
                print(f'****** close {name} layers and set it as eval mode! ******')

    def set_specific_models_eval(self, models_list):
        copy_list = copy.deepcopy(list(self.model_dict.keys()))
        print(f'****** close following modules for testing! ******')
        for specific_name in models_list:
            if specific_name in copy_list:
                self.model_dict[specific_name] = self.model_dict[specific_name].eval()
                self.model_dict[specific_name].training = False
                copy_list.remove(specific_name)
                print(f'close < {specific_name} > modules !')
        print(f'**************************************************\n')

        print(f'****** open the other modules for testing! ******')
        for non_specific_name in copy_list:
            self.model_dict[non_specific_name] = self.model_dict[non_specific_name].train()
            self.model_dict[non_specific_name].training = True
            print(f'close < {specific_name} > modules !')
        print(f'**************************************************\n')

    def set_bn_to_eval(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm2d') != -1:
            m.eval()

    def set_bn_to_train(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm2d') != -1:
            m.train()

    def set_model_and_optimizer_zero_grad(self, mode=['model', 'optimizer']):
        if 'model' in mode:
            for name, module in self.model_dict.items():
                module.zero_grad()
        if 'optimizer' in mode:
            for name, optimizer in self.optimizer_dict.items():
                optimizer.zero_grad()

    def make_onehot(self, label):
        onehot_vec = torch.zeros(label.size()[0], self.config.class_num)
        for i in range(label.size()[0]):
            onehot_vec[i, label[i]] = 1
        return onehot_vec

    def get_current_learning_rate(self):
        str_output = 'current learning rate: '
        dict_output = {}
        for name, optim in self.optimizer_dict.items():
            str_output += f" <{name}> = <{optim.param_groups[0]['lr']}>; "
            dict_output[name] = optim.param_groups[0]['lr']

        return str_output + f'\n', dict_output

    def copy_model(self, model_name='tasknet'):
        old_model = copy.deepcopy(self.model_dict[model_name])
        old_model = old_model.to(self.device)
        return old_model.train().requires_grad_(True)

    def copy_model_and_frozen(self, model_name='tasknet'):
        old_model = copy.deepcopy(self.model_dict[model_name])
        old_model = old_model.to(self.device)
        return old_model.eval().requires_grad_(False)
