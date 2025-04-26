import torch.nn as nn
import torchvision
import copy
import torch
import numpy as np
from .bnneck import BNClassifier
from torch.autograd import Variable


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class GlobalPoolFlat(nn.Module):
    def __init__(self, pool_mode='avg'):
        super(GlobalPoolFlat, self).__init__()
        if pool_mode == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        x = self.pool(x)
        if len(x.size()) == 4:
            n, c = x.size(0), x.size(1)
        else:
            assert len(x.size()) == 4
        flatted = x.view(n, -1)
        assert flatted.size(1) == c
        return flatted


class LwFNet(nn.Module):
    def __init__(self, class_num_list, pretrained=True):
        super(LwFNet, self).__init__()

        self.class_num_list = class_num_list
        # backbone and optimize its architecture
        resnet = torchvision.models.resnet50(pretrained=pretrained)
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)
        self.backbone = nn.Sequential(
            copy.deepcopy(resnet.conv1),
            copy.deepcopy(resnet.bn1),
            # copy.deepcopy(resnet.relu), # no relu
            copy.deepcopy(resnet.maxpool),
            copy.deepcopy(resnet.layer1),
            copy.deepcopy(resnet.layer2),
            copy.deepcopy(resnet.layer3),
            copy.deepcopy(resnet.layer4)
        )
        # cnn backbone

        self.feature_dim = resnet.fc.in_features
        self.encoder_feature = GlobalPoolFlat(pool_mode='avg')
        del resnet

        # classifier

        self.classifier_dict = nn.ModuleDict()
        for step, num in enumerate(self.class_num_list):
            self.classifier_dict[f'step:{step}'] = BNClassifier(self.feature_dim, num)

        self.classifier_dict_patch = nn.ModuleDict()
        for step, num in enumerate(self.class_num_list):
            self.classifier_dict_patch[f'step:{step}'] = BNClassifier(self.feature_dim, num)

    def forward(self, x, current_step=0, force_output_map=False):
        if isinstance(current_step, list):
            feature_maps = self.backbone(x)
            cls_score_list = []
            features = self.encoder_feature(feature_maps)
            for c_s in current_step:
                bned_features, cls_score = self.classifier_dict[f'step:{c_s}'](features, force_output_score=force_output_map)
                cls_score_list.append(cls_score)
            if self.training or force_output_map:
                return features, cls_score_list, feature_maps
            else:
                return bned_features, feature_maps
        else:
            feature_maps = self.backbone(x)
            features = self.encoder_feature(feature_maps)
            bned_features, cls_score = self.classifier_dict[f'step:{current_step}'](features, force_output_score=force_output_map)
            if self.training or force_output_map:
                return features, cls_score, feature_maps
            else:
                return bned_features, feature_maps

    def classify_latent_codes(self, latent_codes, current_step, return_bn=False):
        if isinstance(current_step, list):
            cls_score_list = []
            for c_s in current_step:
                bned_features, cls_score = self.classifier_dict[f'step:{c_s}'](latent_codes, True)
                cls_score_list.append(cls_score)
            if not return_bn:
                return cls_score_list
            else:
                return bned_features, cls_score_list
        else:
            bned_features, cls_score = self.classifier_dict[f'step:{current_step}'](latent_codes, True)
            if not return_bn:
                return cls_score
            else:
                return bned_features, cls_score

    def classify_latent_codes_patch(self, latent_codes, current_step, return_bn=False):
        if isinstance(current_step, list):
            cls_score_list = []
            for c_s in current_step:
                bned_features, cls_score = self.classifier_dict_patch[f'step:{c_s}'](latent_codes, True)
                cls_score_list.append(cls_score)
            if not return_bn:
                return cls_score_list
            else:
                return bned_features, cls_score_list
        else:
            bned_features, cls_score = self.classifier_dict_patch[f'step:{current_step}'](latent_codes, True)
            if not return_bn:
                return cls_score
            else:
                return bned_features, cls_score
