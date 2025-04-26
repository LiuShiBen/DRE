import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.autograd import Function
from einops import rearrange, repeat
from pkd.evaluation import accuracy
from pkd.utils import set_bn_to_train, set_bn_to_eval, MultiItemAverageMeter
from pkd.losses import *
from pkd.visualization import featuremaps2heatmaps


def generate_patch_features(x, theta):
    output = (x[:, np.newaxis] * theta[:, :, np.newaxis]).sum(dim=(3, 4))
    return output, rearrange(output, 'n k c -> (n k) c')


def train_p_s_an_epoch(config, base, loader, current_step, old_model, current_epoch=None, output_featuremaps=True):
    base.set_all_model_train()
    meter = MultiItemAverageMeter()
    if old_model is None:
        print('****** training from sketch ******\n')
    else:
        print('****** training with old model ******\n')

    grad_scaler = GradScaler() if config.fp_16 else None
    k = base.model_dict['patchnet'].K
    # we assume 200 iterations as an epoch
    for _ in range(config.steps):
        with autocast(config.fp_16):
            base.set_model_and_optimizer_zero_grad()
            # load a batch data
            mini_batch = loader.continual_train_iter_dict[
                current_step].next_one()
            if mini_batch[0].size(0) != config.p * config.k:
                mini_batch = loader.continual_train_iter_dict[
                    current_step].next_one()
            imgs, global_pids, global_cids, dataset_name, local_pids, image_paths = mini_batch

            if len(mini_batch) > 6:
                assert config.continual_step == 'task'
            imgs, local_pids, global_pids = imgs.to(base.device), local_pids.to(base.device), global_pids.to(base.device)

            loss = 0
            # forward
            if old_model is None:
                features, cls_score, _ = base.model_dict['tasknet'](imgs, current_step)
            else:
                old_current_step = list(range(current_step))
                new_current_step = list(range(current_step + 1))

                features, cls_score_list, feature_maps = base.model_dict['tasknet'](imgs, new_current_step)
                new_logit = torch.cat(cls_score_list, dim=1)
                cls_score = cls_score_list[-1]
                theta = base.model_dict['patchnet'](feature_maps.detach())

                set_bn_to_train(old_model.classifier_dict)
                with torch.no_grad():
                    old_features, old_cls_score_list, old_feature_maps = old_model(imgs, old_current_step, force_output_map=True)
                    old_logit = torch.cat(old_cls_score_list, dim=1)

                old_patch_features_instance, old_patch_features = generate_patch_features(old_feature_maps, theta)
                old_patch_cls_score_list = old_model.classify_latent_codes(old_patch_features, old_current_step)
                old_patch_logit = torch.cat(old_patch_cls_score_list, dim=1)
                set_bn_to_eval(old_model.classifier_dict)

                patch_features_instance, patch_features = generate_patch_features(feature_maps, theta.detach())
                _, patch_features2 = generate_patch_features(feature_maps, theta)
                patch_cls_score_list = base.model_dict['tasknet'].classify_latent_codes_patch(patch_features, new_current_step)
                set_bn_to_eval(base.model_dict['tasknet'].classifier_dict_patch)
                bned_patch_features, _ = base.model_dict['tasknet'].classify_latent_codes_patch(patch_features2, current_step, return_bn=True)
                set_bn_to_train(base.model_dict['tasknet'].classifier_dict_patch)
                patch_logit = torch.cat(patch_cls_score_list, dim=1)

                kd_loss = config.weight_kd * loss_fn_kd(new_logit, old_logit, config.kd_T)

                del old_features, old_feature_maps
                torch.cuda.empty_cache()

                conf_loss = config.weight_conf * loss_fn_kd(old_patch_logit, old_patch_logit, config.kd_T)
                div_loss = config.weight_div * loss_fn_div(bned_patch_features, k=base.model_dict['patchnet'].K)
                pd_loss = config.weight_pd * loss_fn_kd(patch_logit, old_patch_logit.detach(), config.kd_T)
                rd_loss = config.weight_rd * (loss_fn_rd(patch_features_instance, old_patch_features_instance) + loss_fn_rd(patch_features_instance.transpose(0, 1), old_patch_features_instance.transpose(0, 1)))

                meter.update({
                    'Kd_loss': kd_loss.data,
                    'Pd_loss': pd_loss.data,
                    'Rd_loss': rd_loss.data,
                    'Conf_loss': conf_loss.data,
                    'Div_loss': div_loss.data,
                })
                loss += kd_loss + pd_loss + rd_loss + div_loss + conf_loss

            # loss
            ide_loss = config.weight_x * base.ide_criterion(cls_score, local_pids)
            triplet_loss = config.weight_t * base.triplet_criterion(features, features, features, local_pids, local_pids, local_pids)

            loss += ide_loss + triplet_loss
            acc = accuracy(cls_score, local_pids, [1])[0]

            # recored
            meter.update({
                'ide_loss': ide_loss.data,
                'triplet_loss': triplet_loss.data,
                'acc': acc,
            })

        # optimize
        base.optimizer_dict['tasknet'].zero_grad()
        base.optimizer_dict['patchnet'].zero_grad()
        if config.fp_16:  # we use optimier to backward loss
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(base.optimizer_dict['tasknet'])
            grad_scaler.step(base.optimizer_dict['tasknet'])
            # dirty fix
            if old_model is not None:
                grad_scaler.step(base.optimizer_dict['patchnet'])
            grad_scaler.update()
        else:
            loss.backward()
            base.optimizer_dict['tasknet'].step()
            base.optimizer_dict['patchnet'].step()

    if config.re_init_lr_scheduler_per_step:
        _lr_scheduler_step = current_epoch
    else:
        _lr_scheduler_step = current_step * config.total_train_epochs + current_epoch
    base.lr_scheduler_dict['tasknet'].step(_lr_scheduler_step)
    base.lr_scheduler_dict['patchnet'].step(_lr_scheduler_step)

    if output_featuremaps and old_model is not None and current_epoch % config.output_featuremaps_frequency == 0:
        heatmap = featuremaps2heatmaps(base, imgs.detach().cpu().float(), theta.detach().cpu().float(), image_paths,
                                            current_step, current_epoch, if_save=config.save_heatmaps)
        return meter.get_value_dict(), meter.get_str(), heatmap
    else:
        return meter.get_value_dict(), meter.get_str()
