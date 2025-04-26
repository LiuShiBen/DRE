
import PIL.Image as Image
import time
from .loss import TripletLoss, CrossEntropyLabelSmooth, SoftTripletLoss
import torch
import torch.nn as nn
from .utils.meters import AverageMeter
from .utils.my_tools import *
import numpy as np
from torch.nn import functional as F

class Trainer:
    def __init__(self, args, model, tmodel, optimizer, num_classes,
                 data_loader_train, data_loader_replay, training_phase, add_num=0, replay=False, margin=0.0,
                 ):

        self.model = model
        self.model.cuda()
        self.tmodel = tmodel
        if self.tmodel is not None:
            self.tmodel.cuda()
        self.replay = replay
        self.data_loader_train = data_loader_train
        self.data_loader_replay = data_loader_replay
        self.training_phase = training_phase
        self.add_num = add_num
        self.gamma = 0.5
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_triple = SoftTripletLoss(margin=margin).cuda()
        self.trip_hard = TripletLoss(margin=margin).cuda()
        self.T = 2
        self.consistency = 0.2
        self.consistency_rampup = 100.0
        self.train_iters = len(self.data_loader_train)
        self.device, available_gpus = self._get_available_devices(args.gpu)
        self.model = torch.nn.DataParallel(self.model, device_ids=available_gpus)

        # set optimizer and learning rate
        self.optimizer = optimizer

    @torch.no_grad()
    def update_teachers(self, teacher, keep_rate=0.996):
        # exponential moving average(EMA)
        for ema_param, param in zip(teacher.parameters(), self.model.parameters()):
            ema_param.data = (keep_rate * ema_param.data) + (1 - keep_rate) * param.data

    def predict_with_out_grad(self, imgs, domains, training_phase):
        with torch.no_grad():
            features_old, bn_features_old, cls_out_old = self.tmodel(imgs, domains, training_phase)

        return features_old, bn_features_old, cls_out_old

    def freeze_teachers_parameters(self):
        for p in self.tmodel.parameters():
            p.requires_grad = False

    def get_reliable(self, teacher_predict, student_predict, positive_list, p_name, score_r):
        N = teacher_predict.shape[0]
        score_t = self.iqa_metric(teacher_predict).detach().cpu().numpy()
        score_s = self.iqa_metric(student_predict).detach().cpu().numpy()
        positive_sample = positive_list.clone()
        for idx in range(0, N):
            if score_t[idx] > score_s[idx]:
                if score_t[idx] > score_r[idx]:
                    positive_sample[idx] = teacher_predict[idx]
                    # update the reliable bank
                    temp_c = np.transpose(teacher_predict[idx].detach().cpu().numpy(), (1, 2, 0))
                    temp_c = np.clip(temp_c, 0, 1)
                    arr_c = (temp_c*255).astype(np.uint8)
                    arr_c = Image.fromarray(arr_c)
                    arr_c.save('%s' % p_name[idx])
        del N, score_r, score_s, score_t, teacher_predict, student_predict, positive_list
        return positive_sample

    def train(self, epoch):

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_base = AverageMeter()
        losses_KD = AverageMeter()

        end = time.time()
        self.model.train()
        if self.tmodel is not None:
            self.freeze_teachers_parameters()

        for i in range(len(self.data_loader_train)):
            train_inputs = self.data_loader_train.next()
            data_time.update(time.time() - end)
            imgs, targets, cids, domains = self._parse_data(train_inputs)
            # print("imgs:", imgs.shape, targets.shape)
            targets += self.add_num
            #Current network output
            features, bn_features, cls_out = self.model(imgs, domains, self.training_phase)
            #corss-entroy loss of new samples
            loss_ce = self.CE_loss(cls_out, targets)
            #triplet loss of new samples
            loss_tp = self.Hard_loss(bn_features, targets)
            #orthogonal loss of auxilary embedding representations
            loss_ort = self.Dissimilar(bn_features[1], bn_features[2])
            loss = loss_ce + loss_tp + loss_ort

            #rehearsal
            if self.replay is True:
                imgs_r, fnames_r, pid_r, cid_r, domain_r = next(iter(self.data_loader_replay))
                imgs_r = imgs_r.cuda()
                pid_r = pid_r.cuda()
                # Current network output
                features_r, bn_features_r, cls_out_r = self.model(imgs_r, domain_r, self.training_phase)
                # triplet loss of memory samples
                loss_tr_r = self.Hard_loss(bn_features_r, pid_r)
                loss += loss_tr_r
                #Memory network output
                features_r_old, bn_features_r_old, cls_out_r_old = self.predict_with_out_grad(imgs_r, domain_r, self.training_phase)
                features_old, bn_features_old, cls_out_old = self.predict_with_out_grad(imgs, domains, self.training_phase)

                #consostent and logit-level supervisory loss
                loKD_loss_r = self.loss_kd_old(bn_features_r, bn_features_r_old, cls_out_r, cls_out_r_old)
                loss += loKD_loss_r

                losses_KD.update(loKD_loss_r)
                loss += self.loss_kd_js(cls_out_old, cls_out)

                del cls_out, cls_out_r_old, cls_out_r, bn_features_r, bn_features_r_old, features

            losses_base.update(loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.tmodel is not None:
                with torch.no_grad():
                    self.update_teachers(teacher=self.tmodel)

            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) == self.train_iters or (i + 1) % (self.train_iters // 4) == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Loss_base {:.3f} ({:.3f})\t'
                      'Loss_kd {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, self.train_iters,
                              batch_time.val, batch_time.avg,
                              losses_base.val, losses_base.avg,
                              losses_KD.val, losses_KD.avg))

    def loss_kd_old(self, new_feature, old_feature, new_logit, old_logit):
        new_features = torch.cat([new_feature[0], new_feature[1], new_feature[2]], dim=1)
        new_features = new_features.detach()

        old_features = torch.cat([old_feature[0], old_feature[1], old_feature[2]], dim=1)
        old_features = old_features.detach()

        old_logits = old_logit.detach()
        new_logits = new_logit.detach()

        logsoftmax = nn.LogSoftmax(dim=1).cuda()

        L1 = torch.nn.L1Loss()

        old_simi_matrix = self.cosine_distance(old_features, old_features)
        new_simi_matrix = self.cosine_distance(new_features, new_features)

        simi_loss = L1(old_simi_matrix, new_simi_matrix)
        loss_ke_ce = (- F.softmax(old_logits, dim=1).detach() * logsoftmax(new_logits)).mean(0).sum()

        return loss_ke_ce + simi_loss

    def _get_available_devices(self, n_gpu):
        sys_gpu = torch.cuda.device_count()
        if sys_gpu == 0:
            print('No GPUs detected, using the CPU')
            n_gpu = 0
        elif n_gpu > sys_gpu:
            print(f'Nbr of GPU requested is {n_gpu} but only {sys_gpu} are available')
            n_gpu = sys_gpu
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        available_gpus = list(range(n_gpu))
        return device, available_gpus

    def get_current_consistency_weight(self, epoch):
        return self.consistency * self.sigmoid_rampup(epoch, self.consistency_rampup)

    def sigmoid_rampup(self, current, rampup_length):
        # Exponential rampup
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))
    def _parse_data(self, inputs):
        imgs, _, pids, cids, domains = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets, cids, domains

    def CE_loss(self, s_outputs, targets):
        loss_ce = self.criterion_ce(s_outputs, targets)  #ID loss
        return loss_ce

    def Tri_loss(self, s_features, targets):
        fea_loss = []
        for i in range(len(s_features)):
            loss_tr = self.criterion_triple(s_features[i], s_features[i], targets) #tri loss
            fea_loss.append(loss_tr)
        loss_tr = sum(fea_loss) / len(fea_loss)
        return loss_tr

    def Hard_loss(self, s_features, targets):
        fea_loss = []
        for i in range(0, len(s_features)):
            loss_tr = self.trip_hard(s_features[i], targets)[0]
            fea_loss.append(loss_tr)
        loss_tr = sum(fea_loss) / len(fea_loss)
        return loss_tr

    def cosine_distance(sself, input1, input2):
        """Computes cosine distance.
        Args:
            input1 (torch.Tensor): 2-D feature matrix.
            input2 (torch.Tensor): 2-D feature matrix.
        Returns:
            torch.Tensor: distance matrix.
        """
        input1_normed = F.normalize(input1, p=2, dim=1)
        input2_normed = F.normalize(input2, p=2, dim=1)
        distmat = 1 - torch.mm(input1_normed, input2_normed.t())
        return distmat

    def loss_kd_js(self, old_logit, new_logit):
        old_logits = old_logit.detach()
        new_logits = new_logit
        #print("new_logits:", new_logits.shape, old_logits.shape)
        p_s = F.log_softmax((new_logits + old_logits)/(2*self.T), dim=1)
        p_t = F.softmax(old_logits/self.T, dim=1)
        p_t2 = F.softmax(new_logits/self.T, dim=1)
        loss = 0.5*F.kl_div(p_s, p_t, reduction='batchmean')*(self.T**2) + 0.5*F.kl_div(p_s, p_t2, reduction='batchmean')*(self.T**2)
        return loss

    def Dissimilar(self, feat2, feat3):
        feat23 = torch.cat((feat2.unsqueeze(1), feat3.unsqueeze(1)), 1)
        B, N, C = feat23.shape
        dist_mat = self.cosine_dist(feat23, feat23)
        top_triu = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
        _dist = dist_mat[:, top_triu]
        dist = torch.mean(_dist, dim=(0, 1))
        return dist