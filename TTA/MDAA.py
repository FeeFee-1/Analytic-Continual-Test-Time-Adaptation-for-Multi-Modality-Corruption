import torch
import torch.nn.parallel
import tqdm
import torch.optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import numpy as np
import models
from torch import nn
from torch.cuda.amp import autocast
import json
class MADA(models.CAVMAEFT):
    def __init__(self,args):
        self.args = args
        super(MADA, self).__init__(label_dim=self.args.n_class, modality_specific_depth=11)
        # 添加新的线性映射层
        if self.args.pretrain_path == 'None':
            print('No pre-trained model!')
        else:
            state_dict = torch.load(self.args.pretrain_path)
            state_dict = self.remove_module_prefix(state_dict)
            miss, unexpected = self.load_state_dict(state_dict, strict=False)
            print(miss, unexpected)

        self.a_analytic_classifier = nn.Sequential(nn.Linear(768, self.args.buffer_size),
                                     nn.ReLU(),
                                     nn.Linear(self.args.buffer_size, self.args.n_class)).cuda()
        
        self.v_analytic_classifier = nn.Sequential(nn.Linear(768, self.args.buffer_size),
                                     nn.ReLU(),
                                     nn.Linear(self.args.buffer_size, self.args.n_class)).cuda()

        self.x_analytic_classifier = nn.Sequential(nn.Linear(768, self.args.buffer_size),
                                     nn.ReLU(),
                                     nn.Linear(self.args.buffer_size, self.args.n_class)).cuda()
        
    def remove_module_prefix(self, state_dict):
        """
        Remove 'module.' prefix from the state dict keys if present.
        """
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_state_dict[key[len('module.'):]] = value
            else:
                new_state_dict[key] = value
        return new_state_dict
    
    def forward(self, a, v, mode, get_fea, eval=False):

        a, v, x = super(MADA, self).forward(a, v, mode, get_fea)

        if eval:
            a1 = self.a_analytic_classifier(a)
            a1 = F.softmax(a1, dim=1)
            v1 = self.v_analytic_classifier(v)
            v1 = F.softmax(v1, dim=1)
            x1 = self.x_analytic_classifier(x)
            x1 = F.softmax(x1, dim=1)
            return a, v, x, a1, v1, x1
        else:
            return a, v, x

def cls_align(train_loader, model,args):
    with open(args.weight_list, 'r') as file:
        weights_list = json.load(file)
    a_new_model = torch.nn.Sequential(model.a_analytic_classifier[:2])
    v_new_model = torch.nn.Sequential(model.v_analytic_classifier[:2])
    x_new_model = torch.nn.Sequential(model.x_analytic_classifier[:2])
    model.eval()
    model.cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    a_auto_cor = torch.zeros(model.a_analytic_classifier[-1].weight.size(1), model.a_analytic_classifier[-1].weight.size(1)).cuda(non_blocking=True)
    a_crs_cor = torch.zeros(model.a_analytic_classifier[-1].weight.size(1), args.n_class).cuda( non_blocking=True)
    v_auto_cor = torch.zeros(model.v_analytic_classifier[-1].weight.size(1), model.v_analytic_classifier[-1].weight.size(1)).cuda(non_blocking=True)
    v_crs_cor = torch.zeros(model.v_analytic_classifier[-1].weight.size(1), args.n_class).cuda( non_blocking=True)
    x_auto_cor = torch.zeros(model.x_analytic_classifier[-1].weight.size(1), model.x_analytic_classifier[-1].weight.size(1)).cuda(non_blocking=True)
    x_crs_cor = torch.zeros(model.x_analytic_classifier[-1].weight.size(1), args.n_class).cuda( non_blocking=True)

    with torch.no_grad():
        for epoch in range(1):
            pbar = tqdm.tqdm(enumerate(train_loader), desc='Re-Alignment Base', total=len(train_loader), unit='batch')
            for i, (a_input, v_input, labels) in pbar:
                indices = torch.argmax(labels, dim=1)
                extracted_weights = [weights_list[i] for i in indices]
                weight = torch.sqrt(torch.Tensor(extracted_weights)).unsqueeze(1).cuda()
                
                a_input = a_input.cuda()
                v_input = v_input.cuda()
                labels = labels.cuda()
                with autocast():
                    a_fea ,v_fea, x_fea = model(a_input, v_input,'multimodal', get_fea=1)
                a_new_activation = a_new_model(a_fea) * weight
                v_new_activation = v_new_model(v_fea) * weight
                x_new_activation = x_new_model(x_fea) * weight

                label_onehot = labels * weight
                a_auto_cor += torch.t(a_new_activation) @ a_new_activation
                v_auto_cor += torch.t(v_new_activation) @ v_new_activation
                x_auto_cor += torch.t(x_new_activation) @ x_new_activation
                a_crs_cor += torch.t(a_new_activation) @ (label_onehot)
                v_crs_cor += torch.t(v_new_activation) @ (label_onehot)
                x_crs_cor += torch.t(x_new_activation) @ (label_onehot)
                
    print('numpy inverse')
    gamma = 10
    a_R = np.mat(a_auto_cor.cpu().numpy() + gamma * np.eye(model.a_analytic_classifier[-1].weight.size(1))).I
    v_R = np.mat(v_auto_cor.cpu().numpy() + gamma * np.eye(model.v_analytic_classifier[-1].weight.size(1))).I
    x_R = np.mat(x_auto_cor.cpu().numpy() + gamma * np.eye(model.x_analytic_classifier[-1].weight.size(1))).I
    a_R = torch.tensor(a_R).float().to(device)
    v_R = torch.tensor(v_R).float().to(device)
    x_R = torch.tensor(x_R).float().to(device)
 
    Delta = a_R @ a_crs_cor
    model.a_analytic_classifier[-1].weight = torch.nn.parameter.Parameter(torch.t(0.9*Delta.float()))
    Delta = v_R @ v_crs_cor
    model.v_analytic_classifier[-1].weight = torch.nn.parameter.Parameter(torch.t(0.9*Delta.float()))
    Delta = x_R @ x_crs_cor
    model.x_analytic_classifier[-1].weight = torch.nn.parameter.Parameter(torch.t(0.9*Delta.float()))
    return a_R, v_R, x_R

import torch
from torch.cuda.amp import autocast

def IL_align(a_input, v_input, model, a_R, v_R, x_R, args):
    a_new_model = torch.nn.Sequential(model.a_analytic_classifier[:2])
    v_new_model = torch.nn.Sequential(model.v_analytic_classifier[:2])
    x_new_model = torch.nn.Sequential(model.x_analytic_classifier[:2])

    model.eval()
    model.cuda()

    a_W = model.a_analytic_classifier[-1].weight.t()
    v_W = model.v_analytic_classifier[-1].weight.t()
    x_W = model.x_analytic_classifier[-1].weight.t()

    a_R = a_R.float()
    v_R = v_R.float()
    x_R = x_R.float()

    with torch.no_grad():
        for _ in range(1):
            a_input = a_input.cuda()
            v_input = v_input.cuda()
            with autocast():
                a_fea, v_fea, x_fea, a_pred, v_pred, x_pred = model(a_input, v_input,'multimodal', get_fea=1, eval=True)

            a_max = torch.max(a_pred, dim=1).values
            v_max = torch.max(v_pred, dim=1).values
            x_max = torch.max(x_pred, dim=1).values

            label_max, indices = torch.max(torch.stack([a_max, v_max, x_max]), dim=0)

            diff_a_max = -a_max + label_max
            diff_v_max = -v_max + label_max
            diff_x_max = -x_max + label_max

            indices_pred = indices.unsqueeze(1).expand_as(a_pred)
            labels = torch.where(indices_pred == 0, a_pred, torch.where(indices_pred == 1, v_pred, x_pred)).cuda()

            labels = SPS(labels, args.alpha)

            mask_a = (diff_a_max > args.theta)
            mask_v = (diff_v_max > args.theta)
            mask_x = (diff_x_max > args.theta)

            label_a = labels[mask_a.flatten()]
            a_fea = a_fea[mask_a.flatten()]

            label_v = labels[mask_v.flatten()]
            v_fea = v_fea[mask_v.flatten()]

            label_x = labels[mask_x.flatten()]
            x_fea = x_fea[mask_x.flatten()]

            if len(label_a)!= 0:
                a_new_activation = a_new_model(a_fea)
                a_R = a_R - a_R @ a_new_activation.t() @ torch.pinverse(torch.eye(label_a.size(0)).cuda(non_blocking=True) +
                                                                        a_new_activation @ a_R @ a_new_activation.t()) @ a_new_activation @ a_R
                a_W = a_W + a_R @ a_new_activation.t() @ (label_a - a_new_activation @ a_W)

            if len(label_v)!= 0:
                v_new_activation = v_new_model(v_fea)
                v_R = v_R - v_R @ v_new_activation.t() @ torch.pinverse(torch.eye(label_v.size(0)).cuda(non_blocking=True) +
                                                                        v_new_activation @ v_R @ v_new_activation.t()) @ v_new_activation @ v_R
                v_W = v_W + v_R @ v_new_activation.t() @ (label_v - v_new_activation @ v_W)

            if len(label_x)!= 0:
                x_new_activation = x_new_model(x_fea)
                x_R = x_R - x_R @ x_new_activation.t() @ torch.pinverse(torch.eye(label_x.size(0)).cuda(non_blocking=True) +
                                                                        x_new_activation @ x_R @ x_new_activation.t()) @ x_new_activation @ x_R
                x_W = x_W + x_R @ x_new_activation.t() @ (label_x - x_new_activation @ x_W)

    model.a_analytic_classifier[-1].weight = torch.nn.parameter.Parameter(torch.t(a_W))
    model.v_analytic_classifier[-1].weight = torch.nn.parameter.Parameter(torch.t(v_W))
    model.x_analytic_classifier[-1].weight = torch.nn.parameter.Parameter(torch.t(x_W))

    return labels, a_R, v_R, x_R


def bias(a_input, v_input, model, a_R, v_R, x_R, args):
    # 特征长度为1024--256--numclass
    model.eval()
    model.cuda()

    a_R = a_R.float()
    v_R = v_R.float()
    x_R = x_R.float()

    with torch.no_grad():
        for _ in range(1):
            a_input = a_input.cuda()
            v_input = v_input.cuda()
            
            a_fea, v_fea, x_fea, a_pred, v_pred, x_pred = model(a_input, v_input,'multimodal', get_fea=1, eval=True)

    return a_pred, v_pred, x_pred

def SPS(labels, i):
    sorted_indices = torch.argsort(labels, dim=1)

    # Initialize the new labels tensor
    new_labels = torch.zeros_like(labels)

    # Define the weights based on the rank
    weights = [round((i - j + 1) / (i * (i + 1) // 2), 1) for j in  range(i)]  # weights for max, second max, etc.

    for j in range(i):
        new_labels[torch.arange(labels.shape[0]), sorted_indices[:, -j - 1]] = weights[j] / sum(weights)

    return new_labels