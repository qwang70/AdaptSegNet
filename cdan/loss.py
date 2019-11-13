import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def CDAN(input_list, ad_net, entropy=None, coeff=None, random_layer=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        # original script
        #op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        #ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
        # TODO: is this correct?
        # (2, 19, 720, 1280)
        op_out = torch.mul(softmax_output, feature)
        # TODO: Current output ad_out is [2, 1, 180, 320]
        # Qiwen: I want the output shape to be (2, 1), so I added 2 linear layer 
        # in FCDiscriminatorTest
        # Because the discriminator only discriminate on the overall image,
        # not on the pixel level.
        # (2, 1, 180, 320)
        # ad_net here is the FCDiscriminatorTest
        ad_out = ad_net(op_out)
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))       
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float()

    # Half size of ad_out is the batch size of the synthesis sample/real sample.
    # Half of the batch size of ad_out is fill with label 1, and the other half
    # is filled with label 0
    half_size = ad_out.data[:batch_size].size()
    dc_target = torch.cat((torch.FloatTensor(half_size).fill_(0),torch.FloatTensor(half_size).fill_(1)), dim = 0)
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0)//2:] = 0
        source_weight = entropy*source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0)//2] = 0
        target_weight = entropy*target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
        return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCEWithLogitsLoss()(ad_out, dc_target) 

def DANN(features, ad_net):
    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    return nn.BCELoss()(ad_out, dc_target)
