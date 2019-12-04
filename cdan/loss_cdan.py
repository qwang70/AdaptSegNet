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

def transfer_loss(ad_out, dc_target, entropy=None, coeff=None,):
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
        # pdb.set_trace()
        return nn.BCEWithLogitsLoss()(ad_out, dc_target) 


def CDAN(input_list, ad_net, cdan_implement = 'concat', K = 50):

    output = input_list[1]
    feature = input_list[0]
    if cdan_implement == 'concat':
        # original script
        #op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        #ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
        # TODO: is this correct?
        # (2, 19, 720, 1280)
        op_out = torch.cat((output, feature), dim = 1)
        # pdb.set_trace()
        # TODO: Current output ad_out is [2, 1, 180, 320]
        # Qiwen: I want the output shape to be (2, 1), so I added 2 linear layer 
        # in FCDiscriminatorTest
        # Because the discriminator only discriminate on the overall image,
        # not on the pixel level.
        # (2, 1, 180, 320)
        # ad_net here is the FCDiscriminatorTest
        ad_out = ad_net(op_out)
    
    elif cdan_implement == 'elemul':
        op_out = output * feature
        # pdb.set_trace()
        ad_out = ad_net(op_out)
    
    elif cdan_implement == 'random':
        ### random algorithm1 ####
        # g: (B, C, H, W), Rg:(C, K, H, W)
        # f: (B, F, H, W), Rf:(F, K, H, W)
        # random_out = (Rg@g)*(Rf@f)
        B, C, H, W = output.size()
        _, F, _, _ = feature.size()
        
        output = torch.transpose(torch.transpose(output.view(B, C, H*W), 1, 2), 0, 1)
        feature = torch.transpose(torch.transpose(feature.view(B, F, H*W), 1, 2), 0, 1)

        random_matrix_f = torch.randn(H*W, F, K).cuda()
        random_matrix_g = torch.randn(H*W, C, K).cuda()
        random_out_f = torch.transpose(torch.transpose(torch.bmm(feature, random_matrix_f), 0,1), 1,2)
        random_out_g = torch.transpose(torch.transpose(torch.bmm(output, random_matrix_g), 0,1), 1,2)

        random_out = torch.mul(random_out_f, random_out_g).view(B, K, H, W) 
        ad_out = ad_net(random_out)
    else:
        raise Exception("cdan implementation type wrong!")

    return ad_out
    


def DANN(features, ad_net):
    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    return ad_out
    # return nn.BCELoss()(ad_out, dc_target)
