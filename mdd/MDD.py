import torch.nn as nn
import mdd.backbone as backbone
import model.deeplab_multi as deeplab_multi
import torch.nn.functional as F
import torch
import numpy as np

from torch.utils import data, model_zoo
import pdb

RESTART = False
RESTART_FROM = './snapshots/baseline_single_50_seg0.1_adv10.0002_adv20.001_bs1_11-10-8-20/'
RESTART_ITER = 2

class GradientReverseLayer(torch.autograd.Function):
    def __init__(self, iter_num=0, alpha=1.0, low_value=0.0, high_value=0.1, max_iter=1000.0):
        self.iter_num = iter_num
        self.alpha = alpha
        self.low_value = low_value
        self.high_value = high_value
        self.max_iter = max_iter

    def forward(self, input):
        self.iter_num += 1
        output = input * 1.0
        return output

    def backward(self, grad_output):
        self.coeff = np.float(
            2.0 * (self.high_value - self.low_value) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter)) - (
                        self.high_value - self.low_value) + self.low_value)
        return -self.coeff * grad_output


class MDDNet(nn.Module):
    def __init__(self, base_net='ResNet18', use_bottleneck=True, bottleneck_dim=1024, width=512, class_num=31, args=None):
        super(MDDNet, self).__init__()
        # TODO: check the effect of bottleneck dim and the width
        if not use_bottleneck:
            bottleneck_dim = class_num
        ## set base network
        #self.base_network = backbone.network_dict[base_net]()
        self.base_network = deeplab_multi.DeeplabMulti(num_classes=class_num)
        self.use_bottleneck = use_bottleneck
        self.grl_layer = GradientReverseLayer()
        self.bottleneck_layer_list = [nn.Linear(
            #self.base_network.output_num(), 
            class_num,
            bottleneck_dim), nn.BatchNorm2d(bottleneck_dim), nn.ReLU(), nn.Dropout(0.5)]
        self.bottleneck_layer = nn.Sequential(*self.bottleneck_layer_list)
        self.classifier_layer_list = [nn.Linear(bottleneck_dim, width), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Linear(width, class_num)]
        self.classifier_layer = nn.Sequential(*self.classifier_layer_list)
        self.classifier_layer_2_list = [nn.Linear(bottleneck_dim, width), nn.ReLU(), nn.Dropout(0.5),

                                        nn.Linear(width, class_num)]
        self.classifier_layer_2 = nn.Sequential(*self.classifier_layer_2_list)
        self.softmax = nn.Softmax(dim=1)

        ## initialization
        self.init_deeplab_multi(args)
        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)
        for dep in range(2):
            self.classifier_layer_2[dep * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer_2[dep * 3].bias.data.fill_(0.0)
            self.classifier_layer[dep * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer[dep * 3].bias.data.fill_(0.0)


        ## collect parameters
        self.parameter_list = [{"params":self.base_network.parameters(), "lr":0.1},
                            {"params":self.bottleneck_layer.parameters(), "lr":1},
                        {"params":self.classifier_layer.parameters(), "lr":1},
                               {"params":self.classifier_layer_2.parameters(), "lr":1}]


    def init_deeplab_multi(self, args):
        #### restore model_D and model
        if RESTART:
            # model parameters
            restart_from_model = RESTART_FROM + 'GTA5_{}.pth'.format(RESTART_ITER)
            saved_state_dict = torch.load(restart_from_model)
            self.base_network.load_state_dict(saved_state_dict)

        else:
            # model parameters
            if args.restore_from[:4] == 'http' :
                saved_state_dict = model_zoo.load_url(args.restore_from)
            else:
                saved_state_dict = torch.load(args.restore_from)

            new_params = self.base_network.state_dict().copy()
            for i in saved_state_dict:
                # Scale.layer5.conv2d_list.3.weight
                i_parts = i.split('.')
                # print i_parts
                if not args.num_classes == 19 or not i_parts[1] == 'layer5':
                    new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
                    # print i_parts
            self.base_network.load_state_dict(new_params)
        self.base_network.train()

    def forward(self, inputs):
        _, features = self.base_network(inputs) # (b, 3, w_small, h_small)
        features = features.permute(0,2,3,1) # (b, w, h, 19)
        if self.use_bottleneck:
            features = self.bottleneck_layer(features)
        features_adv = self.grl_layer(features)
        outputs_adv = self.classifier_layer_2(features_adv)
        
        outputs = self.classifier_layer(features)
        softmax_outputs = self.softmax(outputs)

        # outputs, outputs_adv: [2, 65, 129, 19]
        # permute back
        outputs = outputs.permute(0,3,1,2)
        outputs_adv = outputs_adv.permute(0,3,1,2)
        # outputs, outputs_adv: [2, 19, 65, 129]
        return features, outputs, softmax_outputs, outputs_adv

class MDD(object):
    def __init__(self, base_net='ResNet18', width=1024, class_num=19, use_bottleneck=True, use_gpu=True, srcweight=3, args=None):
        self.c_net = MDDNet(base_net, use_bottleneck, width, width, class_num, args)

        self.use_gpu = use_gpu
        self.is_train = False
        self.iter_num = 0
        self.class_num = class_num
        if self.use_gpu:
            self.c_net = self.c_net.cuda()
        self.srcweight = srcweight

        #w, h = map(int, args.input_size.split(','))
        input_size = (1024, 512)
        self.interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)

    def get_loss(self, inputs, labels_source):
        class_criterion = nn.CrossEntropyLoss(ignore_index=255)

        _, outputs, _, outputs_adv = self.c_net(inputs)
        outputs = self.interp(outputs)
        outputs_adv = self.interp(outputs_adv)

        #pdb.set_trace()
        classifier_loss = class_criterion(outputs.narrow(0, 0, labels_source.size(0)), labels_source)

        target_adv = outputs.max(1)[1]
        target_adv_src = target_adv.narrow(0, 0, labels_source.size(0))
        target_adv_tgt = target_adv.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0))

        classifier_loss_adv_src = class_criterion(outputs_adv.narrow(0, 0, labels_source.size(0)), target_adv_src)

        logloss_tgt = torch.log(1 - F.softmax(outputs_adv.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0)), dim = 1))
        classifier_loss_adv_tgt = F.nll_loss(logloss_tgt, target_adv_tgt)

        transfer_loss = self.srcweight * classifier_loss_adv_src + classifier_loss_adv_tgt

        self.iter_num += 1

        total_loss = classifier_loss + transfer_loss

        return total_loss

    def predict(self, inputs):
        _, _, softmax_outputs,_= self.c_net(inputs)
        return softmax_outputs

    def get_parameter_list(self):
        return self.c_net.parameter_list

    def set_train(self, mode):
        self.c_net.train(mode)
        self.is_train = mode
