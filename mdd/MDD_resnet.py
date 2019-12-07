import torch.nn as nn
import model.backbone as backbone
import torch.nn.functional as F
import torch
import numpy as np

class GradientReverseLayer(torch.autograd.Function):
    iter_num = 0
    alpha = 1.0
    low_value = 0.0
    high_value = 0.1
    max_iter = 1000.0

    def __init__(self, iter_num=0, alpha=1.0, low_value=0.0, high_value=0.1, max_iter=1000.0):
        GradientReverseLayer.iter_num = iter_num
        GradientReverseLayer.alpha = alpha
        GradientReverseLayer.low_value = low_value
        GradientReverseLayer.high_value = high_value
        GradientReverseLayer.max_iter = max_iter

    @staticmethod
    def forward(ctx, input):
        try:
            ctx.iter_num += 1
        except:
            ctx.iter_num = 1
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        coeff = np.float(
            2.0 * (GradientReverseLayer.high_value - GradientReverseLayer.low_value) / (1.0 + np.exp(-GradientReverseLayer.alpha * ctx.iter_num / GradientReverseLayer.max_iter)) - (
                        GradientReverseLayer.high_value - GradientReverseLayer.low_value) + GradientReverseLayer.low_value)
        return -coeff * grad_output

class Classifier_Module(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
            return out



def trans(x):
    ## x = (B, F, H, W)
    ## output = (B*H*W, F)
    
    B, F, H, W = x.size()
    output = x.permute(0,2,3,1)
    output = output.view(B*H*W, F)

    return output

def detrans(x, B, H, W):
    # x = (B*H*W, K)
    ## output = (B, K, H, W)
    
    output = x.view(B, H, W, -1)
    output = output.permute(0,3,1,2)

    return output

class MDDNet(nn.Module):
    def __init__(self, base_net='ResNet101', use_bottleneck=True, bottleneck_dim=1024, width=1024, class_num=31):
        super(MDDNet, self).__init__()
        ## set base network
        self.base_network = backbone.network_dict[base_net]()
        self.use_bottleneck = use_bottleneck
        self.grl_layer = GradientReverseLayer()
        self.bottleneck_layer_list = [nn.Linear(self.base_network.output_num(), bottleneck_dim), nn.BatchNorm1d(bottleneck_dim), nn.ReLU(), nn.Dropout(0.5)]
        self.bottleneck_layer = nn.Sequential(*self.bottleneck_layer_list)
        self.classifier_layer_list = [nn.Linear(bottleneck_dim, width), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Linear(width, class_num)]
        self.classifier_layer = nn.Sequential(*self.classifier_layer_list)
        self.classifier_layer_2_list = [nn.Linear(bottleneck_dim, width), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Linear(width, class_num)]
        self.classifier_layer_2 = nn.Sequential(*self.classifier_layer_2_list)
        self.softmax = nn.Softmax(dim=1)

        self.classifier_layer_1_conv2d = self._make_pred_layer(Classifier_Module, bottleneck_dim, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.classifier_layer_2_conv2d = self._make_pred_layer(Classifier_Module, bottleneck_dim, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)

        ## initialization
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

    def forward(self, inputs):
        features = self.base_network(inputs)
        B, _, H, W = features.size()
        features = trans(features)
        if self.use_bottleneck:
            features = self.bottleneck_layer(features)
        features = detrans(features, B, H, W)
        features_adv = self.grl_layer(features)
        # features_adv = trans(features_adv)
        outputs_adv = self.classifier_layer_2_conv2d(features_adv)
        
        # features = trans(features)
        outputs = self.classifier_layer_1_conv2d(features)
        softmax_outputs = self.softmax(outputs)

        return features, outputs, softmax_outputs, outputs_adv

    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)

class MDD(object):
    def __init__(self, base_net='ResNet101', width=1024, class_num=31, use_bottleneck=True, use_gpu=True, srcweight=3):
        self.c_net = MDDNet(base_net, use_bottleneck, width, width, class_num)

        self.use_gpu = use_gpu
        self.is_train = False
        self.iter_num = 0
        self.class_num = class_num
        if self.use_gpu:
            self.c_net = self.c_net.cuda()
        self.srcweight = srcweight

    def get_loss(self, inputs, labels_source):
        class_criterion = nn.CrossEntropyLoss()

        _, outputs, _, outputs_adv = self.c_net(inputs)

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
