import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import random
from tensorboardX import SummaryWriter

import pdb

from model.deeplab_multi import DeeplabMulti
from model.discriminator import FCDiscriminator, FCDiscriminatorTest
from utils.loss import CrossEntropy2d
from dataset.gta5_dataset import GTA5DataSet
from dataset.cityscapes_dataset import cityscapesDataSet

from cdan.loss import CDAN

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'DeepLab'
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 2
DATA_DIRECTORY = './data/GTA5'
DATA_LIST_PATH = './dataset/gta5_list/train.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '1280,720'
DATA_DIRECTORY_TARGET = './data/Cityscapes/data'
DATA_LIST_PATH_TARGET = './dataset/cityscapes_list/train.txt'
INPUT_SIZE_TARGET = '1024,512'
INPUT_SIZE_TARGET = '1280,720'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 250000
NUM_STEPS_STOP = 150000  # early stopping
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 2
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005
LOG_DIR = './log'

LEARNING_RATE_D = 1e-4
LAMBDA_SEG = 0.1
LAMBDA_ADV = 0.001
# LAMBDA_ADV_TARGET1 = 0.0002
# LAMBDA_ADV_TARGET2 = 0.001
GAN = 'Vanilla'

TARGET = 'cityscapes'
SET = 'train'

RESTART = False
RESTART_FROM = './snapshots/baseline_single_50_seg0.1_adv10.0002_adv20.001_bs1_11-10-8-20/'
RESTART_ITER = 2

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                        help="lambda_seg.")
    """
    parser.add_argument("--lambda-adv-target1", type=float, default=LAMBDA_ADV_TARGET1,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-adv-target2", type=float, default=LAMBDA_ADV_TARGET2,
                        help="lambda_adv for adversarial training.")"""
    parser.add_argument("--lambda-adv", type=float, default=LAMBDA_ADV,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")
    parser.add_argument("--tensorboard", action='store_true', help="choose whether to use tensorboard.")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                        help="Path to the directory of log.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    parser.add_argument("--gan", type=str, default=GAN,
                        help="choose the GAN objective.")
    return parser.parse_args()


args = get_arguments()


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def generate_snapshot_name(args):

    import datetime
    now_time = datetime.datetime.now()
    snapshot_dir = SNAPSHOT_DIR + 'baseline_cdan_norandom' + 'multi_' \
        + str(NUM_STEPS) + '_seg{}'.format(args.lambda_seg) + '_adv{}'.format(args.lambda_adv)\
        + '_bs{}'.format(BATCH_SIZE) + \
        '_{}-{}-{}-{}'.format(now_time.month, now_time.day, now_time.hour, now_time.minute)

    return snapshot_dir


def main():

    """Create the model and start the training."""
    if RESTART:
        args.snapshot_dir = RESTART_FROM
    else:
        args.snapshot_dir = generate_snapshot_name(args)

    args_dict = vars(args)
    import json

    ###### load args for restart ######
    if RESTART:
        # pdb.set_trace()
        args_dict_file = args.snapshot_dir + 'args_dict_{}.json'.format(RESTART_ITER)
        with open(args_dict_file) as f:
            args_dict_last = json.load(f)
        for arg in args_dict:
            args_dict[arg] = args_dict_last[arg]

    ###### load args for restart ######

    device = torch.device("cuda" if not args.cpu else "cpu")

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)

    cudnn.enabled = True
    cudnn.benchmark = True

    if args.model == 'DeepLab':
        model = DeeplabMulti(num_classes=args.num_classes)
    
    model_D = FCDiscriminatorTest(num_classes=args.num_classes).to(device)

    #### restore model_D and model
    if RESTART:
        # pdb.set_trace()
        # model parameters
        restart_from_model = args.restart_from + 'GTA5_{}.pth'.format(RESTART_ITER)
        saved_state_dict = torch.load(restart_from_model)
        model.load_state_dict(saved_state_dict)

        # model_D parameters
        restart_from_D = args.restart_from + 'GTA5_{}_D.pth'.format(RESTART_ITER)
        saved_state_dict = torch.load(restart_from_D)
        model_D.load_state_dict(saved_state_dict)

    #### model_D1, D2 are randomly initialized, model is pre-trained ResNet on ImageNet
    else:
        # model parameters
        if args.restore_from[:4] == 'http' :
            saved_state_dict = model_zoo.load_url(args.restore_from)
        else:
            saved_state_dict = torch.load(args.restore_from)

        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            # Scale.layer5.conv2d_list.3.weight
            i_parts = i.split('.')
            # print i_parts
            if not args.num_classes == 19 or not i_parts[1] == 'layer5':
                new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
                # print i_parts
        model.load_state_dict(new_params)

    model.train()
    model.to(device)

    model_D.train()
    model_D.to(device)


    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    trainloader = data.DataLoader(
        GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                    crop_size=input_size,
                    scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    trainloader_iter = enumerate(trainloader)

    targetloader = data.DataLoader(cityscapesDataSet(args.data_dir_target, args.data_list_target,
                                                     max_iters=args.num_steps * args.iter_size * args.batch_size,
                                                     crop_size=input_size_target,
                                                     scale=False, mirror=args.random_mirror, mean=IMG_MEAN,
                                                     set=args.set),
                                   batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   pin_memory=True)


    targetloader_iter = enumerate(targetloader)

    # implement model.optim_parameters(args) to handle different models' lr setting

    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D.zero_grad()
    """
    optimizer_D1 = optim.Adam(model_D1.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D1.zero_grad()

    optimizer_D2 = optim.Adam(model_D2.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D2.zero_grad()
    """

    if args.gan == 'Vanilla':
        bce_loss = torch.nn.BCEWithLogitsLoss()
    elif args.gan == 'LS':
        bce_loss = torch.nn.MSELoss()
    seg_loss = torch.nn.CrossEntropyLoss(ignore_index=255)

    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True)

    # labels for adversarial training
    source_label = 0
    target_label = 1

    # set up tensor board
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    writer = SummaryWriter(args.log_dir)

    for i_iter in range(args.num_steps):
        # pdb.set_trace()
        loss_seg_value1 = 0
        loss_seg_value2 = 0
        adv_loss_value = 0
        d_loss_value = 0

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)
        optimizer_D.zero_grad()
        adjust_learning_rate(optimizer_D, i_iter)

        """
        optimizer_D1.zero_grad()
        optimizer_D2.zero_grad()
        adjust_learning_rate_D(optimizer_D1, i_iter)
        adjust_learning_rate_D(optimizer_D2, i_iter)
        """

        for sub_i in range(args.iter_size):

            # train G

            # don't accumulate grads in D
            for param in model_D.parameters():
                param.requires_grad = False
            """
            for param in model_D1.parameters():
                param.requires_grad = False

            for param in model_D2.parameters():
                param.requires_grad = False
            """

            # train with source

            _, batch = trainloader_iter.__next__()

            images, labels, _, _ = batch
            images = images.to(device)
            labels = labels.long().to(device)
            # pdb.set_trace()
            # images.size() == [1, 3, 720, 1280]
            pred1, pred2 = model(images)
            # pred1, pred2 size == [1, 19, 91, 161]
            pred1 = interp(pred1)
            pred2 = interp(pred2)
            # size (1, 19, 720, 1280)
            # pdb.set_trace()

            feature = nn.Softmax(dim=1)(pred1)
            softmax_out = nn.Softmax(dim=1)(pred2)

            loss_seg1 = seg_loss(pred1, labels)
            loss_seg2 = seg_loss(pred2, labels)
            loss = loss_seg2 + args.lambda_seg * loss_seg1
            # pdb.set_trace()
            # proper normalization
            loss = loss / args.iter_size
            # TODO: uncomment
            loss.backward()
            loss_seg_value1 += loss_seg1.item() / args.iter_size
            loss_seg_value2 += loss_seg2.item() / args.iter_size
            # pdb.set_trace()
            # train with target

            _, batch = targetloader_iter.__next__()

            for params in model_D.parameters():
                params.requires_grad_(requires_grad = False)

            images, _, _ = batch
            images = images.to(device)
            # pdb.set_trace()
            # images.size() == [1, 3, 720, 1280]
            pred_target1, pred_target2 = model(images)
            
            # pred_target1, 2 == [1, 19, 91, 161]
            pred_target1 = interp_target(pred_target1)
            pred_target2 = interp_target(pred_target2)
            # pred_target1, 2 == [1, 19, 720, 1280]
            # pdb.set_trace()

            feature_target = nn.Softmax(dim=1)(pred_target1)
            softmax_out_target = nn.Softmax(dim=1)(pred_target2)

            # features = torch.cat((pred1, pred_target1), dim=0)
            # outputs = torch.cat((pred2, pred_target2), dim=0)
            # features.size() == [2, 19, 720, 1280]
            # softmax_out.size() == [2, 19, 720, 1280]
            # pdb.set_trace()
            # transfer_loss = CDAN([features, softmax_out], model_D, None, None, random_layer=None)
            D_out_target = CDAN([feature_target, softmax_out_target], model_D, None, None, random_layer=None)
            dc_target = torch.FloatTensor(D_out_target.size()).fill_(0).cuda()
            # pdb.set_trace()
            adv_loss = args.lambda_adv * nn.BCEWithLogitsLoss()(D_out_target, dc_target)
            adv_loss = adv_loss / args.iter_size
            # pdb.set_trace()
            # classifier_loss = nn.BCEWithLogitsLoss()(pred2, 
            #        torch.FloatTensor(pred2.data.size()).fill_(source_label).cuda())
            # pdb.set_trace()
            adv_loss.backward()
            adv_loss_value += adv_loss.item()
            # optimizer_D.step()
            #TODO: normalize loss?

            for params in model_D.parameters():
                params.requires_grad_(requires_grad = True)

            feature = feature.detach()
            softmax_out = softmax_out.detach()
            D_out = CDAN([feature, softmax_out], model_D, None, None, random_layer=None)
            
            dc_source = torch.FloatTensor(D_out.size()).fill_(1).cuda()
            # d_loss = CDAN(D_out, dc_source, None, None, random_layer=None)
            d_loss = nn.BCEWithLogitsLoss()(D_out, dc_source)
            d_loss = d_loss / args.iter_size
            # pdb.set_trace()
            d_loss.backward()
            d_loss_value += d_loss.item()

            feature_target = feature_target.detach()
            softmax_out_target = softmax_out_target.detach()
            D_out_target = CDAN([feature_target, softmax_out_target], model_D, None, None, random_layer=None)
            
            dc_target = torch.FloatTensor(D_out_target.size()).fill_(0).cuda()
            d_loss = nn.BCEWithLogitsLoss()(D_out_target, dc_target)
            d_loss = d_loss / args.iter_size
            # pdb.set_trace()
            d_loss.backward()
            d_loss_value += d_loss.item()

            continue

        optimizer.step()
        optimizer_D.step()

        scalar_info = {
            'loss_seg1': loss_seg_value1,
            'loss_seg2': loss_seg_value2,
            'generator_loss': adv_loss_value,
            'discriminator_loss': d_loss_value,
        }

        if i_iter % 10 == 0:
            for key, val in scalar_info.items():
                writer.add_scalar(key, val, i_iter)
        # pdb.set_trace()
        print('exp = {}'.format(args.snapshot_dir))
        print(
        'iter = {0:8d}/{1:8d}, loss_seg1 = {2:.3f} loss_seg2 = {3:.3f} generator = {4:.3f}, discriminator = {5:.3f}'.format(
            i_iter, args.num_steps, loss_seg_value1, loss_seg_value2, adv_loss_value, d_loss_value))

        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '.pth'))
            torch.save(model_D.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '_D.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '.pth'))
            torch.save(model_D.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_D.pth'))

            ###### also record latest saved iteration #######
            args_dict['learning_rate'] = optimizer.param_groups[0]['lr']
            args_dict['learning_rate_D'] = optimizer_D.param_groups[0]['lr']
            args_dict['start_steps'] = i_iter 

            args_dict_file = args.snapshot_dir + '/args_dict_{}.json'.format(i_iter)
            with open(args_dict_file, 'w') as f:
                json.dump(args_dict, f)

            ###### also record latest saved iteration #######

    writer.close()


if __name__ == '__main__':
    main()
