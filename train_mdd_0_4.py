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
from model.discriminator import FCDiscriminator
from utils.loss import CrossEntropy2d
from dataset.gta5_dataset import GTA5DataSet
from dataset.cityscapes_dataset import cityscapesDataSet
from mdd.MDD_resnet import MDD

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'DeepLab'
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 1
DATA_DIRECTORY = './data/GTA5'
DATA_LIST_PATH = './dataset/gta5_list/train.txt'
IGNORE_LABEL = 255
#INPUT_SIZE = '1280,720'
INPUT_SIZE = '1024,512'
DATA_DIRECTORY_TARGET = './data/Cityscapes/data'
DATA_LIST_PATH_TARGET = './dataset/cityscapes_list/train.txt'
INPUT_SIZE_TARGET = '1024,512'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 250000
NUM_STEPS_STOP = 150000  # early stopping
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 5000
#SAVE_PRED_EVERY = 1
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005
LOG_DIR = './log'

LEARNING_RATE_D = 1e-4
LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET1 = 0.0002
LAMBDA_ADV_TARGET2 = 0.001
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
    parser.add_argument("--lambda-adv-target1", type=float, default=LAMBDA_ADV_TARGET1,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-adv-target2", type=float, default=LAMBDA_ADV_TARGET2,
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
    parser.add_argument("--restart_from", type = str, default = RESTART_FROM, help = 'restore a whole training')
    parser.add_argument("--start_steps", type = int, default = 0, help = 'where to start the training')
    
    ###### record args #######

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
    snapshot_dir = SNAPSHOT_DIR + 'mdd_' \
        + str(NUM_STEPS) + '_bs{}'.format(BATCH_SIZE) + \
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
        #model = DeeplabMulti(num_classes=args.num_classes)
        model = DeeplabMulti(num_classes=args.num_classes)
        width = 1024
        srcweight = 3
        model = MDD(width=width, use_bottleneck=False, use_gpu=not args.cpu, class_num=args.num_classes, srcweight=srcweight, args=args)
        model.set_train(True)
    
    model.c_net.to(device)

    #### From here, code should not be related to model reload ####
    # but we would need hyperparameters: n_iter, 
    # [lr, momentum, weight_decay, betas](these are all in args)
    # args.snapshot_dir = generate_snapshot_name()


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
    
    # pdb.set_trace()
    targetloader_iter = enumerate(targetloader)

    # implement model.optim_parameters(args) to handle different models' lr setting

    optimizer = optim.SGD(model.get_parameter_list(),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = INVScheduler(gamma=0.001,
                                decay_rate=0.75,
                                init_lr=0.004)


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

    for i_iter in range(args.start_steps, args.num_steps):

        param_groups = model.get_parameter_list()
        group_ratios = [group['lr'] for group in param_groups]
        optimizer = lr_scheduler.next_optimizer(group_ratios, optimizer, i_iter/5)
        optimizer.zero_grad()
        #adjust_learning_rate(optimizer, i_iter)

        total_loss = 0
        for sub_i in range(args.iter_size):

            # train with source

            _, batch = trainloader_iter.__next__()

            # load src
            src_images, src_labels, _, _ = batch
            src_images = src_images.to(device)
            src_labels = src_labels.long().to(device)

            # load target
            _, batch = targetloader_iter.__next__()
            tgt_images, _, _ = batch
            tgt_images = tgt_images.to(device)

            inputs = torch.cat((src_images, tgt_images), dim=0)

            loss = model.get_loss(inputs, src_labels)
            loss = loss / args.iter_size
            loss.backward()
            total_loss += loss.item()

        optimizer.step()

        scalar_info = {
            'loss': total_loss,
        }

        if i_iter % 10 == 0:
            for key, val in scalar_info.items():
                writer.add_scalar(key, val, i_iter)

        print('exp = {}'.format(args.snapshot_dir))
        print(
        'iter = {0:8d}/{1:8d}, loss = {2:.3f}'.format(
            i_iter, args.num_steps, total_loss))

        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(model.c_net.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model.c_net.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '.pth'))

            ###### also record latest saved iteration #######
            args_dict['learning_rate'] = optimizer.param_groups[0]['lr']
            args_dict['start_steps'] = i_iter 

            args_dict_file = args.snapshot_dir + '/args_dict_{}.json'.format(i_iter)
            with open(args_dict_file, 'w') as f:
                json.dump(args_dict, f)

            ###### also record latest saved iteration #######

    writer.close()

class INVScheduler(object):
    def __init__(self, gamma, decay_rate, init_lr=0.001):
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.init_lr = init_lr

    def next_optimizer(self, group_ratios, optimizer, num_iter):
        lr = self.init_lr * (1 + self.gamma * num_iter) ** (-self.decay_rate)
        i=0
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * group_ratios[i]
            i+=1
        return optimizer

if __name__ == '__main__':
    main()
