import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys

import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
from model.deeplab import Res_Deeplab
from model.deeplab_multi import DeeplabMulti
from model.deeplab_vgg import DeeplabVGG
from dataset.cityscapes_dataset import cityscapesDataSet
from collections import OrderedDict
import os
from PIL import Image
import pdb
import torch.nn as nn
from scipy.special import softmax

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATA_DIRECTORY = './data/Cityscapes/data'
DATA_LIST_PATH = './dataset/cityscapes_list/val.txt'
# SAVE_PATH = './result/cityscapes'
SAVE_PATH = './result/cityscapes_test'

IGNORE_LABEL = 255
NUM_CLASSES = 19
NUM_STEPS = 500 # Number of images in the validation set.
# RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_multi-ed35151c.pth'
RESTORE_FROM = '/project/AdaptSegNet/snapshots/multi_class_large50000_seg0.1_adv0.001_bs1_12-2-7-11/GTA5_48000.pth'
RESTORE_FROM_VGG = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_vgg-ac4ac9f6.pth'
RESTORE_FROM_ORC = 'http://vllab1.ucmerced.edu/~whung/adaptSeg/cityscapes_oracle-b7b9934.pth'
SET = 'val'

MODEL = 'DeeplabMulti'

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="Model Choice (DeeplabMulti/DeeplabVGG/Oracle).")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")

    parser.add_argument("--restore-from-list", nargs='+', type=str, default=RESTORE_FROM,
                        help="Where restore models parameters from.")
    parser.add_argument("--ensemble", type=str, default="avg",
                        help="Ensemble method.")
    parser.add_argument("--weight", nargs='+', type=float, default=None,
                        help="Weight for ensemble method 'avg'.")
    return parser.parse_args()


def main():
    """Create the model and start the evaluation process."""

    args = get_arguments()
    args.save += '_{}'.format(args.ensemble)

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    device = torch.device("cuda" if not args.cpu else "cpu")

    for i in range(len(args.restore_from_list)):
        restore_from = args.restore_from_list[i]
        print("Model {} {}".format(i, restore_from))
        model_cls_prob_folder = '{}/{}'.format(args.save, restore_from)
        if not os.path.exists(model_cls_prob_folder):
            os.makedirs(model_cls_prob_folder)
        if args.model == 'DeeplabMulti':
            model = DeeplabMulti(num_classes=args.num_classes)
        elif args.model == 'Oracle':
            model = Res_Deeplab(num_classes=args.num_classes)
            if restore_from == RESTORE_FROM:
                restore_from = RESTORE_FROM_ORC
        elif args.model == 'DeeplabVGG':
            model = DeeplabVGG(num_classes=args.num_classes)
            if restore_from == RESTORE_FROM:
                restore_from = RESTORE_FROM_VGG

        if restore_from[:4] == 'http' :
            saved_state_dict = model_zoo.load_url(restore_from)
        else:
            saved_state_dict = torch.load(restore_from)
        ### for running different versions of pytorch
        model_dict = model.state_dict()
        saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_dict}
        model_dict.update(saved_state_dict)
        ###
        model.load_state_dict(saved_state_dict)

        model = model.to(device)

        model.eval()

        testloader = data.DataLoader(cityscapesDataSet(args.data_dir, args.data_list, crop_size=(1024, 512), mean=IMG_MEAN, scale=False, mirror=False, set=args.set),
                                        batch_size=1, shuffle=False, pin_memory=True)

        interp = nn.Upsample(size=(1024, 2048), mode='bilinear', align_corners=True)

        for index, batch in enumerate(testloader):
            if index % 100 == 0:
                print('%d processd' % index)
            image, _, name = batch
            # pdb.set_trace()
            image = image.to(device)

            if args.model == 'DeeplabMulti':
                output1, output2 = model(image)
                output = interp(output2).cpu().data[0].numpy()
            elif args.model == 'DeeplabVGG' or args.model == 'Oracle':
                output = model(image)
                output = interp(output).cpu().data[0].numpy()

            output = output.transpose(1,2,0)
            # softmax output
            output = softmax(output, axis=2)
            name = name[0].split('/')[-1]
            model_cls_prob_file = '{}.npy'.format(name)
            model_cls_prob_file_name = '{}/{}'.format(model_cls_prob_folder, model_cls_prob_file)
            # save file
            np.save(model_cls_prob_file_name, output)

    testloader = data.DataLoader(cityscapesDataSet(args.data_dir, args.data_list, crop_size=(1024, 512), mean=IMG_MEAN, scale=False, mirror=False, set=args.set),
                                    batch_size=1, shuffle=False, pin_memory=True)

    interp = nn.Upsample(size=(1024, 2048), mode='bilinear', align_corners=True)

    for index, batch in enumerate(testloader):
        image, _, name = batch

        name = name[0].split('/')[-1]
        outputs = []
        for restore_from in args.restore_from_list:
            model_cls_prob_folder = '{}/{}'.format(args.save, restore_from)
            model_cls_prob_file = '{}.npy'.format(name)
            model_cls_prob_file_name = '{}/{}'.format(model_cls_prob_folder, model_cls_prob_file)
            outputs.append(np.load(model_cls_prob_file_name))
        outputs = np.stack(outputs, axis=0)
        # load file
        if args.ensemble == "avg":
            output = np.average(outputs, axis=0, weights=args.weight)
        elif args.ensemble == "max":
            output = np.amax(outputs, axis=0)

        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

        output_col = colorize_mask(output)
        output = Image.fromarray(output)

        output.save('%s/%s' % (args.save, name))
        output_col.save('%s/%s_color.png' % (args.save, name.split('.')[0]))


if __name__ == '__main__':
    main()
