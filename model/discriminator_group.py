import torch.nn as nn
import torch.nn.functional as F
import pdb

class FCDiscriminator_group_large(nn.Module):

    def __init__(self, num_classes):
        super(FCDiscriminator_group_large, self).__init__()
        ndf = num_classes*8
        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=3, stride=1, padding=1, groups=num_classes)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1, groups=num_classes)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1, groups=num_classes)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1, groups=num_classes)
        self.conv5 = nn.Conv2d(ndf*8, ndf*16, kernel_size=4, stride=2, padding=1, groups=num_classes)
        self.conv6 = nn.Conv2d(ndf*16, ndf*8, kernel_size=3, stride=1, padding=1, groups=num_classes)
        self.conv7 = nn.Conv2d(ndf*8, ndf*4, kernel_size=3, stride=1, padding=1, groups=num_classes)
        self.classifier = nn.Conv2d(ndf*4, num_classes, kernel_size=4, stride=2, padding=1, groups=num_classes)
        # self.fc = nn.Linear()

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        #self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
        #self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        # pdb.set_trace()
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.conv5(x)
        x = self.leaky_relu(x)
        x = self.conv6(x)
        x = self.leaky_relu(x)
        x = self.conv7(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        #x = self.up_sample(x)
        #x = self.sigmoid(x) 

        return x


class FCDiscriminator_group(nn.Module):

    def __init__(self, num_classes):
        super(FCDiscriminator_group, self).__init__()
        ndf = num_classes*4
        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1, groups=num_classes)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1, groups=num_classes)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1, groups=num_classes)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1, groups=num_classes)
        self.classifier = nn.Conv2d(ndf*8, num_classes, kernel_size=4, stride=2, padding=1, groups=num_classes)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        #self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
        #self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        #x = self.up_sample(x)
        #x = self.sigmoid(x) 

        return x
