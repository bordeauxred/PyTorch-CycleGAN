#!/usr/bin/python3

import argparse
import itertools
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
import cv2
import numpy as np
from utils import tensor2image
from matplotlib import pyplot as plt
from itertools import cycle
from utils import ignore_exceptions

from unet import UNet as Generator
from models import Discriminator
from guided_filter import GuidedFilterTransfrom
from utils import LambdaLR
from utils import Logger, ReplayBuffer
from utils import weights_init_normal
from datasets import ImageDataset, CelebA, InverseTransformImageDataset
from time import time

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--resume', action='store_true', help='continue training from last checkpoint')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG = Generator(3, 3)
netD = Discriminator(3)

if opt.resume:
    print("Loading checkpoint")
    netG = torch.load('output/netR_G.pth')
    netD = torch.load('output/netR_D.pth')
else:
    netG.apply(weights_init_normal)
    netD.apply(weights_init_normal)

if opt.cuda:
    netG.cuda()
    netD.cuda()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor

###################################



criterion_GAN = torch.nn.MSELoss()
criterion_identity = torch.nn.L1Loss()

# currently must be a factor of 16
# epoch, size, batch_size

current_scale = 128
max_scale = 256
max_scale_batch_size = 4
epochs_per_scale = 2

iterations_per_epoch = 100
# Loss plot
logger = Logger(opt.n_epochs, iterations_per_epoch)

lr = opt.lr
ii = 0

l1_weight = 10

forward_transform = GuidedFilterTransfrom(r=5, eps=0.005)




###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):

    if epoch % epochs_per_scale == 0:
        if current_scale < max_scale:

            batch_size = max_scale_batch_size * int(max_scale//current_scale)**2

            print("Current scale: {}, batch size: {}".format(current_scale, batch_size))
            # Dataset loader
            pre_transforms_ = [
                            transforms.Resize(int(current_scale*1.12), Image.BICUBIC),
                            transforms.RandomCrop(current_scale),
                            transforms.RandomHorizontalFlip(),
                            ]
            transforms_ = [
                            transforms.ToTensor(),
                            forward_transform,
                            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

            target_transforms_ = [
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

            dataset = InverseTransformImageDataset("datasets/rome/train/B", pre_transforms_=pre_transforms_, transforms_=transforms_, target_transforms_=target_transforms_)
            dataset_test = InverseTransformImageDataset("datasets/rome/train/A",pre_transforms_=pre_transforms_, transforms_=transforms_, target_transforms_=target_transforms_)
            #dataset = CelebA("/home/msu/Data/celeba", transforms_=transforms_, unaligned=True, attribute = "Young")

            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                    num_workers=opt.n_cpu,drop_last=True)

            dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True,
                                    num_workers=0,drop_last=True)

            dataloader_test = cycle(dataloader_test) # infinite iteration

            input_A = Tensor(batch_size, 3, current_scale, current_scale)
            input_A_T = Tensor(batch_size, 3, current_scale, current_scale)
            input_B = Tensor(batch_size, 3, current_scale, current_scale)
            input_B_T = Tensor(batch_size, 3, current_scale, current_scale)

            _ones = Tensor(batch_size).fill_(1.0)
            _half = Tensor(batch_size).fill_(0.5)
            _zeros = Tensor(batch_size).fill_(0)


            target_fake = Variable(_zeros,requires_grad=False)
            target_real = Variable(_ones,requires_grad=False)

            fake_A_buffer = ReplayBuffer()
            fake_B_buffer = ReplayBuffer()

            optimizer_G = torch.optim.Adam(netG.parameters(),
                                            lr=lr, betas=(0.5, 0.999))
            optimizer_D = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
            lr /= 2
            current_scale += 16

    with ignore_exceptions(Exception):
        for i, batch in enumerate(dataloader):
            netG.train()


            #with torch.autograd.profiler.profile() as prof:
            tic = time()
            # Set model input
            real_A = Variable(input_A.copy_(batch['image']))
            real_A_T = Variable(input_A_T.copy_(batch['target']))

            batch_test = next(dataloader_test)
            real_B = Variable(input_B.copy_(batch_test['image']))
            real_B_T = Variable(input_B_T.copy_(batch_test['target']))

            ###### Generators A2B and B2A ######
            optimizer_G.zero_grad()

            #same_B = netG(real_B)
            #loss_identity = criterion_identity(same_B, real_B)

            # GAN loss
            fake_A_T = netG(real_A)
            pred_fake_A_T = netD(fake_A_T)
            loss_GAN_A2AT = criterion_GAN(pred_fake_A_T, target_real)

            fake_B_T = netG(real_B)
            pred_fake_B_T = netD(fake_B_T)
            loss_GAN_B2BT = criterion_GAN(pred_fake_B_T, target_real)

            fake_A = forward_transform(fake_A_T)
            fake_B = forward_transform(fake_B_T)

            loss_l1_A = criterion_identity(fake_A,real_A)*l1_weight
            loss_l1_B = criterion_identity(fake_B,real_B)*l1_weight
            #import pdb; pdb.set_trace()

            # Total loss
            loss_G = loss_l1_A + loss_l1_B + loss_GAN_A2AT + loss_GAN_B2BT

            loss_G.backward()

            optimizer_G.step()
            #import pdb; pdb.set_trace()

            ###################################

            ###### Discriminator A ######
            optimizer_D.zero_grad()

            # Real loss
            #pred_real_A = netD(real_A)
            #loss_D_real_A = criterion_GAN(pred_real_A, target_A_real)

            # Fake loss
            #fake_A = fake_A_buffer.push_and_pop(fake_A)
            #pred_fake_A = netD(fake_A.detach())
            #pred_fake_A = netD(fake_A)
            #loss_D_fake_A = criterion_GAN(pred_fake_A, target_A_fake)

            # Real loss
            pred_real = netD(real_A_T.detach())
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_A_T = fake_A_buffer.push_and_pop(fake_A_T)
            pred_fake_A_T = netD(fake_A_T.detach())
            loss_D_A_fake = criterion_GAN(pred_fake_A_T, target_fake)

            fake_B_T = fake_B_buffer.push_and_pop(fake_B_T)
            pred_fake_B_T = netD(fake_B_T.detach())
            loss_D_B_fake = criterion_GAN(pred_fake_B_T, target_fake)

            # Total loss
            loss_D = (loss_D_real*2 + loss_D_A_fake + loss_D_B_fake)/4
            #import pdb; pdb.set_trace()
            loss_D.backward()

            optimizer_D.step()
            ###################################
            # Test and log

            #netG.eval()
            #


            #fake_test_B = netG(real_test_A)


            print("time", time() - tic)
            # Progress report (http://localhost:8097)
            logger.log({'loss_G': loss_G, 'loss_l1_A': (loss_l1_A), 'loss_l1_B': (loss_l1_B), 'loss_GAN_A2AT': (loss_GAN_A2AT), 'loss_GAN_B2BT': (loss_GAN_B2BT),
                        'loss_D': loss_D},
                        images={'real_A': real_A, 'real_A_T': real_A_T, 'fake_A': fake_A, 'fake_A_T': fake_A_T, 'real_B': real_B, 'real_B_T': real_B_T, 'fake_B_T': fake_B_T})

            if (ii+1) % 20 == 0:
                print("EPOCH FINISH =======================================")
                print("save checkpoint")
                torch.save(netG, 'output/netG_R.pth')
                torch.save(netD, 'output/netD_R.pth')
                #import pdb; pdb.set_trace()
            #    plt.imsave("log/epoch_{}_A_real.png".format(epoch), np.moveaxis(tensor2image(real_A.data),0,2))
            #    plt.imsave("log/epoch_{}_B_real.png".format(epoch), np.moveaxis(tensor2image(real_B.data),0,2))
                #plt.imsave("log/epoch_{}_A_fake.png".format(epoch), np.moveaxis(tensor2image(fake_A.data),0,2))
                #plt.imsave("log/epoch_{}_B_fake.png".format(epoch), np.moveaxis(tensor2image(fake_B.data),0,2))
            ii += 1
            #break


    """ TODO netG.eval()
    for i, batch in enumerate(dataloader_test):


        real_A = Variable(input_A.copy_(batch['image']))
        real_B = Variable(input_B.copy_(batch['target']))

        fake_B = netG(real_A)
    """

###################################
