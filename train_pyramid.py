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

#from models import Generator
from unet import UNet as Generator
#from linknet import LinkNet as Generator
from models import Discriminator
from utils import LambdaLR
from utils import Logger, ReplayBuffer, GuidedFilter
from utils import weights_init_normal
from datasets import ImageDataset, CelebA
from time import time
#import pdb; pdb.set_trace()
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=2000, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--resume', action='store_true', help='continue training from last checkpoint')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc, nf = 64)
netG_B2A = Generator(opt.output_nc, opt.input_nc, nf = 64)
netD = Discriminator(opt.input_nc)

if opt.resume:
    print("Loading checkpoint")
    netG_A2B = torch.load('output/netG_A2B.pth')
    netG_B2A = torch.load('output/netG_B2A.pth')
    netD = torch.load('output/netD.pth')
else:
    netG_A2B.apply(weights_init_normal)
    netG_B2A.apply(weights_init_normal)
    netD.apply(weights_init_normal)

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD.cuda()



# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()





#lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
#lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor




def make_pyramid(x, num_levels=2):
    xlist = []
    for i in range(num_levels):
        xlist.append(x)
        x = F.avg_pool2d(x, 2)
    return xlist



###################################


# currently must be a factor of 16
# epoch, size, batch_size

current_scale = 64
max_scale = 128
max_scale_batch_size = 8
epochs_per_scale = 10

iterations_per_epoch = 100
# Loss plot
logger = Logger(opt.n_epochs, iterations_per_epoch)

lr = opt.lr
ii = 0
###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):

    if epoch % epochs_per_scale == 0:
        if current_scale < max_scale:

            batch_size = max_scale_batch_size * int(max_scale//current_scale)**2

            print("Current scale: {}, batch size: {}".format(current_scale, batch_size))
            # Dataset loader
            transforms_ = [
                            transforms.Resize(int(current_scale*1.12), Image.BICUBIC),
                            transforms.RandomCrop(current_scale),
                            #GuidedFilter(),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

            #dataset = ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True)
            dataset = CelebA("/home/msu/Data/celeba", transforms_=transforms_, unaligned=True, attribute = "Male")

            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                    num_workers=opt.n_cpu,drop_last=True)
            input_A = Tensor(batch_size, opt.input_nc, current_scale, current_scale)
            input_B = Tensor(batch_size, opt.output_nc, current_scale, current_scale)

            _ones = Tensor(batch_size).fill_(1.0)
            _half = Tensor(batch_size).fill_(0.0)
            _zeros = Tensor(batch_size).fill_(-1.0)

            target_A_real = Variable(_ones,requires_grad=False)
            target_A_fake = Variable(_half,requires_grad=False)
            target_B_fake = Variable(_half,requires_grad=False)
            target_B_real = Variable(_zeros,requires_grad=False)
            #import pdb; pdb.set_trace()
            #target_A_real = Variable(torch.stack([_ones,_zeros],dim=1),requires_grad=False)
            #target_A_fake = Variable(torch.stack([_half,_ones]),requires_grad=False)
            #target_B_fake = Variable(torch.stack([_half,_ones]),requires_grad=False)
            #target_B_real = Variable(torch.stack([_zeros,_zeros],dim=1),requires_grad=False)

            fake_A_buffer = ReplayBuffer()
            fake_B_buffer = ReplayBuffer()

            optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                            lr=lr, betas=(0.5, 0.999))
            optimizer_D = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
            lr /= 2
            current_scale += 16

    for i, batch in enumerate(dataloader):
        #with torch.autograd.profiler.profile() as prof:
        tic = time()
        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        #b, c, w, h = real_A.shape[0]
        #real_Az = torch.stack[[(real_A, torch.zeros(b, 1, w, h)],dim=1)
        #real_Bz = torch.stack[[(real_B, torch.zeros(b, 1, w, h)],dim=1)

        # Identity loss
        # G_A2B(B) should equal B if real B is fed


        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B)*5.0


        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A)*5.0

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake_B = netD(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake_B, target_B_real)

        fake_A = netG_B2A(real_B)
        pred_fake_A = netD(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake_A, target_A_real)

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()

        optimizer_G.step()


        ###################################

        ###### Discriminator A ######
        optimizer_D.zero_grad()

        real_A_pyr = make_pyramid(real_A)
        real_B_pyr = make_pyramid(real_B)

        fake_A = fake_A_buffer.push_and_pop(fake_A)
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        fake_A_pyr = make_pyramid(fake_A)
        fake_B_pyr = make_pyramid(fake_B)

        loss_D_real_A = 0
        loss_D_fake_A = 0
        loss_D_real_B = 0
        loss_D_fake_B = 0
        #import pdb; pdb.set_trace()
        for idx in range(len(real_B_pyr)):
            # Real loss
            pred_real_A = netD(real_A_pyr[idx].detach())
            loss_D_real_A += criterion_GAN(pred_real_A, target_A_real)

            # Fake loss
            pred_fake_A = netD(fake_A_pyr[idx].detach())
            loss_D_fake_A += criterion_GAN(pred_fake_A, target_A_fake)

            # Real loss
            pred_real_B = netD(real_B_pyr[idx].detach())
            loss_D_real_B += criterion_GAN(pred_real_B, target_B_real)

            # Fake loss
            pred_fake_B = netD(fake_B_pyr[idx].detach())
            loss_D_fake_B += criterion_GAN(pred_fake_B, target_B_fake)

        # Total loss
        loss_D = (loss_D_real_A + loss_D_fake_A + loss_D_real_B + loss_D_fake_B)*0.25/4

        loss_D.backward()

        optimizer_D.step()
        ###################################
        print("time", time() - tic)
        # Progress report (http://localhost:8097)
        logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B), 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                    'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': loss_D},
                    images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})

        if i > iterations_per_epoch:
            print("EPOCH FINISH =======================================")
            print("save checkpoint")
            torch.save(netG_A2B, 'output/netG_A2B.pth')
            torch.save(netG_B2A, 'output/netG_B2A.pth')
            torch.save(netD, 'output/netD.pth')
            #import pdb; pdb.set_trace()
            plt.imsave("log/epoch_{}_A_real.png".format(epoch), np.moveaxis(tensor2image(real_A.data),0,2))
            plt.imsave("log/epoch_{}_B_real.png".format(epoch), np.moveaxis(tensor2image(real_B.data),0,2))
            plt.imsave("log/epoch_{}_A_fake.png".format(epoch), np.moveaxis(tensor2image(fake_A.data),0,2))
            plt.imsave("log/epoch_{}_B_fake.png".format(epoch), np.moveaxis(tensor2image(fake_B.data),0,2))
            break



###################################