from __future__ import print_function
import argparse
import os
import time
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import util 
import classifier
import classifier2
import model_retrain
import model
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='CUB', help='CUB/SUN/AWA2/FLO')
parser.add_argument('--dataroot', default='./EGANS/data/xlsa17/data/', help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--syn_num', type=int, default=1800, help='number features to generate per class')
parser.add_argument('--gzsl', action='store_true', default=True, help='enable generalized zero-shot learning')
parser.add_argument('--preprocessing', action='store_true', default=True, help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=1024, help='size of semantic features')
parser.add_argument('--nz', type=int, default=1024, help='size of the latent z vector')
parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--epochs', type=int, default=100, help='# of training epochs')
parser.add_argument('--cls1_nepoch', type=int, default=50, help='# of training cls1 epochs')
parser.add_argument('--cls2_nepoch', type=int, default=25, help='# of training cls2 epochs')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--cls_weight', type=float, default=0.2, help='weight of the classification loss')
parser.add_argument('--lr', type=float, default=0.00001, help='learning rate to train GANs ')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--pretrain_classifier', default='', help="path to pretrain classifier (to continue training)")
parser.add_argument('--outf', default='./EGANS/output/checkpoint/', help='folder to output data and model checkpoints')
parser.add_argument('--geo_dir', default='./EGANS/output/genotypes/', help='folder to save the searched architectures')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--nclass_all', type=int, default=717, help='number of all classes')
parser.add_argument('--num_nodes', type=int, default=5, help='# num of population')
parser.add_argument('--num_initial_input', type=int, default=3, help='# num of population')
parser.add_argument('--original', action='store_true', default=False, help='use original strcture')
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--slow', type=float, default=1, help='beta1 for adam. default=0.5')

opt = parser.parse_args()
print(opt)

time_str = time.strftime("%m-%d-%H-%M", time.localtime())
exp_dir =  opt.outf + opt.dataset + '_' + time_str + '_retrain'
os.makedirs(exp_dir, exist_ok=True)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# load data
data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)

if opt.original==True:
    netG = model.Generator(opt)
    netD = model.Discriminator_D1(opt)
else:
    genotype_G = np.load(opt.geo_dir+"{}_G.npy".format(opt.dataset))
    genotype_D = np.load(opt.geo_dir+"{}_D.npy".format(opt.dataset))
    netG = model_retrain.NetworkRetrain(opt, 'g', genotype_G)
    netD = model_retrain.NetworkRetrain(opt, 'd', genotype_D)
print(netG)
print(netD)

cls_criterion = nn.NLLLoss()

input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
noise = torch.FloatTensor(opt.batch_size, opt.nz)
one = torch.tensor(1.)
mone = one * -1
input_label = torch.LongTensor(opt.batch_size)

if opt.cuda:
    netD.to(opt.device)
    netG.to(opt.device)
    input_res = input_res.to(opt.device)
    noise, input_att = noise.to(opt.device), input_att.to(opt.device)
    one = one.to(opt.device)
    mone = mone.to(opt.device)
    cls_criterion.to(opt.device)
    input_label = input_label.to(opt.device)

def sample():
    batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))

def generate_syn_feature(netG, classes, attribute, num):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
    syn_label = torch.LongTensor(nclass*num) 
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.to(opt.device)
        syn_noise = syn_noise.to(opt.device)
        
    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        output = netG(Variable(syn_noise, volatile=True), Variable(syn_att, volatile=True))
        syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i*num, num).fill_(iclass)

    return syn_feature, syn_label

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr*opt.slow, betas=(opt.beta1, 0.999))


def calc_gradient_penalty(netD, real_data, fake_data, input_att):
    #print real_data.size()
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.to(opt.device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if opt.cuda:
        interpolates = interpolates.to(opt.device)

    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates, Variable(input_att))

    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.to(opt.device)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty


pretrain_cls = classifier.CLASSIFIER(opt, data.train_feature, util.map_label(data.train_label, data.seenclasses), data.seenclasses.size(0), opt.resSize, opt.cuda, 0.001, 0.5, 50, 100, opt.pretrain_classifier)


# freeze the classifier during the optimization
for p in pretrain_cls.model.parameters(): # set requires_grad to False
    p.requires_grad = False
best_acc = 0
best_epoch = 0
best_cls = None
for epoch in range(opt.nepoch):
    FP = 0 
    mean_lossD = 0
    mean_lossG = 0
    for i in range(0, data.ntrain, opt.batch_size):
        ############################
        # (1) Update D network: optimize WGAN-GP objective, Equation (2)
        ###########################
        for p in netD.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update

        for iter_d in range(opt.critic_iter):
            sample()
            netD.zero_grad()
            # train with realG
            # sample a mini-batch
            sparse_real = opt.resSize - input_res[1].gt(0).sum()
            input_resv = Variable(input_res)
            input_attv = Variable(input_att)
            criticD_real = netD(input_resv, input_attv)
            criticD_real = criticD_real.mean()
            criticD_real.backward(mone)

            # train with fakeG
            noise.normal_(0, 1)
            noisev = Variable(noise)
            fake = netG(noisev, input_attv)
            criticD_fake = netD(fake.detach(), input_attv)
            criticD_fake = criticD_fake.mean()
            criticD_fake.backward(one)

            # gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, input_res, fake.data, input_att)
            gradient_penalty.backward()

            Wasserstein_D = criticD_real - criticD_fake
            D_cost = criticD_fake - criticD_real + gradient_penalty
            optimizerD.step()

        ############################
        # (2) Update G network: optimize WGAN-GP objective, Equation (2)
        ###########################
        for p in netD.parameters(): # reset requires_grad
            p.requires_grad = False # avoid computation

        netG.zero_grad()
        input_attv = Variable(input_att)
        noise.normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev, input_attv)
        criticG_fake = netD(fake, input_attv)
        criticG_fake = criticG_fake.mean()
        G_cost = -criticG_fake
        # classification loss
        c_errG = cls_criterion(pretrain_cls.model(fake), Variable(input_label))
        errG = G_cost + opt.cls_weight*c_errG
        errG.backward()
        optimizerG.step()

    mean_lossG /=  data.ntrain / opt.batch_size 
    mean_lossD /=  data.ntrain / opt.batch_size 
    print('[%d/%d] Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist: %.4f, c_errG:%.4f'
              % (epoch, opt.nepoch, D_cost.item(), G_cost.item(), Wasserstein_D.item(), c_errG.item()))

    # evaluate the model, set G to evaluation mode
    netG.eval()
    # Generalized zero-shot learning
    if opt.gzsl:
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
        train_X = torch.cat((data.train_feature, syn_feature), 0)
        train_Y = torch.cat((data.train_label, syn_label), 0)
        nclass = opt.nclass_all
        cls = classifier2.CLASSIFIER(opt, train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, True)
        acc = cls.H
        print('unseen=%.4f, seen=%.4f, h=%.4f' % (cls.acc_unseen, cls.acc_seen, cls.H))
        cls2 = classifier2.CLASSIFIER(opt, syn_feature, util.map_label(syn_label, data.unseenclasses), data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, False)
        print('unseen class accuracy= ',cls2.acc)        

    # Zero-shot learning
    else:
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
        cls = classifier2.CLASSIFIER(opt, syn_feature, util.map_label(syn_label, data.unseenclasses), data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, False)
        acc = cls.acc
        print('unseen class accuracy= ', acc)

    if best_acc < acc:
        best_acc = acc
        best_epoch = epoch
        best_cls = cls
    netG.train()


if opt.gzsl:
    print('best_acc', best_epoch, best_cls.acc_unseen, best_cls.acc_seen, best_cls.H)
else:
    print('best_acc', best_epoch, best_acc)