from __future__ import print_function
import argparse
import os
import time
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import util 
import classifier
import classifier2
import model
from plot import plot_genotype
import dateutil
from dateutil import tz
import numpy as np
from tqdm import tqdm
from search_algs import GanAlgorithm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='CUB', help='CUB/SUN/AWA2/FLO')
parser.add_argument('--dataroot', default='./EGANS/data/xlsa17/data/', help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--syn_num', type=int, default=400, help='number features to generate per class')
parser.add_argument('--gzsl', action='store_true', default=True, help='enable generalized zero-shot learning')
parser.add_argument('--preprocessing', action='store_true', default=True, help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=312, help='size of semantic features')
parser.add_argument('--nz', type=int, default=312, help='size of the latent z vector')
parser.add_argument('--nepoch', type=int, default=5, help='number of epochs to train for')
parser.add_argument('--epochs', type=int, default=5, help='# of training epochs')
parser.add_argument('--warmup_nepoch', type=int, default=2, help='number of epochs to warm up')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--cls_weight', type=float, default=0.2, help='weight of the classification loss')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--pretrain_classifier', default='', help="path to pretrain classifier (to continue training)")
parser.add_argument('--outf', default='./EGANS/output/checkpoint/', help='folder to output data and model checkpoints')
parser.add_argument('--geo_dir', default='./EGANS/output/genotypes/', help='folder to save the searched architectures')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
parser.add_argument('--num_individual', type=int, default=50, help='# num of population')
parser.add_argument('--num_nodes', type=int, default=5, help='# num of population')
parser.add_argument('--num_initial_input', type=int, default=3, help='# num of population')
parser.add_argument('--num_train', type=int, default=1, help='# num of train_times')
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--slow', type=float, default=0.1, help='beta1 for adam. default=0.5')
parser.add_argument('--regular_weight', type=float, default=0.0, help='weight regular')

opt = parser.parse_args()
print(opt)


time_str = time.strftime("%m-%d-%H-%M", time.localtime())
exp_dir =  opt.outf + opt.dataset + '_' + time_str + '_G_search_{}'.format(opt.regular_weight)
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

with open(exp_dir + '/G_searching_init.txt', 'a') as file:
    file.write(str(opt))

# load data
data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)
n_train = data.ntrain
split = n_train // 10 * 9
indices = list(range(n_train))
random.shuffle(indices)
valid_index = indices[:200]

netG_search = model.MLP_search(opt, 'g')
netD = model.Discriminator_D1(opt)

gan_alg = GanAlgorithm(opt)
genotypes = np.stack([gan_alg.search() for i in range(opt.num_individual)], axis=0)
genotype_init = gan_alg.genotype_init

cls_criterion = nn.NLLLoss()

input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
noise = torch.FloatTensor(opt.batch_size, opt.nz)
one = torch.tensor(1.)
mone = one * -1
input_label = torch.LongTensor(opt.batch_size)

input_res_val, input_label_val, input_att_val = data.next_batch_darts(valid_index, len(valid_index))
noise_val = torch.FloatTensor(len(valid_index), opt.nz)

if opt.cuda:
    netD.to(opt.device)
    netG_search.to(opt.device)
    netG_search.to(opt.device)
    netD.to(opt.device)
    input_res = input_res.to(opt.device)
    noise, input_att = noise.to(opt.device), input_att.to(opt.device)
    one = one.to(opt.device)
    mone = mone.to(opt.device)
    cls_criterion.to(opt.device)
    input_label = input_label.to(opt.device)
    input_res_val, input_label_val, input_att_val = input_res_val.to(opt.device), input_label_val.to(opt.device), input_att_val.to(opt.device)
    noise_val = noise_val.to(opt.device)

def generate_syn_feature(netG, genotype, classes, attribute, num):
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
        output = netG(Variable(syn_noise, volatile=True), Variable(syn_att, volatile=True), genotype)
        syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i*num, num).fill_(iclass)

    return syn_feature, syn_label

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

def sample():
    batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)
    input_res.copy_(batch_feature).to(opt.device)
    input_att.copy_(batch_att).to(opt.device)
    input_label.copy_(util.map_label(batch_label, data.seenclasses)).to(opt.device)

def sample_darts(index):
    batch_feature, batch_label, batch_att = data.next_batch_darts(index, opt.batch_size)
    input_res.copy_(batch_feature).to(opt.device)
    input_att.copy_(batch_att).to(opt.device)
    input_label.copy_(util.map_label(batch_label, data.seenclasses)).to(opt.device)

def search_evol_arch(gen_net, dis_net, pretrain_cls, genotypes, gan_alg):
    offsprings = gen_offspring(genotypes, gan_alg)
    genotypes_new = np.concatenate((genotypes, offsprings), axis=0)
    d_values, cls_values, fid_values = np.zeros(len(genotypes_new)), np.zeros(
        len(genotypes_new)), np.zeros(len(genotypes_new))
    keep_N = len(offsprings)
    for idx, genotype_G in enumerate(tqdm(genotypes_new)):
        d_value, cls_value, fid_value  = validate(gen_net, dis_net, pretrain_cls, genotype_G)
        d_values[idx] = d_value
        cls_values[idx] = cls_value
        fid_values[idx] = fid_value

    keep = np.argsort(d_values)[0:keep_N]
    gan_alg.update(genotypes_new[keep])
    return genotypes_new[keep]

def search_evol_arch1(genotypes, gan_alg):
    offsprings = gen_offspring(genotypes, gan_alg)
    genotypes_new = np.concatenate((genotypes, offsprings), axis=0)
    return genotypes_new

def search_evol_arch2(gen_net, dis_net, genotypes, gan_alg):
    keep_N = opt.num_individual
    num = genotypes.shape[0]
    d_values, cls_values, para_values = np.zeros(num), np.zeros(num), np.zeros(num)

    input_res_val, input_label_val, input_att_val = data.next_batch_darts(valid_index, len(valid_index))

    for idx, genotype_G in enumerate(tqdm(genotypes)):
        d_value, cls_value, para_value  = validate(gen_net, dis_net, genotype_G)
        d_values[idx] = d_value + opt.regular_weight*para_value
        cls_values[idx] = cls_value
        para_values[idx] = para_value
    keep = np.argsort(d_values)[0:keep_N]

    gan_alg.update(genotypes[keep])
    print("best_Loss_D:{}".format(d_values[keep[0]]))
    return genotypes[keep]

def validate(gen_net, dis_net, genotype_G):
    clsloss_all, criticD_fake_all = 0,0
    input_attv_val = Variable(input_att_val).to(opt.device)
    noise_val.normal_(0, 1)
    noise_valv = Variable(noise_val).to(opt.device)
    fake = gen_net(noise_valv, input_attv_val, genotype_G)
    criticD_fake = dis_net(fake.detach(), input_attv_val)
    criticD_fake_all = criticD_fake.mean().item()
    clsloss_all = 0
    para_size = (genotype_G.shape[0]-genotype_G[:,-1].sum())/genotype_G.shape[0]
    return -criticD_fake_all, clsloss_all, para_size

def gen_offspring(alphas, gan_alg):
    num_offspring = alphas.shape[0] * 1
    offsprings = []
    while len(offsprings) != num_offspring:
        rand = np.random.rand()
        if rand < 0.5:
            alphas_c = mutation(alphas[np.random.randint(0, alphas.shape[0])], gan_alg)
        else:
            a, b = np.random.randint(
                0, alphas.shape[0]), np.random.randint(0, alphas.shape[0])
            while(a == b):
                a, b = np.random.randint(
                    0, alphas.shape[0]), np.random.randint(0, alphas.shape[0])
            alphas_c = crossover(alphas[a], alphas[b], gan_alg)
        if not gan_alg.judge_repeat(alphas_c):
            offsprings.append(alphas_c)
            gan_alg.genotypes[gan_alg.encode(alphas_c)] = alphas_c
    offsprings = np.stack(offsprings, axis=0)
    return offsprings

def crossover(alphas_a, alphas_b, gan_alg):
    """Crossover for two individuals."""
    # alpha a
    new_alphas = alphas_a.copy()
    operation = random.randint(0, alphas_a.shape[0]-1)
    while((new_alphas[operation,:] != alphas_a[operation,:]).sum()<1 and (new_alphas[operation,:] != alphas_b[operation,:]).sum()<1):
        for j in range(alphas_a.shape[1]//2): 
            operation = random.randint(0, alphas_a.shape[0]-1)
            new_alphas[operation,:] = alphas_b[operation,:]
        new_alphas = gan_alg.clean(new_alphas)
    return new_alphas

def mutation(alphas_a, gan_alg):
    """Mutation for An individual."""
    new_alphas = alphas_a.copy()
    operation = random.randint(0, alphas_a.shape[0]-1)
    while((new_alphas[operation,:] != alphas_a[operation,:]).sum()<1):
        if np.random.rand()<0.5:
            i = 1
        elif np.random.rand()>0.3:
            i = 2
        else:
            i =4
        for j in range(i):
            operation = random.randint(0, alphas_a.shape[0]-1)
            if np.random.rand()<0.5:
                x1 = alphas_a.shape[1]-1
            else:
                x1 = random.randint(0, alphas_a.shape[1]-2)
            x2 = np.zeros(alphas_a.shape[1])
            x2[x1]=1
            new_alphas[operation,:] = x2
        new_alphas = gan_alg.clean(new_alphas)
    return new_alphas

# train a classifier on seen classes, obtain \theta of Equation (4)
pretrain_cls = classifier.CLASSIFIER(opt, data.train_feature, util.map_label(data.train_label, data.seenclasses), data.seenclasses.size(0), opt.resSize, opt.cuda, 0.001, 0.5, 50, 100, opt.pretrain_classifier)

# freeze the classifier during the optimization
for p in pretrain_cls.model.parameters(): # set requires_grad to False
    p.requires_grad = False

weight_optimizer_G = torch.optim.Adam(
    netG_search.parameters(),
    lr=opt.lr,
    weight_decay=opt.weight_decay
)

weight_scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(
    weight_optimizer_G,
    float(opt.epochs),
    eta_min=1e-1 * opt.lr * opt.slow
)

weight_optimizer_D = torch.optim.Adam(
    netD.parameters(),
    lr=opt.lr,
    weight_decay=opt.weight_decay
)

weight_scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(
    weight_optimizer_D,
    float(opt.epochs),
    eta_min=1e-1 * opt.lr
)


best_acc = 0
# Best genotype
best_genotype = None
for epoch in range(opt.nepoch):
    FP = 0 
    mean_lossD = 0
    mean_lossG = 0
    weight_scheduler_G.step()
    weight_scheduler_D.step()
    if epoch >= opt.warmup_nepoch:
        genotypes = search_evol_arch1(genotypes, gan_alg)
#==================================train GAN===================================#
    for i in tqdm(range(0, data.ntrain * opt.num_train, opt.batch_size), desc='training'):
        ############################
        # (1) Update D network: optimize WGAN-GP objective, Equation (2)
        ###########################

        for iter_d in range(opt.critic_iter):
            sample()
            weight_optimizer_D.zero_grad()
            # train with realG
            # sample a mini-batch
            input_resv_train = Variable(input_res).to(opt.device)
            input_attv_train = Variable(input_att).to(opt.device)
            input_label_train = input_label
            criticD_real = netD(input_resv_train, input_attv_train)
            criticD_real = criticD_real.mean()
            criticD_real.backward(mone)

            if epoch<opt.warmup_nepoch:
                g = genotype_init
            else:
                g = genotypes[random.randint(0, genotypes.shape[0]-1),:,:]
            # train with fakeG
            noise.normal_(0, 1)
            noisev = Variable(noise).to(opt.device)
            fake = netG_search(noisev, input_attv_train, g)
            criticD_fake = netD(fake.detach(), input_attv_train)
            criticD_fake = criticD_fake.mean()
            criticD_fake.backward(one)

            # gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, input_res, fake.data, input_att)
            gradient_penalty.backward()

            Wasserstein_D = criticD_real - criticD_fake
            D_cost = criticD_fake - criticD_real + gradient_penalty
            weight_optimizer_D.step()


        ############################
        # (2) Update G network: optimize WGAN-GP objective, Equation (2)
        ###########################
        
        weight_optimizer_G.zero_grad()
        noise.normal_(0, 1)
        noisev = Variable(noise).to(opt.device)
        fake = netG_search(noisev, input_attv_train,g)
        criticG_fake = netD(fake, input_attv_train)
        criticG_fake = criticG_fake.mean()
        G_cost = -criticG_fake
        # classification loss
        c_errG = cls_criterion(pretrain_cls.model(fake), Variable(input_label_train).to(opt.device))
        errG = G_cost + opt.cls_weight*c_errG
        errG.backward()
        weight_optimizer_G.step()

    mean_lossG /=  data.ntrain / opt.batch_size 
    mean_lossD /=  data.ntrain / opt.batch_size 
    print('[%d/%d] Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist: %.4f, c_errG:%.4f'
              % (epoch, opt.nepoch, D_cost.item(), G_cost.item(), Wasserstein_D.item(), c_errG.item()))

#==================================search genarator===================================#
    netG_search.eval()
    netD.eval()
    if epoch >= opt.warmup_nepoch:
        # evaluate the model, set G to evaluation mode
        genotypes = search_evol_arch2(netG_search, netD, genotypes, gan_alg)
        
        # Generalized zero-shot learning
        if opt.gzsl:
            syn_feature, syn_label = generate_syn_feature(netG_search, genotypes[0], data.unseenclasses, data.attribute, opt.syn_num)
            train_X = torch.cat((data.train_feature, syn_feature), 0)
            train_Y = torch.cat((data.train_label, syn_label), 0)
            nclass = opt.nclass_all
            cls = classifier2.CLASSIFIER(opt, train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, True)
            acc = cls.H
            print('unseen=%.4f, seen=%.4f, h=%.4f' % (cls.acc_unseen, cls.acc_seen, cls.H))            
            cls = classifier2.CLASSIFIER(opt, syn_feature, util.map_label(syn_label, data.unseenclasses), data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, False)
            print('unseen class accuracy= ', cls.acc.item())
        # Zero-shot learning
        else:
            syn_feature, syn_label = generate_syn_feature(netG_search, genotypes[0], data.unseenclasses, data.attribute, opt.syn_num)
            cls = classifier2.CLASSIFIER(opt, syn_feature, util.map_label(syn_label, data.unseenclasses), data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, False)
            acc = cls.acc.item()
            print('unseen class accuracy= ', acc)
        if best_acc < acc:
            best_acc = acc
            best_genotype = genotypes[0]
            print('Best')
            print('generator', genotypes[0])
            # Plot the architecture picture
            if opt.outf:
                plot_genotype(opt,
                    'g',
                    genotypes[0],
                    file_name=opt.dataset+'_NEW_G_' + str(epoch),
                    figure_dir=exp_dir,
                    save_figure=True
                )
                print('Figure saved.')        
    else:
        if opt.gzsl:
            syn_feature, syn_label = generate_syn_feature(netG_search, genotype_init, data.unseenclasses, data.attribute, opt.syn_num)
            train_X = torch.cat((data.train_feature, syn_feature), 0)
            train_Y = torch.cat((data.train_label, syn_label), 0)
            nclass = opt.nclass_all
            cls = classifier2.CLASSIFIER(opt, train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, True)
            acc = cls.H
            print('unseen=%.4f, seen=%.4f, h=%.4f' % (cls.acc_unseen, cls.acc_seen, cls.H))
            cls = classifier2.CLASSIFIER(opt, syn_feature, util.map_label(syn_label, data.unseenclasses), data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, False)
            print('unseen class accuracy= ', cls.acc.item())
        # Zero-shot learning
        else:
            syn_feature, syn_label = generate_syn_feature(netG_search, genotype_init, data.unseenclasses, data.attribute, opt.syn_num)
            cls = classifier2.CLASSIFIER(opt, syn_feature, util.map_label(syn_label, data.unseenclasses), data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, False)
            print('unseen class accuracy= ', cls.acc.item())   

    # reset G to training model
    netG_search.train()
    netD.train()

netG_search.eval()
for i in range(opt.num_individual):
    if opt.gzsl:
        syn_feature, syn_label = generate_syn_feature(netG_search, genotypes[i], data.unseenclasses, data.attribute, opt.syn_num)
        train_X = torch.cat((data.train_feature, syn_feature), 0)
        train_Y = torch.cat((data.train_label, syn_label), 0)
        nclass = opt.nclass_all
        cls = classifier2.CLASSIFIER(opt, train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, True)
        acc = cls.H
        print('final:unseen=%.4f, seen=%.4f, h=%.4f' % (cls.acc_unseen, cls.acc_seen, cls.H))
    # Zero-shot learning
    else:
        syn_feature, syn_label = generate_syn_feature(netG_search, genotypes[i], data.unseenclasses, data.attribute, opt.syn_num)
        cls = classifier2.CLASSIFIER(opt, syn_feature, util.map_label(syn_label, data.unseenclasses), data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, False)
        acc = cls.acc.item()
        print('unseen class accuracy= ', acc)
    if best_acc < acc:
        best_acc = acc
        best_genotype = genotypes[i]
        print('Best')
        print('generator', genotypes[i])
        if opt.outf:
            plot_genotype(opt,
                'g',
                genotypes[i],
                file_name=opt.dataset+'_NEW_G_' + 'final',
                figure_dir=exp_dir,
                save_figure=True
            )
            print('Figure saved.') 

np.save(opt.geo_dir+"/{}_G.npy".format(opt.dataset),best_genotype)
print(opt.geo_dir+"/{}_G.npy".format(opt.dataset))

print('best_acc', best_acc)
print('best_genotype', best_genotype)