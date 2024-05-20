import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# A layer of all operations
class MixedLayer(nn.Module):
    def __init__(self, input_dim, output_dim, operation_dict):
        super(MixedLayer, self).__init__()

        self.layers = nn.ModuleList()
        for operation in operation_dict.keys():
            # Create corresponding layer
            layer = operation_dict[operation](input_dim, output_dim)
            self.layers.append(layer)

    def forward(self, x, weights):
        res = [w * layer(x) for w, layer in zip(weights, self.layers)]
        res = sum(res)

        return res

from operations import *

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator,self).__init__()
        layer_sizes = [4096, 2048]
        latent_size= opt.attSize
        input_size = latent_size * 2
        self.fc1 = nn.Linear(input_size, layer_sizes[0])
        self.fc3 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, noise, att):
        z = torch.cat((noise, att), dim=-1)
        x1 = self.lrelu(self.fc1(z))
        x = self.relu(self.fc3(x1))
        return x

#conditional discriminator for inductive
class Discriminator_D1(nn.Module):
    def __init__(self, opt): 
        super(Discriminator_D1, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, 4096)
        self.fc2 = nn.Linear(4096, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1) 
        self.hidden = self.lrelu(self.fc1(h))
        h = self.lrelu(self.fc2(self.hidden))
        return h

# A network with mixed layers
class Network(nn.Module):
    def __init__(self, num_nodes, initial_input_dims, hidden_dim):
        super(Network, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList()
        self.operation_name_list = []
        self.initial_input_dims = initial_input_dims
        self.num_initial_input = len(self.initial_input_dims)

        # Generate all the mixed layer
        for i in range(self.num_nodes):
            # All previous outputs and additional inputs
            for j in range(i + self.num_initial_input):
                if j < self.num_initial_input:  # Input layer
                    layer = MixedLayer(self.initial_input_dims[j], self.hidden_dim[i], operation_dict_all)
                    self.layers.append(layer)
                    self.operation_name_list.append(list(operation_dict_all.keys()))

                else:  # Middle layers
                    layer = MixedLayer(self.hidden_dim[j-self.num_initial_input], self.hidden_dim[i], operation_dict_all)
                    self.layers.append(layer)
                    self.operation_name_list.append(list(operation_dict_all.keys()))
        print("")

    def forward(self, s_0, s_1, genotype):
        states = [s_0, s_1, torch.cat((s_0, s_1), dim=-1)]
        offset = 0

        # Input from all previous layers
        for i in range(self.num_nodes):
            s = sum(
                self.layers[offset + j](cur_state, genotype[offset + j]) for j, cur_state
                in enumerate(states))
            offset += len(states)
            states.append(s)

        # Keep last layer output
        return states[-1]

    def get_operation_name_list(self):
        return self.operation_name_list

class MLP_search(nn.Module):
    def __init__(self, opt, flag, num_nodes = 4):
        super(MLP_search, self).__init__()
        self.num_nodes = opt.num_nodes
        self.att_size = opt.attSize
        self.nz = opt.nz
        self.res_size = opt.resSize
        if flag=='g':
            if self.num_nodes==4:
                self.hidden_dim = [512, 2048, 4096, 2048]
            else:
                self.hidden_dim = [512, 1024, 2048, 4096, 2048]
            self.initial_input_dims = [
                self.att_size,
                self.nz,
                self.att_size + self.nz
            ]
        else:
            if self.num_nodes==4:
                self.hidden_dim = [4096, 2048, 1024, 1]
            else:
                self.hidden_dim = [4096, 2048, 1024, 512, 1]
            self.initial_input_dims = [
                self.att_size,
                self.res_size,
                self.att_size + self.res_size
            ]
        print('self.hidden_dim', self.hidden_dim)

        self.num_initial_input = len(self.initial_input_dims)
        self.network = Network(self.num_nodes, self.initial_input_dims, self.hidden_dim)
        # Get operation list
        self.operation_name_list = self.network.get_operation_name_list()


    def forward(self, noise, att, genotype):
        h = self.network(att, noise, genotype)
        return h