import numpy as np
import torch
import torch.nn as nn
from operations import operation_dict_all, operation_list_all

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class NetworkRetrain(nn.Module):
    def __init__(self, opt, flag, genotype):
        super(NetworkRetrain, self).__init__()
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
        self.layers = nn.ModuleList()
        self.operation_name_list = []
        self.num_initial_input = len(self.initial_input_dims)

        offset = 0
        # Generate all the mixed layer
        for i in range(self.num_nodes):
            # All previous outputs and additional inputs
            for j in range(i + self.num_initial_input):
                if j < self.num_initial_input:  # Input layer
                    layer = operation_dict_all[operation_list_all[genotype[offset+j].argmax()]](self.initial_input_dims[j], self.hidden_dim[i])
                    self.layers.append(layer)
                    self.operation_name_list.append(operation_list_all[genotype[offset+j].argmax()])

                else:  # Middle layers
                    layer = operation_dict_all[operation_list_all[genotype[offset+j].argmax()]](self.hidden_dim[j-self.num_initial_input], self.hidden_dim[i])
                    self.layers.append(layer)
                    self.operation_name_list.append(operation_list_all[genotype[offset+j].argmax()])
            offset += j+1

    def forward(self, s_1, s_0):
        states = [s_0, s_1, torch.cat((s_0, s_1), dim=-1)]
        offset = 0

        # Input from all previous layers
        for i in range(self.num_nodes):
            s = sum(
                self.layers[offset + j](cur_state) for j, cur_state
                in enumerate(states))
            offset += len(states)
            states.append(s)

        # Keep last layer output
        return states[-1]