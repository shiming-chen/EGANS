from __future__ import absolute_import, division, print_function
import numpy as np
import random
from operations import *

class GanAlgorithm():
    def __init__(self, opt):
        self.genotypes = {}
        self.num_initial_input = opt.num_initial_input
        self.num_node = opt.num_nodes
        self.operation_name_list = []
        # Generate all the mixed layer
        for i in range(self.num_node):
            # All previous outputs and additional inputs
            for j in range(i + self.num_initial_input):
                if j < self.num_initial_input:  # Input layer
                    self.operation_name_list.append(list(operation_dict_all.keys()))
        # Alpha list for each operation
        self.num_eg = int((self.num_node+2*self.num_initial_input-1)*self.num_node/2)
        self.num_op= len(self.operation_name_list[0])
        genotype_init = []
        for i in range(self.num_eg):
            genotype_init.append(np.ones(self.num_op)) 
        self.genotype_init = np.stack(genotype_init)

    def encode(self, genotype):
        lists = [0 for i in range(self.num_eg)]
        for i in range(len(lists)):
            lists[i] = str(genotype[i])
        return tuple(lists)

    def clean(self, genotype):
        node=[0 for i in range(self.num_node-1)]
        node_=[0 for i in range(self.num_node-1)]
        input_node=[0 for i in range(self.num_initial_input)]

        for i in range(2):
            offset = 0
            for i in range(self.num_node):
                for j in range(self.num_initial_input):
                    input_node[j] += genotype[offset+j,-1]
                offset += self.num_initial_input+i
            for i in range(self.num_initial_input):
                input_node[i] = input_node[i]//self.num_node
            while((sum(input_node)==self.num_initial_input-1) and (input_node[-1]==1)):
                i = np.random.randint(0,self.num_initial_input)
                if input_node[i]==1:
                    genotype[-self.num_node-self.num_initial_input+i+1,:]=np.zeros(self.num_op)
                    genotype[-self.num_node-self.num_initial_input+i+1,0]=1                
                    input_node[i]=0
            offset = 0
            for i in range(self.num_node):
                if i<self.num_node-1:
                    node_[i] = (genotype[offset:offset+self.num_initial_input+i,-1]-1).sum()
                for j in range(self.num_initial_input+i):
                    if j-3>=0:
                        node[j-3] += genotype[offset,-1]-1
                    offset += 1
            offset = 0
            offset_ = self.num_initial_input
            for i in range(self.num_node-1):
                if node_[i]==0:
                    offset_2 = offset_
                    for j in range(self.num_node-1-i):
                        offset_2 += self.num_initial_input+i
                        genotype[offset_2,:]=np.zeros(self.num_op)
                        genotype[offset_2,-1]=1
                        offset_2 += j+1
                offset_ += self.num_initial_input+i+1
                if node[i]==0:
                    for j in range(self.num_initial_input+i):
                        genotype[offset,:]=np.zeros(self.num_op)
                        genotype[offset,-1]=1
                        offset+=1
                else:
                    offset += self.num_initial_input+i
        if genotype[-self.num_node-self.num_initial_input+1:,-1].sum()==(self.num_node+self.num_initial_input-1):
            genotype[-self.num_node,0]=1
            genotype[-self.num_node,-1]=0
        return genotype 

    def search(self):
        new_genotype = self.sample()
        t = self.encode(new_genotype)
        while(t in self.genotypes):
            new_genotype = self.sample()
            new_genotype = self.clean(new_genotype)
            t = self.encode(new_genotype)
        self.genotypes[t] = new_genotype
        return new_genotype

    def sample(self):
        genotype = np.zeros((self.num_eg, self.num_op), dtype=int)
        for i in range(self.num_eg):
            if np.random.rand()<0.5:
                x1 = self.num_op-1
            else:
                x1 = random.randint(0, self.num_op-2)
            x2 = np.zeros(self.num_op)
            x2[x1]=1
            genotype[i,:]=x2
        return genotype

    def judge_repeat(self, new_genotype):
        t = self.encode(new_genotype)
        return t in self.genotypes
    
    def update(self, genotypes):
        self.genotypes = {}
        for i in range(genotypes.shape[0]):
             t = self.encode(genotypes[i])
             self.genotypes[t] = genotypes[i]