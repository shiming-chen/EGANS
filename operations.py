import torch
import torch.nn as nn

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

# class Zero(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super(Zero, self).__init__()
#         self.in_dim = in_dim
#         self.out_dim = out_dim

#     def forward(self, x):
#         return torch.einsum('ba, c->bc', x, torch.zeros([self.out_dim]).cuda())

class Zero(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Zero, self).__init__()

    def forward(self, x):
        return 0

class FCReLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FCReLU, self).__init__()
        self.operation = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            # nn.BatchNorm1d(out_dim),  # BN for fast training
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.operation(x)

class FCLeakyReLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FCLeakyReLU, self).__init__()
        self.operation = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            # nn.BatchNorm1d(out_dim),  # BN for fast training
            nn.LeakyReLU(0.2, inplace=False),
        )

    def forward(self, x):
        return self.operation(x)


class FCReLUdrop(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FCReLUdrop, self).__init__()
        self.operation = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5)
        )

    def forward(self, x):
        return self.operation(x)

class FCLeakyReLUdrop(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FCLeakyReLUdrop, self).__init__()
        self.operation = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout(p=0.5)
        )

    def forward(self, x):
        return self.operation(x)

class FCBNReLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FCBNReLU, self).__init__()
        self.operation = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),  # BN for fast training
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.operation(x)

class FCBNLeakyReLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FCBNLeakyReLU, self).__init__()
        self.operation = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),  # BN for fast training
            nn.LeakyReLU(0.2, inplace=False),
        )

    def forward(self, x):
        return self.operation(x)

class FCBNReLUdrop(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FCBNReLUdrop, self).__init__()
        self.operation = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),  # BN for fast training
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5)
        )

    def forward(self, x):
        return self.operation(x)

class FCBNLeakyReLUdrop(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FCBNLeakyReLUdrop, self).__init__()
        self.operation = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),  # BN for fast training
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout(p=0.5)
        )

    def forward(self, x):
        return self.operation(x)

class FCLNReLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FCLNReLU, self).__init__()
        self.operation = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),  # BN for fast training
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.operation(x)

class FCLNLeakyReLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FCLNLeakyReLU, self).__init__()
        self.operation = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),  # BN for fast training
            nn.LeakyReLU(0.2, inplace=False),
        )

    def forward(self, x):
        return self.operation(x)

class FCLNReLUdrop(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FCLNReLUdrop, self).__init__()
        self.operation = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),  # BN for fast training
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5)
        )

    def forward(self, x):
        return self.operation(x)

class FCLNLeakyReLUdrop(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FCLNLeakyReLUdrop, self).__init__()
        self.operation = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),  # BN for fast training
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout(p=0.5)
        )

    def forward(self, x):
        return self.operation(x)

# operation_dict_all = {
#     'fc_relu': lambda in_dim, out_dim: FCReLU(in_dim, out_dim),
#     'fc_lrelu': lambda in_dim, out_dim: FCLeakyReLU(in_dim, out_dim),
#     'fc_bn_relu': lambda in_dim, out_dim: FCBNReLU(in_dim, out_dim),
#     'fc_bn_lrelu': lambda in_dim, out_dim: FCBNLeakyReLU(in_dim, out_dim),
#     'fc_relu_d': lambda in_dim, out_dim: FCReLUdrop(in_dim, out_dim),
#     'fc_lrelu_d': lambda in_dim, out_dim: FCLeakyReLUdrop(in_dim, out_dim),
#     'fc_bn_relu_d': lambda in_dim, out_dim: FCBNReLUdrop(in_dim, out_dim),
#     'fc_bn_lrelu_d': lambda in_dim, out_dim: FCBNLeakyReLUdrop(in_dim, out_dim),
#     'None': lambda in_dim, out_dim: Zero(in_dim, out_dim)
# }

# operation_list_all = [
#     'fc_relu',
#     'fc_lrelu',
#     'fc_bn_relu',
#     'fc_bn_lrelu',
#     'fc_relu_d',
#     'fc_lrelu_d',
#     'fc_bn_relu_d',
#     'fc_bn_lrelu_d',
#     'None']

operation_dict_all = {
    'fc_relu': lambda in_dim, out_dim: FCReLU(in_dim, out_dim),
    'fc_lrelu': lambda in_dim, out_dim: FCLeakyReLU(in_dim, out_dim),
    'fc_relu_d': lambda in_dim, out_dim: FCReLUdrop(in_dim, out_dim),
    'fc_lrelu_d': lambda in_dim, out_dim: FCLeakyReLUdrop(in_dim, out_dim),
    'None': lambda in_dim, out_dim: Zero(in_dim, out_dim)
}

operation_list_all = [
    'fc_relu',
    'fc_lrelu',
    'fc_relu_d',
    'fc_lrelu_d',
    'None']
