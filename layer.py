import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse
from torch.nn.modules.module import Module


class IncidenceConvolution(Module):
    """Core operation of CNFNet"""

    def __init__(self, opt):
        super(IncidenceConvolution, self).__init__()
        self.opt = opt

        self.in_features = 1
        self.out_features = opt.num_feat
        self.fc_dim = opt.hidden

        self.fc1 = nn.Linear(1, self.fc_dim)
        self.fc2 = nn.Linear(self.fc_dim, self.fc_dim)
        self.fc3 = nn.Linear(self.fc_dim, 1)

    def forward(self, inc_m):
        """iterate instances and call energy function"""
        out = []
        for instance in inc_m:
            feat = []
            for arr in instance:
                feat.append(self._to_kernel(torch.FloatTensor(arr[0])))
            out.append(torch.FloatTensor(feat))

        return torch.stack(out)

    def _fc_kernal(self, x):
        """calculate the prediction given energy"""

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def _to_kernel(self, arr):
        # normalization
        norm = arr / torch.sum(arr)

        # FC and product

        weight = torch.tensor([self._fc_kernal(_.unsqueeze(0)) for _ in norm])
        prod = torch.mul(weight, norm)

        # sum
        return torch.sum(prod)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class FC(Module):
    """Fully connected layer for predicting runtime"""

    def __init__(self, opt):
        super(FC, self).__init__()
        self.opt = opt

        self.in_features = opt.energy_input_dim
        self.out_features = 1
        self.fc_dim = opt.hidden

        self.fc1 = nn.Linear(self.in_features, self.fc_dim)
        self.fc2 = nn.Linear(self.fc_dim, self.fc_dim)
        self.fc3 = nn.Linear(self.fc_dim, self.fc_dim)
        self.fc4 = nn.Linear(self.fc_dim, self.out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
