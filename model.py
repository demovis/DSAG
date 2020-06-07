import torch
import torch.nn as nn
import torch.nn.functional as F

from layer import FC, IncidenceConvolution


class CNFNet(nn.Module):
    def __init__(self, opt):
        super(CNFNet, self).__init__()
        self.opt = opt

        # initialize CNFNet
        self.energy_kernel = IncidenceConvolution(opt)
        self.fc_reg = FC(opt)
        self.fc_class = FC(opt)

        self.lmd = torch.nn.Parameter(torch.Tensor([1]))

    def forward(self, un_inc, c_inc, un_f, c_f):
        # calculate energy kernel
        un_y = self.energy_kernel(un_inc)
        c_y = self.energy_kernel(c_inc)

        # concatenate with CNF properties
        un_z = torch.cat((un_f, un_y), 1)
        c_z = torch.cat((c_f, c_y), 1)

        # connect to fully-connected layers
        z_reg_un = self.fc_reg(un_z)
        z_reg_c = self.fc_reg(c_z)
        z_class_un = self.fc_class(un_z)
        z_class_c = self.fc_class(c_z)

        return z_reg_un, z_reg_c, z_class_un, z_class_c

    def get_reg(self, un_inc, un_f):
        un_y = self.energy_kernel(un_inc)
        un_z = torch.cat((un_f, un_y), 1)
        z_reg = self.fc_reg(un_z)

        return z_reg

    def get_class(self, c_inc, c_f):
        c_y = self.energy_kernel(c_inc)
        c_z = torch.cat((c_f, c_y), 1)
        z_class = self.fc_class(c_z)

        return torch.round(torch.sigmoid(z_class))

    def combo_loss(self, pred, un_real_y):
        # MSE loss of uncensored data
        un_loss = F.mse_loss(pred[0], un_real_y)

        # BCE loss of all data
        survival_labels = torch.cat((torch.FloatTensor([0]).repeat(self.opt.batch_size),
                                     torch.FloatTensor([1]).repeat(self.opt.batch_size)))
        c_loss = F.binary_cross_entropy_with_logits(torch.stack(pred[2:]).view(-1), survival_labels)

        # consistence loss
        consist_mul = (torch.stack(pred[:2]) - self.opt.thres).view(-1) * (1 - 2 * survival_labels)
        consist_loss = torch.mean(torch.relu(consist_mul))
        # consist_loss = F.binary_cross_entropy(consist_pred.float().view(-1), survival_labels)
        # print("consist_pred/survival_labels: {} / {}".format(consist_pred, survival_labels))

        ret_str = " | MSE_loss, c_loss, consist_loss: {} / {} / {}".format(un_loss.item(), c_loss.item(),
                                                                           consist_loss.item())

        return un_loss + \
               self.opt.alpha * c_loss + \
               self.opt.beta * consist_loss, ret_str
