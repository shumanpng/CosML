# This code is modified from https://github.com/hytseng0509/CrossDomainFewShot/blob/master/methods/backbone.py

from collections import OrderedDict

import torch
import torch.nn.functional as F

from maml.models.model import Model
import math
from tensorboardX import SummaryWriter


# --- gaussian initialize ---
def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, torch.nn.Conv2d):
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
    elif isinstance(L, torch.nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)

class distLinear(torch.nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = weight_norm(nn.Linear(indim, outdim, bias=False), name='weight', dim=0)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
        self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized)
        scores = 10 * cos_dist
        return scores

# --- flatten tensor ---
class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class ConvBlock(torch.nn.Module):
    maml = False
    def __init__(self, indim, outdim, pool=True, padding=1):
        super(ConvBlock, self).__init__()

        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C = Conv2d_fw(indim, outdim, 3, padding = padding)
            self.BN = BatchNorm2d_fw(outdim)
        else:
            self.C = torch.nn.Conv2d(indim, outdim, 3, padding=padding)
            self.BN = torch.nn.BatchNorm2d(outdim)
        self.relu = torch.nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C, self.BN, self.relu]
        if pool:
            self.pool = torch.nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)

        for layer in self.parametrized_layers:
            init_layer(layer)
        self.trunk = torch.nn.Sequential(*self.parametrized_layers)

    def forward(self, x):
        out = self.trunk(x)
        return out

class ConvNet(torch.nn.Module):
    ## taken from CrossDomainFewShotClassification ##
    def __init__(self, depth, init_num_channels = 64, preset_in_out_channels = None, fix_num_channels = False, flatten = True):
        super(ConvNet, self).__init__()
        self.grads = []
        self.fmaps = []
        trunk = []

        if depth == 1 and preset_in_out_channels is not None:
            indim, outdim = preset_in_out_channels
            B = ConvBlock(indim, outdim, pool=True)
            trunk.append(B)
        elif depth == 2 and preset_in_out_channels is not None:
            indim, outdim = preset_in_out_channels
            for i in range(depth):
                B = ConvBlock(indim, outdim, pool = (i < 4))
                trunk.append(B)
        elif depth == 3 and preset_in_out_channels is not None:
            indim, outdim = preset_in_out_channels
            for i in range(depth):
                B = ConvBlock(indim, outdim, pool = (i < 4))
                trunk.append(B)
        else:
            if fix_num_channels:
                for i in range(depth):
                    indim = 3 if i == 0 else init_num_channels
                    outdim = init_num_channels
                    B = ConvBlock(indim, outdim, pool = (i < 4)) #only pooling for first 4 layers
                    trunk.append(B)
            else:
                for i in range(depth):
                    indim = 3 if i == 0 else init_num_channels*int(math.pow(2, i-1))
                    outdim = init_num_channels if i == 0 else init_num_channels*int(math.pow(2, i))
                    B = ConvBlock(indim, outdim, pool = True)
                    trunk.append(B)

        if flatten:
            trunk.append(Flatten())

        self.trunk = torch.nn.Sequential(*trunk)
        self.final_feat_dim = 512


    def forward(self, x):
        out = self.trunk(x)
        return out


class Linear_fw(torch.nn.Linear): #used in MAML to forward input with fast weight
    def __init__(self, in_features, out_features):
        super(Linear_fw, self).__init__(in_features, out_features)
        self.weight.fast = None #Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast, self.bias.fast) #weight.fast (fast weight) is the temporaily adapted weight
        else:
            out = super(Linear_fw, self).forward(x)
        return out

class Conv2d_fw(torch.nn.Conv2d): #used in MAML to forward input with fast weight
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0, bias = True):
        super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.weight.fast = None
        if not self.bias is None:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None, stride= self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(x, self.weight.fast, self.bias.fast, stride= self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)

        return out

class BatchNorm2d_fw(torch.nn.BatchNorm2d): #used in MAML to forward input with fast weight
    device = 'cuda:0'

    def __init__(self, num_features):
        super(BatchNorm2d_fw, self).__init__(num_features)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        running_mean = torch.zeros(x.data.size()[1]).to(self.device)
        running_var = torch.ones(x.data.size()[1]).to(self.device)

        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.batch_norm(x, running_mean, running_var, self.weight.fast, self.bias.fast, training = True, momentum = 1)
        else:
            out = F.batch_norm(x, running_mean, running_var, self.weight, self.bias, training = True, momentum = 1)
        return out

class PreTrain(torch.nn.Module):
    device = 'cuda:0'
    def __init__(self, model_func, num_class, tf_path = None, loss_type = 'softmax'):
        super(PreTrain, self).__init__()

        self.feature = model_func

        if loss_type == 'softmax':
            self.classifier = torch.nn.Linear(self.feature.final_feat_dim, num_class)
            self.classifier.bias.data.fill_(0)
        self.loss_type = loss_type
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.total_val_it = 0

        self.num_class = num_class
        self.tf_writer = SummaryWriter(log_dir=tf_path) if tf_path is not None else None

    def forward(self, x):
        x = x.to(self.device)
        out = self.feature.forward(x)
        scores = self.classifier.forward(out)
        return scores

    def forward_loss(self, x, y):
        scores = self.forward(x)
        _, predicted = torch.max(scores.data, 1)
        predicted = predicted.to('cpu')
        total = y.size(0)
        correct = (predicted == y).sum().item()
        y = y.to(self.device)

        acc = float(correct)/float(total)
        return acc, self.loss_fn(scores, y)

    def train_loop(self, epoch, train_loader, optimizer, total_it, device):
        print_freq = len(train_loader) // 25
        avg_loss = 0
        avg_acc = 0
        total = 0

        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            acc, loss = self.forward_loss(x, y)
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss + loss.item()
            avg_acc += acc

            if (i + 1) % print_freq == 0:
                print("Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Acc {:f}".format(epoch, i+1, len(train_loader), avg_loss/float(i+1), avg_acc/float(i+1)))
            if (total_it + 1) % 10 == 0:
                self.tf_writer.add_scalar('train/train_loss', loss.item(), total_it + 1)
                self.tf_writer.add_scalar('train/train_acc', acc, total_it + 1)
            total_it += 1
            total = i+1
        return total_it, avg_loss/total, float(avg_acc)/total

    def test_loop(self, val_loader, mode=''):
        correct = 0
        total = 0
        total_loss = 0
        total_acc = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                curr_correct = 0
                curr_total = 0
                outputs = self.forward(x)
                _, predicted = torch.max(outputs.data, 1)
                # ## send predicted back to cpu to compute accuracy
                predicted = predicted.to('cpu')
                curr_total = y.size(0)

                curr_correct = (predicted == y).sum().item()
                correct += curr_correct
                val_acc = float(curr_correct)/float(curr_total)
                total_acc += val_acc

                y = y.to(self.device)
                val_loss =  self.loss_fn(outputs, y)
                total_loss += val_loss.item()
                if (self.total_val_it + 1) % 10 == 0:
                    if self.tf_writer is not None:
                        self.tf_writer.add_scalar('val/val_acc', val_acc, self.total_val_it + 1)
                        self.tf_writer.add_scalar('val/val_loss', val_loss.item(), self.total_val_it + 1)
                self.total_val_it += 1
                total_batches = i+1
        return total_loss/float(total_batches), float(total_acc)/float(total_batches)
