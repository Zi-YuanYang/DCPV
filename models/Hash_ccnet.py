import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
import math
import warnings

class GaborConv2d_old(nn.Module):
    '''
    DESCRIPTION: an implementation of the Learnable Gabor Convolution (LGC) layer \n
    INPUTS: \n
    channel_in: should be 1 \n
    channel_out: number of the output channels \n
    kernel_size, stride, padding: 2D convolution parameters \n
    init_ratio: scale factor of the initial parameters (receptive filed) \n
    '''
    def __init__(self, channel_in, channel_out, kernel_size, stride=1, padding=0, init_ratio=1):
        super(GaborConv2d_old, self).__init__()

        # assert channel_in == 1

        self.channel_in = channel_in
        self.channel_out = channel_out

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding      

        self.init_ratio = init_ratio 

        self.kernel = 0

        if init_ratio <=0:
            init_ratio = 1.0
            print('input error!!!, require init_ratio > 0.0, using default...')

        # initial parameters
        self._SIGMA = 9.2 * self.init_ratio
        self._FREQ = 0.057 / self.init_ratio
        self._GAMMA = 2.0

        
        # shape & scale of the Gaussian functioin:
        self.gamma = nn.Parameter(torch.FloatTensor([self._GAMMA]), requires_grad=True)          
        self.sigma = nn.Parameter(torch.FloatTensor([self._SIGMA]), requires_grad=True)
        self.theta = nn.Parameter(torch.FloatTensor(torch.arange(0, channel_out).float()) * math.pi / channel_out, requires_grad=False)

        # frequency of the cosine envolope:
        self.f = nn.Parameter(torch.FloatTensor([self._FREQ]), requires_grad=True)
        self.psi = nn.Parameter(torch.FloatTensor([0]), requires_grad=False)


    def genGaborBank(self, kernel_size, channel_in, channel_out, sigma, gamma, theta, f, psi):
        xmax = kernel_size // 2
        ymax = kernel_size // 2
        xmin = -xmax
        ymin = -ymax

        ksize = xmax - xmin + 1
        y_0 = torch.arange(ymin, ymax + 1).float()    
        x_0 = torch.arange(xmin, xmax + 1).float()

        # [channel_out, channelin, kernel_H, kernel_W]   
        y = y_0.view(1, -1).repeat(channel_out, channel_in, ksize, 1) 
        x = x_0.view(-1, 1).repeat(channel_out, channel_in, 1, ksize) 

        x = x.float().to(sigma.device)
        y = y.float().to(sigma.device)

        # x=x.float()
        # y=y.float()

        # Rotated coordinate systems
        # [channel_out, <channel_in, kernel, kernel>], broadcasting
        x_theta = x * torch.cos(theta.view(-1, 1, 1, 1)) + y * torch.sin(theta.view(-1, 1, 1, 1))
        y_theta = -x * torch.sin(theta.view(-1, 1, 1, 1)) + y * torch.cos(theta.view(-1, 1, 1, 1))  
                
        gb = -torch.exp(
            -0.5 * ((gamma * x_theta) ** 2 + y_theta ** 2) / (8*sigma.view(-1, 1, 1, 1) ** 2)) \
            * torch.cos(2 * math.pi * f.view(-1, 1, 1, 1) * x_theta + psi.view(-1, 1, 1, 1))
    
        gb = gb - gb.mean(dim=[2,3], keepdim=True)

        return gb


    def forward(self, x):
        kernel = self.genGaborBank(self.kernel_size, self.channel_in, self.channel_out, self.sigma, self.gamma, self.theta, self.f, self.psi)
        self.kernel = kernel
        # print(x.shape)
        out = F.conv2d(x, kernel, stride=self.stride, padding=self.padding)

        return out


class GaborConv2d_rand(nn.Module):
    '''
    DESCRIPTION: an implementation of the Learnable Gabor Convolution (LGC) layer \n
    INPUTS: \n
    channel_in: should be 1 \n
    channel_out: number of the output channels \n
    kernel_size, stride, padding: 2D convolution parameters \n
    init_ratio: scale factor of the initial parameters (receptive filed) \n
    '''
    def __init__(self, channel_in, channel_out, kernel_size, stride=1, padding=0, init_ratio=1):
        super(GaborConv2d_rand, self).__init__()

        # assert channel_in == 1

        self.channel_in = channel_in
        self.channel_out = channel_out

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.init_ratio = init_ratio

        self.kernel = 0

        if init_ratio <=0:
            init_ratio = 1.0
            print('input error!!!, require init_ratio > 0.0, using default...')

        # initial parameters
        self._SIGMA = 9.2 * self.init_ratio
        self._FREQ = 0.057 / self.init_ratio
        self._GAMMA = 2.0

        # shape & scale of the Gaussian functioin:
        self.gamma = nn.Parameter(torch.FloatTensor([self._GAMMA]), requires_grad=True)          
        self.sigma = nn.Parameter(torch.FloatTensor([self._SIGMA]), requires_grad=True)
        self.theta = nn.Parameter(torch.FloatTensor(torch.arange(0, channel_out).float()) * math.pi / channel_out, requires_grad=False)
        self.base_theta = nn.Parameter(torch.FloatTensor(torch.arange(0, channel_out).float()) * math.pi / channel_out, requires_grad=False)
        self.interval = math.pi / channel_out

        torch.manual_seed(912) 
        torch.cuda.manual_seed(912) 
        # ori = torch.randint(0,1,[self.channel_out]) * math.pi / 100
        ori = torch.rand([self.channel_out])
        ori = ori * self.interval
        self.real_theta = nn.Parameter(self.base_theta + ori, requires_grad=False)



        # frequency of the cosine envolope:
        self.f = nn.Parameter(torch.FloatTensor([self._FREQ]), requires_grad=True)
        self.psi = nn.Parameter(torch.FloatTensor([0]), requires_grad=False)


    def genGaborBank(self, kernel_size, channel_in, channel_out, sigma, gamma, theta, f, psi):
    
        # torch.manual_seed(seed) 
        # torch.cuda.manual_seed(seed) 

        # self.theta = nn.Parameter(torch.FloatTensor(torch.arange(0, channel_out).float()) * math.pi / channel_out, requires_grad=False)
        
        xmax = kernel_size // 2
        ymax = kernel_size // 2
        xmin = -xmax
        ymin = -ymax

        ksize = xmax - xmin + 1
        y_0 = torch.arange(ymin, ymax + 1).float()    
        x_0 = torch.arange(xmin, xmax + 1).float()

        # [channel_out, channelin, kernel_H, kernel_W]   
        y = y_0.view(1, -1).repeat(channel_out, channel_in, ksize, 1) 
        x = x_0.view(-1, 1).repeat(channel_out, channel_in, 1, ksize) 

        x = x.float().to(sigma.device)
        y = y.float().to(sigma.device)

        # x=x.float()
        # y=y.float()

        # Rotated coordinate systems
        # [channel_out, <channel_in, kernel, kernel>], broadcasting
        x_theta = x * torch.cos(theta.view(-1, 1, 1, 1)) + y * torch.sin(theta.view(-1, 1, 1, 1))
        y_theta = -x * torch.sin(theta.view(-1, 1, 1, 1)) + y * torch.cos(theta.view(-1, 1, 1, 1))  
                
        gb = -torch.exp(
            -0.5 * ((gamma * x_theta) ** 2 + y_theta ** 2) / (8*sigma.view(-1, 1, 1, 1) ** 2)) \
            * torch.cos(2 * math.pi * f.view(-1, 1, 1, 1) * x_theta + psi.view(-1, 1, 1, 1))
    
        gb = gb - gb.mean(dim=[2,3], keepdim=True)

        return gb.cuda()


    def forward(self, x):
        
        # torch.manual_seed(1234) 
        # torch.cuda.manual_seed(1234) 
        # # ori = torch.randint(0,1,[self.channel_out]) * math.pi / 100
        # ori = torch.rand([self.channel_out])
        # ori = ori * self.interval
        # self.theta = nn.Parameter(self.base_theta + ori, requires_grad=False)

        # ori = torch.rand(self.channel_out) * math.pi
        # t, _ = torch.sort(ori)
        # self.theta = nn.Parameter(t,requires_grad=False)

        # kernel = self.genGaborBank(self.kernel_size, self.channel_in, self.channel_out, self.sigma, self.gamma, self.theta, self.f, self.psi)
        
        kernel = self.genGaborBank(self.kernel_size, self.channel_in, self.channel_out, self.sigma, self.gamma, self.real_theta, self.f, self.psi)
        self.kernel = kernel
        # print(x.shape)
        out = F.conv2d(x, kernel, stride=self.stride, padding=self.padding)

        return out



class SELayer(nn.Module):
    def __init__(self, channel, reduction=1):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CompetitiveBlock_Mul_Ord_Comp(nn.Module):
    '''
    DESCRIPTION: an implementation of the Competitive Block::

    [CB = LGC + argmax + PPU] \n

    INPUTS: \n

    channel_in: only support 1 \n
    n_competitor: number of channels of the LGC (channel_out)  \n

    ksize, stride, padding: 2D convolution parameters \n

    init_ratio: scale factor of the initial parameters (receptive filed) \n

    o1, o2: numbers of channels of the conv_1 and conv_2 layers in the PPU, respectively. (PPU parameters)
    '''

    def __init__(self, channel_in, n_competitor, ksize, stride, padding,weight, init_ratio=1, o1=32, o2=12):
        super(CompetitiveBlock_Mul_Ord_Comp, self).__init__()

        # assert channel_in == 1
        self.channel_in = channel_in
        self.n_competitor = n_competitor
        self.ksize = ksize
        self.init_ratio = init_ratio
        
        self.gabor_conv2d = GaborConv2d_rand(channel_in=channel_in, channel_out=n_competitor, kernel_size=ksize, stride=2,
                                        padding=ksize // 2, init_ratio=init_ratio)
        self.gabor_conv2d2 = GaborConv2d_rand(channel_in=n_competitor, channel_out=n_competitor, kernel_size=ksize, stride=2,
                                         padding=ksize // 2, init_ratio=init_ratio)
        ## 2 2 no conv layer
        # soft-argmax
        # self.a = nn.Parameter(torch.FloatTensor([1]))
        # self.b = nn.Parameter(torch.FloatTensor([0]))

        self.argmax = nn.Softmax(dim=1)
        self.argmax_x = nn.Softmax(dim=2)
        self.argmax_y = nn.Softmax(dim=3)
        # PPU
        self.conv1_1 = nn.Conv2d(n_competitor, o1//2, 5, 2, 0)
        self.conv2_1 = nn.Conv2d(n_competitor, o1//2, 5, 2, 0)
        self.maxpool = nn.MaxPool2d(2, 2)

        self.se1 = SELayer(n_competitor)
        self.se2 = SELayer(n_competitor)

        self.weight_chan = weight
        self.weight_spa = (1-weight) / 2
        # print(self.weight_chan)
    def forward(self, x):

        ## RANDOM GABOR
        #self.gabor_conv2d = GaborConv2d_rand(channel_in=self.channel_in, channel_out=self.n_competitor, kernel_size=self.ksize, stride=2,
        #                                padding=self.ksize // 2, init_ratio=self.init_ratio)
        #self.gabor_conv2d2 = GaborConv2d_rand(channel_in=self.n_competitor, channel_out=self.n_competitor, kernel_size=self.ksize, stride=2,
        #                                 padding=self.ksize // 2, init_ratio=self.init_ratio)

        #1-st order
        x = self.gabor_conv2d(x)
        # print(x.shape)
        x1_1 = self.argmax(x)
        x1_2 = self.argmax_x(x)
        x1_3 = self.argmax_y(x)
        x_1 = self.weight_chan * x1_1 + self.weight_spa * (x1_2 + x1_3)

        x_1 = self.se1(x_1)
        x_1 = self.conv1_1(x_1)
        x_1 = self.maxpool(x_1)

        #2-nd order
        x = self.gabor_conv2d2(x)
        x2_1 = self.argmax(x)
        x2_2 = self.argmax_x(x)
        x2_3 = self.argmax_y(x)
        x_2 = self.weight_chan * x2_1 + self.weight_spa * (x2_2 + x2_3)
        x_2 = self.se2(x_2)
        x_2 = self.conv2_1(x_2)
        x_2 = self.maxpool(x_2)

        xx = torch.cat((x_1.view(x_1.shape[0],-1),x_2.view(x_2.shape[0],-1)),dim=1)

        return xx



class CompetitiveBlock_Mul_Ord_Comp_no_Comp(nn.Module):

    def __init__(self, channel_in, n_competitor, ksize, stride, padding,weight, init_ratio=1, o1=32, o2=12):
        super(CompetitiveBlock_Mul_Ord_Comp_no_Comp, self).__init__()

        # assert channel_in == 1
        self.channel_in = channel_in
        self.n_competitor = n_competitor
        self.ksize = ksize
        self.init_ratio = init_ratio
        
        self.gabor_conv2d = GaborConv2d_rand(channel_in=channel_in, channel_out=n_competitor, kernel_size=ksize, stride=2,
                                        padding=ksize // 2, init_ratio=init_ratio)
        self.gabor_conv2d2 = GaborConv2d_rand(channel_in=n_competitor, channel_out=n_competitor, kernel_size=ksize, stride=2,
                                         padding=ksize // 2, init_ratio=init_ratio)
        ## 2 2 no conv layer
        # soft-argmax
        # self.a = nn.Parameter(torch.FloatTensor([1]))
        # self.b = nn.Parameter(torch.FloatTensor([0]))

        # PPU
        self.conv1_1 = nn.Conv2d(n_competitor, o1//2, 5, 2, 0)
        self.conv2_1 = nn.Conv2d(n_competitor, o1//2, 5, 2, 0)
        self.maxpool = nn.MaxPool2d(2, 2)

        self.se1 = SELayer(n_competitor)
        self.se2 = SELayer(n_competitor)

        self.weight_chan = weight
        self.weight_spa = (1-weight) / 2
        # print(self.weight_chan)
    def forward(self, x):

        ## RANDOM GABOR
        self.gabor_conv2d = GaborConv2d_rand(channel_in=self.channel_in, channel_out=self.n_competitor, kernel_size=self.ksize, stride=2,
                                        padding=self.ksize // 2, init_ratio=self.init_ratio)
        self.gabor_conv2d2 = GaborConv2d_rand(channel_in=self.n_competitor, channel_out=self.n_competitor, kernel_size=self.ksize, stride=2,
                                         padding=self.ksize // 2, init_ratio=self.init_ratio)

        #1-st order
        x = self.gabor_conv2d(x)
        # print(x.shape)

        x_1 = self.se1(x)
        x_1 = self.conv1_1(x_1)
        x_1 = self.maxpool(x_1)

        #2-nd order
        x = self.gabor_conv2d2(x)
        x_2 = self.se2(x)
        x_2 = self.conv2_1(x_2)
        x_2 = self.maxpool(x_2)

        xx = torch.cat((x_1.view(x_1.shape[0],-1),x_2.view(x_2.shape[0],-1)),dim=1)

        return xx




class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance::
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)

        From: https://github.com/ronghuaiyang/arcface-pytorch
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        if self.training :
            assert label is not None
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
            sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))

            phi = cosine * self.cos_m - sine * self.sin_m

            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine > self.th, phi, cosine - self.mm)       
            
            one_hot = torch.zeros(cosine.size(), device=cosine.device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)

            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  
            output *= self.s
        else:
            # assert label is None
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
            output = self.s * cosine

        return output



class ccnet(torch.nn.Module):
    '''
    CompNet = CB1//CB2//CB3 + FC + Dropout + (angular_margin) Output\n
    https://ieeexplore.ieee.org/document/9512475
    '''

    def __init__(self, num_classes,weight):
        super(ccnet, self).__init__()

        self.num_classes = num_classes

        self.cb1 = CompetitiveBlock_Mul_Ord_Comp(channel_in=1, n_competitor=9, ksize=35, stride=3, padding=17, init_ratio=1,weight=weight)
        self.cb2 = CompetitiveBlock_Mul_Ord_Comp(channel_in=1, n_competitor=36, ksize=17, stride=3, padding=8, init_ratio=0.5, o2=24,weight=weight)
        self.cb3 = CompetitiveBlock_Mul_Ord_Comp(channel_in=1, n_competitor=9, ksize=7, stride=3, padding=3, init_ratio=0.25,weight=weight)

        self.fc = torch.nn.Linear(13152, 4096)  # <---
        self.fc1 = torch.nn.Linear(4096, 2048)
        self.drop = torch.nn.Dropout(p=0.5)
        # self.arclayer = torch.nn.Linear(1024,num_classes)
        self.arclayer_ = ArcMarginProduct(2048, num_classes, s=30, m=0.5, easy_margin=False)

    def forward(self, x, y=None):
        x1 = self.cb1(x)
        x2 = self.cb2(x)
        x3 = self.cb3(x)

        x = torch.cat((x1, x2, x3), dim=1)

        x1 = self.fc(x)
        x = self.fc1(x1)
        fe = torch.cat((x1,x),dim=1)
        x = self.drop(x)
        x = self.arclayer_(x, y)

        return x, F.normalize(fe, dim=-1)

    def getFeatureCode(self, x):
        x1 = self.cb1(x)
        x2 = self.cb2(x)
        x3 = self.cb3(x)

        x1 = x1.view(x1.shape[0], -1)
        x2 = x2.view(x2.shape[0], -1)
        x3 = x3.view(x3.shape[0], -1)
        x = torch.cat((x1, x2, x3), dim=1)

        x = self.fc(x)
        x = self.fc1(x)
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)

        return x

class ccnet_hash(torch.nn.Module):
    '''
    CompNet = CB1//CB2//CB3 + FC + Dropout + (angular_margin) Output\n
    https://ieeexplore.ieee.org/document/9512475
    '''

    def __init__(self, num_classes,weight):
        super(ccnet_hash, self).__init__()

        self.num_classes = num_classes

        self.cb1 = CompetitiveBlock_Mul_Ord_Comp(channel_in=1, n_competitor=9, ksize=35, stride=3, padding=17, init_ratio=1,weight=weight)
        self.cb2 = CompetitiveBlock_Mul_Ord_Comp(channel_in=1, n_competitor=36, ksize=17, stride=3, padding=8, init_ratio=0.5, o2=24,weight=weight)
        self.cb3 = CompetitiveBlock_Mul_Ord_Comp(channel_in=1, n_competitor=9, ksize=7, stride=3, padding=3, init_ratio=0.25,weight=weight)

        self.fc = torch.nn.Linear(13152, 4096)  # <---
        self.fc1 = torch.nn.Linear(4096, 2048)
        self.drop = torch.nn.Dropout(p=0.5)
        # self.arclayer = torch.nn.Linear(1024,num_classes)
        self.arclayer_ = ArcMarginProduct(2048, num_classes, s=30, m=0.5, easy_margin=False)

    def forward(self, x, y=None):
        x1 = self.cb1(x)
        x2 = self.cb2(x)
        x3 = self.cb3(x)

        x = torch.cat((x1, x2, x3), dim=1)

        x1 = self.fc(x)
        x = self.fc1(x1)
        fe = F.tanh(x)
        # fe = x
        # fe = torch.cat((x1,x),dim=1)
        x = self.drop(x)
        x = self.arclayer_(x, y)

        # return x, fe
        return x, F.normalize(fe, dim=-1)

    def getFeatureCode(self, x):
        x1 = self.cb1(x)
        x2 = self.cb2(x)
        x3 = self.cb3(x)

        x1 = x1.view(x1.shape[0], -1)
        x2 = x2.view(x2.shape[0], -1)
        x3 = x3.view(x3.shape[0], -1)
        x = torch.cat((x1, x2, x3), dim=1)

        x = self.fc(x)
        x = self.fc1(x)
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)

        return x
        # return F.tanh(x)

    def getFeatureCode_binary(self, x):
        x1 = self.cb1(x)
        x2 = self.cb2(x)
        x3 = self.cb3(x)

        x1 = x1.view(x1.shape[0], -1)
        x2 = x2.view(x2.shape[0], -1)
        x3 = x3.view(x3.shape[0], -1)
        x = torch.cat((x1, x2, x3), dim=1)

        x = self.fc(x)
        x = self.fc1(x)
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        med = torch.median(x,dim=1).values
        fe = torch.where(x > med[:, None], torch.tensor(1).cuda(), torch.tensor(0).cuda())
        # fe = torch.where(x>.5, torch.tensor(1).cuda(), torch.tensor(0).cuda())
        return fe


class ccnet_hash_no_comp(torch.nn.Module):
    '''
    CompNet = CB1//CB2//CB3 + FC + Dropout + (angular_margin) Output\n
    https://ieeexplore.ieee.org/document/9512475
    '''

    def __init__(self, num_classes,weight):
        super(ccnet_hash_no_comp, self).__init__()

        self.num_classes = num_classes

        self.cb1 = CompetitiveBlock_Mul_Ord_Comp_no_Comp(channel_in=1, n_competitor=9, ksize=35, stride=3, padding=17, init_ratio=1,weight=weight)
        self.cb2 = CompetitiveBlock_Mul_Ord_Comp_no_Comp(channel_in=1, n_competitor=36, ksize=17, stride=3, padding=8, init_ratio=0.5, o2=24,weight=weight)
        self.cb3 = CompetitiveBlock_Mul_Ord_Comp_no_Comp(channel_in=1, n_competitor=9, ksize=7, stride=3, padding=3, init_ratio=0.25,weight=weight)

        self.fc = torch.nn.Linear(13152, 4096)  # <---
        self.fc1 = torch.nn.Linear(4096, 2048)
        self.drop = torch.nn.Dropout(p=0.5)
        # self.arclayer = torch.nn.Linear(1024,num_classes)
        self.arclayer_ = ArcMarginProduct(2048, num_classes, s=30, m=0.5, easy_margin=False)

    def forward(self, x, y=None):
        x1 = self.cb1(x)
        x2 = self.cb2(x)
        x3 = self.cb3(x)

        x = torch.cat((x1, x2, x3), dim=1)

        x1 = self.fc(x)
        x = self.fc1(x1)
        # fe = F.tanh(x)
        fe = x
        # fe = x
        # fe = torch.cat((x1,x),dim=1)
        x = self.drop(x)
        x = self.arclayer_(x, y)

        # return x, fe
        return x, F.normalize(fe, dim=-1)

    def getFeatureCode(self, x):
        x1 = self.cb1(x)
        x2 = self.cb2(x)
        x3 = self.cb3(x)

        x1 = x1.view(x1.shape[0], -1)
        x2 = x2.view(x2.shape[0], -1)
        x3 = x3.view(x3.shape[0], -1)
        x = torch.cat((x1, x2, x3), dim=1)

        x = self.fc(x)
        x = self.fc1(x)
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)

        return x
        # return F.tanh(x)

    def getFeatureCode_binary(self, x):
        x1 = self.cb1(x)
        x2 = self.cb2(x)
        x3 = self.cb3(x)

        x1 = x1.view(x1.shape[0], -1)
        x2 = x2.view(x2.shape[0], -1)
        x3 = x3.view(x3.shape[0], -1)
        x = torch.cat((x1, x2, x3), dim=1)

        x = self.fc(x)
        x = self.fc1(x)
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        med = torch.median(x,dim=1).values
        fe = torch.where(x > med[:, None], torch.tensor(1).cuda(), torch.tensor(0).cuda())
        # fe = torch.where(x>.5, torch.tensor(1).cuda(), torch.tensor(0).cuda())
        return fe



class ccnet_hash_can(torch.nn.Module):
    '''
    CompNet = CB1//CB2//CB3 + FC + Dropout + (angular_margin) Output\n
    https://ieeexplore.ieee.org/document/9512475
    '''

    def __init__(self, num_classes,weight):
        super(ccnet_hash_can, self).__init__()

        self.num_classes = num_classes

        self.cb1 = CompetitiveBlock_Mul_Ord_Comp(channel_in=1, n_competitor=9, ksize=35, stride=3, padding=17, init_ratio=1,weight=weight)
        self.cb2 = CompetitiveBlock_Mul_Ord_Comp(channel_in=1, n_competitor=36, ksize=17, stride=3, padding=8, init_ratio=0.5, o2=24,weight=weight)
        self.cb3 = CompetitiveBlock_Mul_Ord_Comp(channel_in=1, n_competitor=9, ksize=7, stride=3, padding=3, init_ratio=0.25,weight=weight)

        self.fc = torch.nn.Linear(13152, 4096)  # <---
        self.fc1 = torch.nn.Linear(4096, 2048)
        self.drop = torch.nn.Dropout(p=0.5)
        # self.arclayer = torch.nn.Linear(1024,num_classes)
        self.arclayer_ = ArcMarginProduct(2048, num_classes, s=30, m=0.5, easy_margin=False)

    def forward(self, x, y=None):
        x1 = self.cb1(x)
        x2 = self.cb2(x)
        x3 = self.cb3(x)

        x = torch.cat((x1, x2, x3), dim=1)

        x1 = self.fc(x)
        x = self.fc1(x1)
        fe = F.tanh(x)
        # fe = x
        # fe = x
        # fe = torch.cat((x1,x),dim=1)
        x = self.drop(x)
        x = self.arclayer_(x, y)

        # return x, fe
        return x, F.normalize(fe, dim=-1)

    def getFeatureCode(self, x):
        x1 = self.cb1(x)
        x2 = self.cb2(x)
        x3 = self.cb3(x)

        x1 = x1.view(x1.shape[0], -1)
        x2 = x2.view(x2.shape[0], -1)
        x3 = x3.view(x3.shape[0], -1)
        x = torch.cat((x1, x2, x3), dim=1)

        x = self.fc(x)
        x = self.fc1(x)
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)

        return x
        # return F.tanh(x)

    def getFeatureCode_binary(self, x):
        x1 = self.cb1(x)
        x2 = self.cb2(x)
        x3 = self.cb3(x)

        x1 = x1.view(x1.shape[0], -1)
        x2 = x2.view(x2.shape[0], -1)
        x3 = x3.view(x3.shape[0], -1)
        x = torch.cat((x1, x2, x3), dim=1)

        x = self.fc(x)
        x = self.fc1(x)
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        med = torch.median(x,dim=1).values
        fe = torch.where(x > med[:, None], torch.tensor(1).cuda(), torch.tensor(0).cuda())
        # fe = torch.where(x>.5, torch.tensor(1).cuda(), torch.tensor(0).cuda())
        return fe


    def getFeatureCode_hash_binary(self, x, seed=None):
        x1 = self.cb1(x)
        x2 = self.cb2(x)
        x3 = self.cb3(x)

        x1 = x1.view(x1.shape[0], -1)
        x2 = x2.view(x2.shape[0], -1)
        x3 = x3.view(x3.shape[0], -1)
        x = torch.cat((x1, x2, x3), dim=1)

        x = self.fc(x)
        # x = bio_gen(x,seed=1234)
        x = self.fc1(x)
        if seed == None:
            x = bio_gen(x,seed=1234)
        else:
            x = bio_gen(x,seed=seed)

        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        med = torch.median(x,dim=1).values
        fe = torch.where(x > med[:, None], torch.tensor(1).cuda(), torch.tensor(0).cuda())
        # fe = torch.where(x>.5, torch.tensor(1).cuda(), torch.tensor(0).cuda())
        return fe


def bio_gen(fe, seed=1234, bn_length=None):
    
    if bn_length == None:
        torch.manual_seed(seed) 
        torch.cuda.manual_seed(seed) 
        fe_length = fe.shape[1]
        bn_length = fe_length
        rand_mat = torch.randn(fe_length,bn_length)
        orth_mat, _ = torch.linalg.qr(rand_mat,mode='reduced')
        biohash = torch.matmul(fe,orth_mat.cuda())
    else:
        torch.manual_seed(seed) 
        torch.cuda.manual_seed(seed) 
        fe_length = fe.shape[1]
        # bn_length = fe_length
        rand_mat = torch.randn(fe_length,bn_length)
        orth_mat, _ = torch.linalg.qr(rand_mat,mode='reduced')
        biohash = torch.matmul(fe,orth_mat.cuda())
    return biohash



if __name__== "__main__" :
    inp = torch.randn(256,1,128,128).cuda()
    y = torch.randn(256).cuda()
    net = ccnet_hash_no_comp(600,weight=0.8).cuda()
    out = net(inp,y)
