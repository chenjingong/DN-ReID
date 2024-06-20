from resnet import resnet50, resnet18
from gen2 import Generator
from SEM import GradientComputation4
from LSM import *
from Net import *
import pdb

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = reduc_ratio//reduc_ratio

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                    padding=0),
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)



        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        '''
                :param x: (b, c, t, h, w)
                :return:
                '''

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        # f_div_C = torch.nn.functional.softmax(f, dim=-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)



class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x, x01):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        b, c, h, w = x.shape
        mask1 = torch.zeros(b, 1, h, w)
        mask1 = mask1.to(x.device)
        x1 = torch.cat([x, mask1], dim=1)
        return x, x1


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t
        self.kernel = torch.nn.Parameter(torch.ones(1, 1, 3, 3), requires_grad=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


    def forward(self, x, x01):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        x01 = 0.114*x01[:,:,:,0:1] + 0.587*x01[:,:,:,1:2] + 0.299*x01[:,:,:,2:3]
        x01 = x01.permute(0, 3, 1, 2) # 64*1*64*64
        mask = (x01 > 220)
        mask = mask.float()
        filled = F.conv2d(mask, self.kernel, padding=1) > 8.0
        filled = F.conv2d(filled.float(), self.kernel, padding=1) < 1.0
        mask = (filled.squeeze(1) == 0).float().unsqueeze(1)
        mask = self.maxpool(self.maxpool(mask))
        x1 = torch.cat([x, mask], dim=1)
        return x, x1


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x


class embed_net(nn.Module):
    def __init__(self,  class_num, no_local= 'on', gm_pool = 'on', arch='resnet50'):
        super(embed_net, self).__init__()

        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)
        self.non_local = no_local
        self.demo = Net(input_nc=3, output_nc=3)
        self.genA2B = Generator(input_nc=200)
        self.SEM4 = GradientComputation4()
        self.LSM = LSM()
        if self.non_local =='on':
            layers=[3, 4, 6, 3]
            non_layers=[0,2,3,0]
            self.NL_1 = nn.ModuleList(
                [Non_local(256) for i in range(non_layers[0])])
            self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
            self.NL_2 = nn.ModuleList(
                [Non_local(512) for i in range(non_layers[1])])
            self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
            self.NL_3 = nn.ModuleList(
                [Non_local(1024) for i in range(non_layers[2])])
            self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
            self.NL_4 = nn.ModuleList(
                [Non_local(2048) for i in range(non_layers[3])])
            self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])


        pool_dim = 2048
        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        self.classifier = nn.Linear(pool_dim, class_num, bias=False)

        self.classifier1 = nn.Linear(256, pool_dim, bias=False)
        self.classifier2 = nn.Linear(512, pool_dim, bias=False)
        self.classifier3 = nn.Linear(1024, pool_dim, bias=False)
        self.classifier4 = nn.Linear(2048, pool_dim, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        self.classifier1.apply(weights_init_classifier)
        self.classifier2.apply(weights_init_classifier)
        self.classifier3.apply(weights_init_classifier)
        self.classifier4.apply(weights_init_classifier)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.gm_pool = gm_pool

    def forward(self, x1, x01, x2, x02, modal=0):
        if modal == 0:
            #pdb.set_trace()
            x1, x01 = self.visible_module(x1, x01) #32*64*64*64
            x2, x02 = self.thermal_module(x2, x02) #32*64*64*64
            x = torch.cat((x1, x2), 0)             #64*64*64*64
            x0 = torch.cat((x01, x02), 0)          #64*65*64*64
            x, x_fc = self.LSM(x, x0)              #64*64*64*64

        elif modal == 1:
            x1, x01  = self.visible_module(x1, x01)
            x, _ = self.LSM(x1, x01)
        elif modal == 2:
            x2, x02 = self.thermal_module(x2, x02)
            x, _ = self.LSM(x2, x02)

        # shared block
        if self.non_local == 'on':
            for i in range(len(self.base_resnet.base.layer1)):
                x = self.base_resnet.base.layer1[i](x)

            x = self.SEM4(x)
            fc_1 = self.classifier1(x.permute(0,2,3,1))
            fc_1 = self.classifier(fc_1)
            x = self.genA2B(x,fc_1,self.training)
            # Layer 2
            for i in range(len(self.base_resnet.base.layer2)):
                x = self.base_resnet.base.layer2[i](x)

            x = self.SEM4(x)
            fc_2 = self.classifier2(x.permute(0,2,3,1))
            fc_2 = self.classifier(fc_2)
            x = self.genA2B(x,fc_2,self.training)

            # Layer 3
            for i in range(len(self.base_resnet.base.layer3)):
                x = self.base_resnet.base.layer3[i](x)

            x = self.SEM4(x)
            fc_3 = self.classifier3(x.permute(0,2,3,1))
            fc_3 = self.classifier(fc_3)
            x = self.genA2B(x,fc_3,self.training)
            # Layer 4
            for i in range(len(self.base_resnet.base.layer4)):
                x = self.base_resnet.base.layer4[i](x)
            
            x = self.SEM4(x)
            fc_4 = self.classifier4(x.permute(0,2,3,1))
            fc_4 = self.classifier(fc_4)
            x = self.genA2B(x,fc_4,self.training)

        else:
            x = self.base_resnet(x)
        if self.gm_pool  == 'on':
            b, c, h, w = x.shape
            x = x.view(b, c, -1)
            p = 3.0
            x_pool = (torch.mean(x**p, dim=-1) + 1e-12)**(1/p)
        else:
            x_pool = self.avgpool(x)
            x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))

        feat = self.bottleneck(x_pool)

        if self.training:
            return x_pool, self.classifier(feat), x_fc
        else:
            return self.l2norm(x_pool), self.l2norm(feat)

