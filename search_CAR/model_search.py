import  torch
from    torch import nn
import  torch.nn.functional as F
from    operations import OPS, FactorizedReduce, ReLUConvBN
from    genotypes import PRIMITIVES, Genotype
from attention import AttentionStem, AttentionConv
from torch.autograd import Variable

class MixedLayer(nn.Module):
    """
    a mixtures output of 8 type of units.

    we use weights to aggregate these outputs while training.
    and softmax to select the strongest edges while inference.
    """
    def __init__(self, c, stride):
        """

        :param c: 16
        :param stride: 1
        """
        super(MixedLayer, self).__init__()
        self.layers = nn.ModuleList()
        for primitive in PRIMITIVES:
            # create corresponding layer
            layer = OPS[primitive](c, stride, False)
            self.layers.append(layer)

    def forward(self, x, weights):
        """

        :param x: data
        :param weights: alpha,[op_num:8], the output = sum of alpha * op(x)
        :return:
        """
        res = [w * layer(x) for w, layer in zip(weights, self.layers)]
        # element-wise add by torch.add
        res = sum(res)
        return res


class Cell(nn.Module):

    def __init__(self, steps, multiplier, cp, c, reduction, reduction_prev):
        """

        :param steps: 4, number of layers inside a cell
        :param multiplier: 4
        :param cpp: 48
        :param cp: 48
        :param c: 16
        :param reduction: indicates whether to reduce the output maps width
        :param reduction_prev: when previous cell reduced width, s1_d = s0_d//2
        in order to keep same shape between s1 and s0, we adopt prep0 layer to
        reduce the s0 width by half.
        """
        super(Cell, self).__init__()

        # indicating current cell is reduction or not
        self.reduction = reduction
        self.reduction_prev = reduction_prev

        # preprocess0 deal with output from prev_prev cell
        # if reduction_prev:
        #     # if prev cell has reduced channel/double width,
        #     # it will reduce width by half
        #     self.preprocess0 = FactorizedReduce(cpp, c, affine=False)
        # else:
        #     self.preprocess0 = ReLUConvBN(cpp, c, 1, 1, 0, affine=False)
        # preprocess1 deal with output from prev cell
        self.preprocess1 = ReLUConvBN(cp, c, 1, 1, 0, affine=False)

        # steps inside a cell
        self.steps = steps # 3
        self.multiplier = multiplier # 4

        self.layers = nn.ModuleList()

        for i in range(self.steps):
            # for each i inside cell, it connects with all previous output
            # plus previous two cells' output
            # for reduction cell, it will reduce the heading 2 inputs only
            stride = 2 if reduction and i < 1 else 1
            layer = MixedLayer(c, stride)
            self.layers.append(layer)

    def forward(self, s1, weights):
        """

        :param s0:
        :param s1:
        :param weights: [14, 8]
        :return:
        """
        # print('s0:', s0.shape,end='=>')
        # s0 = self.preprocess0(s0) # [40, 48, 32, 32], [40, 16, 32, 32]
        # print(s0.shape, self.reduction_prev)
        # print('s1:', s1.shape,end='=>')
        s1 = self.preprocess1(s1) # [40, 48, 32, 32], [40, 16, 32, 32]
        # print(s1.shape)

        states = [s1]
        offset = 0
        # for each node, receive input from all previous intermediate nodes and s0, s1
        for i in range(self.steps): # 3
            # [40, 16, 32, 32]
            s = self.layers[i](states[i], weights[i])
            # append one state since s is the elem-wise addition of all output
            states.append(s)
            # print('node:',i, s.shape, self.reduction)

        # concat along dim=channel
        return states[-1] # 6 of [40, 16, 32, 32]


class Network(nn.Module):

    """
    stack number:layer of cells and then flatten to fed a linear layer
    """
    def __init__(self, c , num_classes, layers, criterion, steps=3, multiplier=3, stem_multiplier=1):
        """

        :param c: 64
        :param num_classes: 10
        :param layers: number of cells of current network
        :param criterion:
        :param steps: nodes num inside cell
        :param multiplier: output channel of cell = multiplier * ch
        :param stem_multiplier: output channel of stem net = stem_multiplier * ch
        """
        super(Network, self).__init__()

        self.c = c
        self.num_classes = num_classes
        self.layers = layers
        self.criterion = criterion
        self.steps = steps
        self.multiplier = multiplier


        # stem_multiplier is for stem network,
        # and multiplier is for general cell
        c_curr = stem_multiplier * c # 64
        # stem network, convert 3 channel to c_curr
        self.stem = nn.Sequential(
            AttentionStem(3, c_curr, kernel_size=3, stride=1, padding=1, groups=1),
            nn.BatchNorm2d(c_curr),
        )

        # c_curr means a factor of the output channels of current cell
        # output channels = multiplier * c_curr
        cp, c_curr = c_curr, c # 64，64
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            # for layer in the middle [1/3, 2/3], reduce via stride=2

            if i in [layers // 3, 2 * layers // 3]:
                c_curr *= 2
                reduction = True
            else:
                reduction = False


            # [cp, h, h] => [multiplier*c_curr, h/h//2, h/h//2]
            # the output channels = multiplier * c_curr
            cell = Cell(steps, multiplier, cp, c_curr, reduction, reduction_prev)
            # update reduction_prev
            reduction_prev = reduction

            self.cells += [cell]

            cp = c_curr

        # adaptive pooling output size to 1x1
        # self.global_pooling = nn.AdaptiveAvgPool2d(1)
        # since cp records last cell's output channels
        # it indicates the input channel number
        # self.classifier = nn.Linear(cp, num_classes)
        self.conv_feature = nn.Conv2d(
            32, 16, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.feature_projection = nn.Sequential(self.conv_feature, self.bn1)
        self.conv2 = nn.Sequential(nn.Conv2d(80, 32, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(32),
                                   nn.Conv2d(32, 3, kernel_size=1, stride=1))
        # self.head = nn.Sequential(
        #     nn.Conv2d(cp, 3, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(3),
        #     nn.Tanh(),
        # )

        self._initialize_alphas()
        # k is the total number of edges inside single cell, 14


    def new(self):
        """
        create a new model and initialize it with current alpha parameters.
        However, its weights are left untouched.
        :return:
        """
        model_new = Network(self.c, self.num_classes, self.layers, self.criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, x):
        """
        in: torch.Size([3, 3, 32, 32])
        stem: torch.Size([3, 48, 32, 32])
        cell: 0 torch.Size([3, 64, 32, 32]) False
        cell: 1 torch.Size([3, 64, 32, 32]) False
        cell: 2 torch.Size([3, 128, 16, 16]) True
        cell: 3 torch.Size([3, 128, 16, 16]) False
        cell: 4 torch.Size([3, 128, 16, 16]) False
        cell: 5 torch.Size([3, 256, 8, 8]) True
        cell: 6 torch.Size([3, 256, 8, 8]) False
        cell: 7 torch.Size([3, 256, 8, 8]) False
        pool:   torch.Size([16, 256, 1, 1])
        linear: [b, 10]
        :param x:
        :return:
        """
        # print('in:', x.shape)
        # s0 & s1 means the last cells' output
        # print(x.size())
        s1 = self.stem(x) # [b, 3, 32, 32] => [b, 48, 32, 32]
        # print('stem:', s1.shape)
        # print('stem:', s1.shape)

        for i, cell in enumerate(self.cells):
            # weights are shared across all reduction cell or normal cell
            # according to current cell's type, it choose which architecture parameters
            # to use
            # weights = F.softmax(self.alpha_normal, dim=-1)
            if cell.reduction: # if current cell is reduction cell
                if i == 1:
                    weights = F.softmax(self.alpha_reduce1, dim=-1)
                if i == 3:
                    weights = F.softmax(self.alpha_reduce2, dim=-1)
            else:
                if i == 0:
                    weights = F.softmax(self.alpha_normal1, dim=-1)
                if i == 2:
                    weights = F.softmax(self.alpha_normal2, dim=-1)
                if i == 4:
                    weights = F.softmax(self.alpha_normal3, dim=-1)
                        # execute cell() firstly and then assign s0=s1, s1=result
            s0, s1 = s1, cell(s1, weights) # [40, 64, 32, 32]
            if i == 1:
                low_level_feature = s1
            # print('cell:',i, s1.shape, cell.reduction)
            # print('\n')

        # s1 is the last cell's output
        s2 = self.feature_projection(low_level_feature)
        x = F.interpolate(s1, size=(16,16), mode='bilinear', align_corners=True)
        x = torch.cat((x, s2), dim=1)
        x = self.conv2(x)
        out = F.interpolate(x, size=(32,32), mode='bilinear', align_corners=True)

        return out

    def loss(self, x, target):
        """

        :param x:
        :param target:
        :return:
        """
        logits = self(x)
        return self.criterion(logits, target)

    def _initialize_alphas(self):
        k = self.steps
        num_ops = len(PRIMITIVES) # 8

        self.alpha_normal1 = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.alpha_normal2 = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.alpha_normal3 = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.alpha_reduce1 = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.alpha_reduce2 = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self._arch_parameters = [
            self.alpha_normal1,
            self.alpha_normal2,
            self.alpha_normal3,
            self.alpha_reduce1,
            self.alpha_reduce2,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def print_arch_parameters(self):
        return self.alpha_normal1,self.alpha_normal2,self.alpha_normal3,self.alpha_reduce1,self.alpha_reduce2




    def genotype(self):
        """

        :return:
        """
        def _parse(weights):
            """

            :param weights: [3, 7]
            :return:
            """
            gene = []
            for i in range(self.steps): # for each node
                W = weights[i].copy() # [2, 8], [3, 8], ...
                # print(W)
                k_best = None
                for k in range(len(W)): # get strongest ops for current input j->i
                    if k_best is None or W[k] > W[k_best]:
                        k_best = k
                gene.append((PRIMITIVES[k_best], i)) # save ops and input node
            return gene

        # gene_normal = _parse(F.softmax(self.alpha_normal, dim=-1).data.cpu().numpy())
        gene_normal1 = _parse(F.softmax(self.alpha_normal1, dim=-1).data.cpu().numpy())
        gene_normal2 = _parse(F.softmax(self.alpha_normal2, dim=-1).data.cpu().numpy())
        gene_normal3 = _parse(F.softmax(self.alpha_normal3, dim=-1).data.cpu().numpy())
        gene_reduce1 = _parse(F.softmax(self.alpha_reduce1, dim=-1).data.cpu().numpy())
        gene_reduce2 = _parse(F.softmax(self.alpha_reduce2, dim=-1).data.cpu().numpy())


        # print(F.softmax(self.alpha_normal, dim=-1))

        print(F.softmax(self.alpha_normal1, dim=-1))
        print(F.softmax(self.alpha_normal2, dim=-1))
        print(F.softmax(self.alpha_normal3, dim=-1))
        print(F.softmax(self.alpha_reduce1, dim=-1))
        print(F.softmax(self.alpha_reduce2, dim=-1))

        concat = range(self.steps, self.steps + 1)
        # genotype = Genotype(
        #     normal=gene_normal, normal_concat1=concat,
        # )
        genotype = Genotype(
            normal1=gene_normal1, normal_concat1=concat,
            normal2=gene_normal2, normal_concat2=concat,
            normal3=gene_normal3, normal_concat3=concat,
            reduce1=gene_reduce1, reduce_concat1=concat,
            reduce2=gene_reduce2, reduce_concat2=concat,
        )

        return genotype
