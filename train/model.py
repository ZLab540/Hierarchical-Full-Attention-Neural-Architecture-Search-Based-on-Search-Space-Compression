import  torch
import  torch.nn as nn
from    train_operations import *
from    utils import drop_path
from attention import AttentionStem



class Cell(nn.Module):

    def __init__(self, genotype, C_prev, C, reduction,  normal1, normal2, normal3, reduction1, reduction2,reduction_prev):
        """

        :param genotype:
        :param C_prev_prev:
        :param C_prev:
        :param C:
        :param reduction:
        :param reduction_prev:
        """
        super(Cell, self).__init__()

        # print(C_prev, C)

        # if reduction_prev:
        #     self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        # else:
        #     self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction1:
            op_names, indices = zip(*genotype.reduce1)
            concat = genotype.reduce_concat1
        elif reduction2:
            op_names, indices = zip(*genotype.reduce2)
            concat = genotype.reduce_concat2
        elif normal1:
            op_names, indices = zip(*genotype.normal1)
            concat = genotype.normal_concat1
        elif normal2:
            op_names, indices = zip(*genotype.normal2)
            concat = genotype.normal_concat2
        elif normal3:
            op_names, indices = zip(*genotype.normal3)
            concat = genotype.normal_concat3
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        """

        :param C:
        :param op_names:
        :param indices:
        :param concat:
        :param reduction:
        :return:
        """
        assert len(op_names) == len(indices)

        self._steps = 3
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 1 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self,  s1, drop_prob):
        """

        :param s0:
        :param s1:
        :param drop_prob:
        :return:
        """
        s1 = self.preprocess1(s1)

        states = [s1]
        for i in range(self._steps):
            h1 = states[self._indices[i]]
            op1 = self._ops[i]
            h1 = op1(h1)

            # if self.training and drop_prob > 0.:
            #     if not isinstance(op1, Identity):
            #         h1 = drop_path(h1, drop_prob)

            s = h1
            states += [s]
        return states[-1]


class AuxiliaryHeadCIFAR(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()

        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),  # image size = 2 x 2
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class AuxiliaryHeadImageNet(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 14x14"""
        super(AuxiliaryHeadImageNet, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            # NOTE: This batchnorm was omitted in my earlier implementation due to a typo.
            # Commenting it out for consistency with the experiments in the paper.
            # nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class NetworkCIFAR(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary, genotype):
        super(NetworkCIFAR, self).__init__()

        self._layers = layers
        self._auxiliary = auxiliary

        stem_multiplier = 1
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr),
            # nn.Conv2d(C_curr, 96, kernel_size=1, bias=False),
            # nn.BatchNorm2d(96),
            # nn.ReLU(),
        )
        # C_curr = 96
        # C = C_curr

        C_prev, C_curr = C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [1,3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False

            if i in [0]:
                normal1 = True
                reduction1 = False
                normal2 = False
                reduction2 = False
                normal3 = False
            elif i in [1]:
                normal1 = False
                reduction1 = True
                normal2 = False
                reduction2 = False
                normal3 = False
            elif i in [2]:
                normal1 = False
                reduction1 = False
                normal2 = True
                reduction2 = False
                normal3 = False
            elif i in [3]:
                normal1 = False
                reduction1 = False
                normal2 = False
                reduction2 = True
                normal3 = False
            elif i in [4]:
                normal1 = False
                reduction1 = False
                normal2 = False
                reduction2 = False
                normal3 = True

            cell = Cell(genotype, C_prev, C_curr, reduction, normal1, normal2, normal3, reduction1, reduction2, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev = C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
        self.dropout = nn.Dropout(p=0.5)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        out = self.dropout(out)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux











#
# class NetworkImageNet(nn.Module):
#
#     def __init__(self, C, num_classes, layers, auxiliary, genotype):
#         super(NetworkImageNet, self).__init__()
#         self._layers = layers
#         self._auxiliary = auxiliary
#
#         self.stem0 = nn.Sequential(
#             nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(C // 2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(C),
#         )
#
#         self.stem1 = nn.Sequential(
#             nn.ReLU(inplace=True),
#             nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(C),
#         )
#
#         C_prev_prev, C_prev, C_curr = C, C, C
#
#         self.cells = nn.ModuleList()
#         reduction_prev = True
#         for i in range(layers):
#             if i in [layers // 3, 2 * layers // 3]:
#                 C_curr *= 2
#                 reduction = True
#             else:
#                 reduction = False
#             cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
#             reduction_prev = reduction
#             self.cells += [cell]
#             C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
#             if i == 2 * layers // 3:
#                 C_to_auxiliary = C_prev
#
#         if auxiliary:
#             self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
#         self.global_pooling = nn.AvgPool2d(7)
#         self.classifier = nn.Linear(C_prev, num_classes)
#
#     def forward(self, input):
#         logits_aux = None
#         s0 = self.stem0(input)
#         s1 = self.stem1(s0)
#         for i, cell in enumerate(self.cells):
#             s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
#             if i == 2 * self._layers // 3:
#                 if self._auxiliary and self.training:
#                     logits_aux = self.auxiliary_head(s1)
#         out = self.global_pooling(s1)
#         logits = self.classifier(out.view(out.size(0), -1))
#         return logits, logits_aux
