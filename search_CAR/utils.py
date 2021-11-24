import  os
import  numpy as np
import  torch
import  shutil
import  torchvision.transforms as transforms
from typing import Union, Optional, List, Tuple, Text, BinaryIO
import io
import pathlib
import math
irange = range
import random

class AverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    """

    :param output: logits, [b, classes]
    :param target: [b]
    :param topk:
    :return:
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


class Cutout:
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
    """

    :param args:
    :return:
    """
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def count_parameters_in_MB(model):
    """
    count all parameters excluding auxiliary
    :param model:
    :return:
    """
    return np.sum(v.numel() for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    print('saved to model:', model_path)
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    print('load from model:', model_path)
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

def create_exp_dir1(path):
    if not os.path.exists(path):
        os.mkdir(path)

def mask(input_cropped):
    x1 = random.randint(0, 24)
    x2 = random.randint(0, 24)
    x3 = random.randint(0, 24)
    x4 = random.randint(0, 24)
    y1 = random.randint(0, 24)
    y2 = random.randint(0, 24)
    y3 = random.randint(0, 24)
    y4 = random.randint(0, 24)
    input_cropped[:, 0, int(x1):int(x1 + 8),int(y1):int(y1+8)] = 0
    input_cropped[:, 1, int(x1):int(x1 + 8),int(y1):int(y1+8)] = 0
    input_cropped[:, 2, int(x1):int(x1 + 8),int(y1):int(y1+8)] = 0

    input_cropped[:, 0, int(x2):int(x2 + 8),int(y2):int(y2+8)] = 0
    input_cropped[:, 1, int(x2):int(x2 + 8),int(y2):int(y2+8)] = 0
    input_cropped[:, 2, int(x2):int(x2 + 8),int(y2):int(y2+8)] = 0

    input_cropped[:, 0, int(x3):int(x3 + 8),int(y3):int(y3+8)] = 0
    input_cropped[:, 1, int(x3):int(x3 + 8),int(y3):int(y3+8)] = 0
    input_cropped[:, 2, int(x3):int(x3 + 8),int(y3):int(y3+8)] = 0

    input_cropped[:, 0, int(x4):int(x4 + 8),int(y4):int(y4+8)] = 0
    input_cropped[:, 1, int(x4):int(x4 + 8),int(y4):int(y4+8)] = 0
    input_cropped[:, 2, int(x4):int(x4 + 8),int(y4):int(y4+8)] = 0

    return input_cropped

def save_image(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    fp: Union[Text, pathlib.Path, BinaryIO],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: int = 0,
    format: Optional[str] = None,
) -> None:
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)
