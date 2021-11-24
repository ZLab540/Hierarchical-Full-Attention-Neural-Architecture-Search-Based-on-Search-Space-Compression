import  os,sys,time, glob
import  numpy as np
import  torch
import  utils
import  logging
import  argparse
import  torch.nn as nn
from    torch import optim
import  torchvision.datasets as dset
import  torch.backends.cudnn as cudnn
import  torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as vutils
from    model_search import Network
from    arch import Arch
import random


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='..\data', help='location of the data corpus')
parser.add_argument('--batchsz', type=int, default=32, help='batch size')
parser.add_argument('--lr', type=float, default=0.025, help='init learning rate')
parser.add_argument('--lr_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--wd', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=200, help='report frequency')
parser.add_argument('--gpu', type=int, default=1, help='gpu device id')
parser.add_argument('--epochs', type=int, default=20, help='num of training epochs')
parser.add_argument('--init_ch', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=5, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_len', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--exp_path', type=str, default='/media/lab540/disk2/why/code/pre_search_11.24/exp', help='experiment name')
parser.add_argument('--seed', type=int, default=88, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping range')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training/val splitting')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_lr', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_wd', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--overlapPred',type=int,default=0,help='overlapping edges')
args = parser.parse_args()

utils.create_exp_dir(args.exp_path, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.exp_path, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
device = torch.device('cuda:0')

path4= args.exp_path + '/result'
utils.create_exp_dir1(path4)

path1= args.exp_path + '/result/real'
path2= args.exp_path + '/result/cropped'
path3= args.exp_path + '/result/recon'
utils.create_exp_dir1(path1)
utils.create_exp_dir1(path2)
utils.create_exp_dir1(path3)


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    cudnn.enabled = True
    torch.manual_seed(args.seed)

    # ================================================
    total, used = os.popen(
        'nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader'
            ).read().split('\n')[args.gpu].split(',')
    total = int(total)
    used = int(used)

    print('Total GPU mem:', total, 'used:', used)

    args.unrolled = False


    logging.info('GPU device = %d' % args.gpu)
    logging.info("args = %s", args)


    criterion = nn.L1Loss().to(device)


    model = Network(args.init_ch, 10, args.layers, criterion).to(device)

    logging.info("Total param size = %f MB", utils.count_parameters_in_MB(model))

    # this is the optimizer to optimize
    optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd)

    #select 100 classes from Imagenet
    traindir = '/media/lab540/disk2/why/datasets/Small_Image/train'

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_data = dset.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize(40),
            transforms.CenterCrop(32),
            # transforms.Resize(args.imageSize),
            # transforms.RandomResizedCrop(args.imageSize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    num_train = len(train_data)  
    print(num_train)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))  

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batchsz,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=False, num_workers=4)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batchsz,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:]),
        pin_memory=False, num_workers=4)

    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [5,10,15,20,40], gamma=0.5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

    arch = Arch(model, args)

    for epoch in range(args.epochs):

        lr = scheduler.get_last_lr()[0]
        logging.info('\nEpoch: %d lr: %e', epoch, lr)

        genotype = model.genotype()
        logging.info('Genotype: %s', genotype)

        para = model.print_arch_parameters()
        logging.info('para: %s',para)

        # pre-search
        train_obj = train(train_queue, valid_queue, model, arch, criterion, optimizer, lr ,epoch )
        logging.info('train loss: %f', train_obj)

        # valid_obj = infer(valid_queue, model, criterion)
        # logging.info('valid loss: %f', valid_obj)

        scheduler.step()

        utils.save(model, os.path.join(args.exp_path, 'search.pt'))

def train(train_queue, valid_queue, model, arch, criterion, optimizer, lr, epoch):
    """

    :param train_queue: train loader
    :param valid_queue: validate loader
    :param model: network
    :param arch: Arch class
    :param criterion:
    :param optimizer:
    :param lr:
    :return:
    """
    losses = utils.AverageMeter()
    # top1 = utils.AverageMeter()
    # top5 = utils.AverageMeter()

    valid_iter = iter(valid_queue)

    input_real = torch.FloatTensor(args.batchsz, 3, args.imageSize, args.imageSize)
    input_cropped = torch.FloatTensor(args.batchsz, 3, args.imageSize, args.imageSize)

    input_real_search = torch.FloatTensor(args.batchsz, 3, args.imageSize, args.imageSize)
    input_cropped_search = torch.FloatTensor(args.batchsz, 3, args.imageSize, args.imageSize)
    
    for step, (x, target) in enumerate(train_queue):

        batchsz = x.size(0)
        model.train()

        # [b, 3, 32, 32], [40]
        # x = x.to(device)
        x_search, target_search = next(valid_iter) # [b, 3, 32, 32], [b]
        # x_search = x_search.to(device)
        x_search = torch.tensor([item.cpu().detach().numpy() for item in x_search]).cuda()

        real_cpu= x
        input_real.resize_(real_cpu.size()).copy_(real_cpu)
        input_cropped.resize_(real_cpu.size()).copy_(real_cpu)
        input_cropped = utils.mask(input_cropped)
        input_real = input_real.to(device)
        input_cropped = input_cropped.to(device)

        # print(input_real.size())

        real_cpu1 = x_search
        input_real_search.resize_(real_cpu.size()).copy_(real_cpu1)
        input_cropped_search.resize_(real_cpu.size()).copy_(real_cpu1)
        input_cropped_search = utils.mask(input_cropped_search)
        input_real_search = input_real_search.to(device)
        input_cropped_search = input_cropped_search.to(device)

        # 1. update alpha
        arch.step(input_real, input_cropped, input_real_search, input_cropped_search, lr, optimizer, unrolled=args.unrolled)

        output = model(input_cropped)
        # print(output.size())
        loss = criterion(output, input_real)

        # 2. update weight
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        losses.update(loss.item(), batchsz)

        if step % args.report_freq == 0:
            vutils.save_image(real_cpu,
                    os.path.join(args.exp_path, 'result/real/real_samples_epoch_%03d_step_%d.png' % (epoch,step)))
            vutils.save_image(input_cropped.data,
                    os.path.join(args.exp_path, 'result/cropped/cropped_samples_epoch_%03d_step_%d.png' % (epoch,step)))
            recon_image = input_cropped.clone()
            recon_image.data[:,:,:] = output.data
            vutils.save_image(recon_image.data,
                    os.path.join(args.exp_path, 'result/recon/recon_center_samples_epoch_%03d_step_%d.png' % (epoch,step)))
            logging.info('Step:%03d loss:%f ', step, losses.avg)

    return losses.avg


def infer(valid_queue, model, criterion):
    """

    :param valid_queue:
    :param model:
    :param criterion:
    :return:
    """
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (x, target) in enumerate(valid_queue):

            x, target = x.to(device), target.cuda(non_blocking=True)
            batchsz = x.size(0)

            logits = model(x)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            losses.update(loss.item(), batchsz)
            top1.update(prec1.item(), batchsz)
            top5.update(prec5.item(), batchsz)

            if step % args.report_freq == 0:
                logging.info('>> Validation: %3d %e %f %f', step, losses.avg, top1.avg, top5.avg)

    return top1.avg, losses.avg


if __name__ == '__main__':
    main()
