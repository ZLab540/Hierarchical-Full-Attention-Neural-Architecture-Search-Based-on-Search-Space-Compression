import  os
import  sys
import  time
import  glob
import  numpy as np
import  torch
import  utils
import  logging
import  argparse
import  torch.nn as nn
import  genotypes
import  torch.utils
import  torchvision.datasets as dset
import  torch.backends.cudnn as cudnn
import  shutil
from    model import NetworkCIFAR as Network
# from torch.utils.tensorboard import SummaryWriter
import math

parser = argparse.ArgumentParser("cifar10")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batchsz', type=int, default=56, help='batch size')
parser.add_argument('--lr', type=float, default=0.04, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=200, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=500, help='num of training epochs')
parser.add_argument('--init_ch', type=int, default=96, help='num of init channels')
parser.add_argument('--layers', type=int, default=5, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.1, help='drop path probability')
parser.add_argument('--exp_path', type=str, default='trainpath', help='experiment name')
parser.add_argument('--seed', type=int, default=23, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--pretrained-model', type=bool, default=False)
args = parser.parse_args()

args.save = args.exp_path + '0823'
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# writer = SummaryWriter('dir')
#
# def adjust_learning_rate(optimizer, epoch, args):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = args.lr * (0.1 ** (epoch // 80))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

def main():


    np.random.seed(args.seed)
    torch.cuda.set_device(0)
    cudnn.benchmark = True
    cudnn.enabled = True
    torch.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    genotype = eval("genotypes.%s" % args.arch)
    print(genotype)
    model = Network(args.init_ch, 10, args.layers, args.auxiliary, genotype).cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.wd
    )

    t= 20
    T= args.epochs
    n_t = 0.5
    lambda1 = lambda epoch: (0.7 * epoch / t + 0.3) if epoch < t else 0.0001 if n_t * (
                1 + math.cos(math.pi * (epoch - t) / (T - t))) < 0.0001 else n_t * (
                1 + math.cos(math.pi * (epoch - t) / (T - t)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [80, 160], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    if args.pretrained_model:
        filename = 'best_model_494_ckpt.tar'
        print('filename :: ', filename)
        file_path = os.path.join('./checkpoint', filename)
        checkpoint = torch.load(file_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint['best_acc']
        print('Load model, Start_epoch: {0}, Acc: {1}'.format(start_epoch, best_acc))
        logging.info('Load model,  Start_epoch: {0}, Acc: {1}'.format(start_epoch, best_acc))

    else:
        start_epoch = 0
        best_acc = 0.0



    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batchsz, shuffle=True, pin_memory=True, num_workers=0)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batchsz, shuffle=False, pin_memory=True, num_workers=0)

    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20,50,80], gamma=0.1)



    for epoch in range(start_epoch,args.epochs+1):
        # logging.info('epoch: %f', epoch)
        lr = scheduler.get_last_lr()[0]
        logging.info('\nEpoch: %d lr: %e', epoch, lr)
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc, train_obj = train(train_queue, model, criterion, epoch, optimizer)
        logging.info('train_acc: %f', train_acc)

        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc: %f', valid_acc)

        # writer.add_scalar('train_acc', train_acc, epoch)
        # writer.add_scalar('train_obj', train_obj, epoch)
        #
        # writer.add_scalar('valid_acc', valid_acc, epoch)
        # writer.add_scalar('valid_obj', valid_obj, epoch)

        scheduler.step()

        is_best = valid_acc > best_acc
        best_acc = max(valid_acc, best_acc)

        logging.info('best_acc: %f', best_acc)

        # if not os.path.isdir('checkpoint'):
        #     os.mkdir('checkpoint')
        # filename = 'model_' + str(epoch)  + '_ckpt.tar'
        # print('filename :: ', filename)

        # utils.save(model, os.path.join(args.save, 'trained.pt'))
        # print('saved to: trained.pt')
        save_checkpoint({
            'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.save)


    # writer.close()

def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)

def train(train_queue, model, criterion, epoch, optimizer):

    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.train()

    for step, (x, target) in enumerate(train_queue):
        # adjust_learning_rate(optimizer, epoch, args)
        x = x.cuda()
        target = target.cuda(non_blocking=True)

        optimizer.zero_grad()
        logits, logits_aux = model(x)
        # print(loss.grad)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = x.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
            for param_group in optimizer.param_groups:
                # print(",  Current learning rate is: {}".format(param_group['lr']))
                logging.info("Current learning rate is: {}".format(param_group['lr']))
            # writer.add_scalar('train_Loss', objs.avg, step)
            # writer.add_scalar('train_top1', top1.avg, step)
            # writer.add_scalar('train_top5', top5.avg, step)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):

    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.eval()

    for step, (x, target) in enumerate(valid_queue):
        x = x.cuda()
        target = target.cuda(non_blocking=True)

        with torch.no_grad():
            logits, _ = model(x)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = x.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('>>Validation: %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
            # writer.add_scalar('infer_Loss', objs.avg, step)
            # writer.add_scalar('infer_top1', top1.avg, step)
            # writer.add_scalar('infer_top5', top5.avg, step)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
