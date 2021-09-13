import random
import time
import warnings
import sys
import argparse
import shutil

import torch
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
import os

import wandb

sys.path.append('../../..')
from common.modules.classifier import Classifier
import common.vision.datasets as datasets
import common.vision.models as models
from common.vision.transforms import ResizeImage
from common.utils.metric import accuracy
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.data import ForeverDataIterator
from common.utils.logger import CompleteLogger
from common.utils.seeder import set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    global_step = 0

    if args.seed is not None:
        set_seed(args.seed)
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # Data loading code
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = T.Compose([
        ResizeImage(256),
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize
    ])
    val_transform = T.Compose([
        ResizeImage(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])


    if len(args.data.split(",")) > 1:
        current_datasets = args.data.split(",")
        current_datasets_directories = args.root.split(",")
    else:
        current_datasets = [args.data]
        current_datasets_directories = [args.root]

    for i in range(len(current_datasets)):

        print("Dataset: ", current_datasets[i])

        dataset = datasets.__dict__[current_datasets[i]]
        train_dataset = dataset(root=current_datasets_directories[i], split='train', sample_rate=args.sample_rate,
                                download=True, transform=train_transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.workers, drop_last=True)
        train_iter = ForeverDataIterator(train_loader)
        val_dataset = dataset(root=current_datasets_directories[i], split='test', sample_rate=100, download=True, transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

        # create model
        print("=> using pre-trained model '{}'".format(args.arch))

        if args.arch.startswith("AutoGrow"):
            if not args.pretrained:
                raise Exception("AutoGrow needs pretrained model")
                
            pretrained_dict = torch.load(args.pretrained)
            residual = pretrained_dict["residual"]
            print("Residual taken from pretrained_dict: ", residual)

            current_arch = pretrained_dict["current_arch"]
            current_arch = list(map(int, current_arch.split("-")))
            backbone = models.__dict__["AutoGrow" + residual](current_arch)

            if 'module' in pretrained_dict['net'].keys()[0]:
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in pretrained_dict['net'].items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                pretrained_dict['net'] = new_state_dict

            backbone.load_state_dict(pretrained_dict['net'], strict=False) # NOTE: sometimes it does not match
        else:
            backbone = models.__dict__[args.arch](pretrained=True)
            if args.pretrained:
                print("=> loading pre-trained model from '{}'".format(args.pretrained))
                pretrained_dict = torch.load(args.pretrained, map_location=device)
                backbone.load_state_dict(pretrained_dict, strict=False)
        num_classes = train_dataset.num_classes

        classifier = Classifier(backbone, num_classes)


        if torch.cuda.device_count() > 1 and args.phase != "test":
            print("***USING GPUs:", os.environ["CUDA_VISIBLE_DEVICES"], "***")
            classifier = torch.nn.DataParallel(classifier)

        classifier.to(device)

        # define optimizer and lr scheduler
        optimizer = SGD(classifier.get_parameters(args.lr), lr=args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=args.lr_patience, \
            factor=args.lr_gamma)

        # resume from the best checkpoint
        if args.phase == 'test':
            checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location=device)
            classifier.load_state_dict(checkpoint)
            acc1 = validate(val_loader, classifier, args)
            print(acc1)
            return

        # start training
        best_test_acc = 0.0
        for epoch in range(args.epochs):
            # increment step
            global_step += 1

            # train for one epoch
            train_loss, train_acc = train(train_iter, classifier, optimizer, epoch, args)
            # evaluate on validation set
            test_loss, test_acc = validate(val_loader, classifier, args)

            lr_scheduler.step(test_acc)

            # remember best acc@1 and save checkpoint
            torch.save(classifier.state_dict() if type(classifier) != \
                torch.nn.DataParallel else classifier.module.state_dict(), logger.get_checkpoint_path('latest'))
            if test_acc > best_test_acc:
                shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
            best_test_acc = max(test_acc, best_test_acc)

            lr = optimizer.param_groups[0]['lr']
            print("LR: ", lr)
            wandb.log({"lr": lr}, step=global_step)

            if lr <= 1e-6:
                print("LR lesser than 1e-6. Breaking...")
                break

            wandb.log({current_datasets[i] + "/" + "train/loss": train_loss, current_datasets[i] \
                + "/" + "train/acc": train_acc, \
                current_datasets[i] + "/" + "epoch": epoch}, step=global_step)

            wandb.log({current_datasets[i] + "/" + "test/loss": test_loss, current_datasets[i] \
                + "/" + "test/acc": test_acc, \
                current_datasets[i] + "/" + "test/best_acc": best_test_acc, current_datasets[i] \
                    + "/" + "epoch": epoch}, step=global_step)

            print("best_test_acc = {:3.1f}".format(best_test_acc))
    
    
    logger.close()


def train(train_iter: ForeverDataIterator, model: Classifier, optimizer: SGD,
          epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x, labels = next(train_iter)

        x = x.to(device)
        label = labels.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y, f = model(x)
        cls_loss = F.cross_entropy(y, label)
        loss = cls_loss

        # measure accuracy and record loss
        losses.update(loss.item(), x.size(0))
        cls_acc = accuracy(y, label)[0]
        cls_accs.update(cls_acc.item(), x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return losses.avg, cls_accs.avg

def validate(val_loader: DataLoader, model: Classifier, args: argparse.Namespace) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output, _ = model(images)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return losses.avg, top1.avg


if __name__ == '__main__':
    architecture_names = sorted(
        name for name in models.__dict__
        if not name.startswith("__")
        and callable(models.__dict__[name])
    ) + ["AutoGrow"]
    dataset_names = sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )
    parser = argparse.ArgumentParser(description='Baseline for Finetuning')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA',
                        help='dataset: ' + ' | '.join(dataset_names))
    parser.add_argument('-sr', '--sample-rate', default=100, type=int,
                        metavar='N',
                        help='sample rate of training dataset (default: 100)')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=architecture_names,
                        help='backbone architecture: ' +
                             ' | '.join(architecture_names) +
                             ' (default: resnet50)')
    parser.add_argument('--pretrained', default=None,
                        help="pretrained checkpoint of the backbone. "
                             "(default: None, use the ImageNet supervised pretrained backbone)")
    # training parameters
    parser.add_argument('-b', '--batch-size', default=48, type=int,
                        metavar='N',
                        help='mini-batch size (default: 48)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-patience', type=int, default=3, help='epochs to decay lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--log", type=str, default='baseline',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test'],
                        help="When phase is 'test', only test the model.")
    parser.add_argument("--run-name", type=str, default='', 
                    help="When phase is 'test', only test the model.")
    args = parser.parse_args()

    wandb.init(project="transfer-learning", entity="arjunashok", config=vars(args))
    if args.run_name:
        wandb.run.name = args.run_name
    wandb.config.update({"gpus": os.environ["CUDA_VISIBLE_DEVICES"]}, allow_val_change=True)

    main(args)

    wandb.finish()
