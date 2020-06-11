import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
from models.fc_ae import FCAutoEncoder
import util
from torchvision import datasets, transforms

def save_state(model, acc):
    print('==> Saving model ...')
    state = {
            'acc': acc,
            'state_dict': model.state_dict(),
            }
    for key in state['state_dict'].keys():
        if 'module' in key:
            state['state_dict'][key.replace('module.', '')] = \
                    state['state_dict'].pop(key)
    torch.save(state, 'models/'+args.arch+'.best.pth.tar')

# def correct_calculate(inputs, outputs):
#     inputs = (inputs + 1) / 2
#     outputs = (outputs + 1) / 2
#     equal = inputs.eq(outputs).float() * inputs

    

def train(epoch):
    model.train()
    correct = 0
    total = 0
    for batch_idx, (inputs, ) in enumerate(train_loader):
        if args.cuda:
            inputs = inputs.cuda()
        optimizer.zero_grad()

        # process the weights including binarization
        bin_op.binarization()

        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()

        # restore weights
        bin_op.restore()
        bin_op.updateBinaryGradWeight()

        optimizer.step()

        total += inputs.size(0)
        correct += inputs.eq(outputs).sum().item()/inputs.size(1)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} | Acc: {:.3f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item(), 100.*correct/total))
    return

def test(evaluate=False):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0

    bin_op.binarization()
    for (inputs, ) in test_loader:
        if args.cuda:
            inputs = inputs.cuda()
        outputs = model(inputs)
        test_loss += criterion(output, target).data.item()
        correct += inputs.eq(outputs).sum().item()/inputs.size(1)

    bin_op.restore()
    
    acc = 100. * float(correct) / len(test_loader.dataset)
    if (acc > best_acc):
        best_acc = acc
        if not evaluate:
            save_state(model, best_acc)

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * args.batch_size, correct, len(test_loader.dataset),
        100. * float(correct) / len(test_loader.dataset)))
    print('Best Accuracy: {:.2f}%\n'.format(best_acc))

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 15 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.lr_epochs))
    print('Learning rate:', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__=='__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch XNOR Testing Compression')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
            help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
            help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=60, metavar='N',
            help='number of epochs to train (default: 60)')
    parser.add_argument('--lr-epochs', type=int, default=15, metavar='N',
            help='number of epochs to decay the lr (default: 15)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
            help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
            help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
            metavar='W', help='weight decay (default: 1e-5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
            help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
            help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
            help='how many batches to wait before logging training status')
    parser.add_argument('--arch', action='store', default='fc_ae',
            help='the autoencoder structure: FCAutoEncoder')
    parser.add_argument('--pretrained', action='store', default=None,
            help='pretrained model')
    parser.add_argument('--evaluate', action='store_true', default=False,
            help='whether to run evaluation')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    print(args)
    
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # generata fake training dataset and testing dataset
    np.random.seed(0)
    size_training = 10000
    size_testing = 1000
    len_sc = 1000
    state_sc = [1., -1.]
    prob_sc = [0.2, 0.8]
    train_samples = np.random.choice(state_sc, size=(size_training, len_sc), p=prob_sc)
    test_samples = np.random.choice(state_sc, size=(size_testing, len_sc), p=prob_sc)
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_samples).float())
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_samples).float())
    
    # load data
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)
    
    # generate the model
    if args.arch == 'fc_ae':
        model = FCAutoEncoder()
    else:
        print('ERROR: specified arch is not suppported')
        exit()

    if not args.pretrained:
        best_acc = 0.0
    else:
        pretrained_model = torch.load(args.pretrained)
        best_acc = pretrained_model['acc']
        model.load_state_dict(pretrained_model['state_dict'])

    if args.cuda:
        model.cuda()
    
    print(model)
    # param_dict = dict(model.named_parameters())
    # params = []
    
    base_lr = 0.1
    
    # for key, value in param_dict.items():
    #     params += [{'params':[value], 'lr': args.lr,
    #         'weight_decay': args.weight_decay,
    #         'key':key}]
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
            weight_decay=args.weight_decay)

    criterion = nn.L1Loss()

    # define the binarization operator
    bin_op = util.BinOp(model)

    if args.evaluate:
        test(evaluate=True)
        exit()

    for epoch in range(1, args.epochs + 1):
        # adjust_learning_rate(optimizer, epoch)
        train(epoch)
        test()
