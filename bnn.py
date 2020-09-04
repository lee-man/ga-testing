import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import csv
from models.fc_ae import FCAutoEncoder
import util
from torchvision import datasets, transforms
import logging
logging.basicConfig(level=logging.INFO)



def create_mlb(num_id=338):
    ###########################
    # 1. get the unique id list 
    num_sc = 0
    num_exp = 0
    logging.info('#' * 15)
    logging.info('Preproces the `patterns.csv` files.')
    id_list = []
    with open('data/cube.csv', encoding='utf-8') as f:
        f_csv = csv.reader(f)
        
        for row in f_csv:
            if len(row) == 0:
                continue
            num_exp += 1
            
    logging.info('The size of testing data is {}'.format(num_exp))

    ###########################
    # 2. Multi-Label Binarizer
    # Create the Multi-Label Matrix
    num_id = 338
    logging.info('Create Multi-Label Binarizer')
    mlb = -1 * np.ones((num_exp, num_id))
    id_list = list(range(1, num_id + 1))
    with open('data/cube.csv', encoding='utf-8') as f:
        f_csv = csv.reader(f)
        id = 0
        for row in f_csv:
            if len(row) == 0:
                continue
            for element in row:
                idx = int(element) - 1
                mlb[id, idx] = 1
                num_sc += 1
            id += 1
    logging.info('The # and percentage of activated scan chains are {:.2f} and {:.2f}%.'.format(num_sc / num_exp, \
            100. * num_sc / (num_exp * num_id)))

    np.save('data/mlb.npy', mlb)

# create_mlb()

class BNNAutoEncoder(object):
    '''
    The class for BNN AutoEncoder.
    The basic idea is to use BNN as an approach to search the matrix A in EDT testing structure (might imposes stacked XOR network with some AND or OR ops).
    ``````````````
    Training phase: recover all 1's in x, and minimize # 1's (or # 1's approached the constrant definend by users, e.g. 50%);
    Deployment phase:
        Encode: either analogy to the EDT solver to get the encoding bits, or directly use BNN encoder to get the encoding bits;
        Encode efficiency: get x^ from the BNN deocder, compare it with x. If x^ have all 1's of x, we successfully encode x.
    ``````````````
    Remaining problems:
    The operataions should be totally bit-wise, without floating operation.
    '''
    def __init__(self, mlb_path='data/mlb.npy', num_ctrl=37, num_sc=338, arch='fc_ae', epoches=60, batch_size=128, lr=0.0001, wd=1e-5, seed=0):
        self.mlb = np.load(mlb_path)
        self.num_ctrl = num_ctrl
        self.num_sc = num_sc
        self.epoches = epoches
        self.seed = seed
        self._get_device()
        self._set_random_seed()
        
        # Traininig dataset and its loader
        self.train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(self.mlb).float())
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True)

        # Define models
        if arch == 'fc_ae':
            self.model = FCAutoEncoder(num_sc, num_ctrl)
            self.bin_op = util.BinOp(self.model)
        else:
            raise NotImplementedError

        # Define optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)

        # Define loss function
        self.criterion = nn.L1Loss() 

    

    def _get_device(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def _set_random_seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device == 'cuda':
            torch.cuda.manual_seed(self.seed)
    
    def correct_calculate(self, inputs, outputs):
        mask = inputs.eq(1)
        count_inputs = (inputs * mask).sum(dim=1)
        count_outputs = (outputs.sign() * mask).sum(dim=1)
        correct = count_inputs.eq(count_outputs).sum().item()
    
        return correct
    
    def onepercent_calculate(self, outputs):
        count = (outputs.sign().eq(1)).sum().item()

        return count/outputs.size(1)
      

    def train(self):
        best_acc = 0
        logging.info('Start Training...')
        for epoch in range(self.epoches):
            logging.info('\nEpoch {}:'.format(epoch))
            self.model.train()
            correct = 0
            total = 0
            onepercent = 0
            for batch_idx, (inputs, ) in enumerate(self.train_loader):
                inputs = inputs.to(self.device)
                self.optimizer.zero_grad()

                # process the weights including binarization
                self.bin_op.binarization()

                outputs = self.model(inputs)
                mask = inputs.eq(1).float()
                loss = self.criterion(outputs*mask, inputs*mask)
                mask = inputs.eq(-1).float()
                loss += 0.25 * self.criterion(outputs*mask, inputs*mask)
                loss.backward()

                # restore weights
                self.bin_op.restore()
                self.bin_op.updateBinaryGradWeight()

                self.optimizer.step()

                total += inputs.size(0)
                correct += self.correct_calculate(inputs, outputs)
                onepercent += self.onepercent_calculate(outputs)

                util.progress_bar(batch_idx, len(self.train_loader), 'Loss: {:.6f} | Acc: {:.3f} | OneP: {:.3f}'\
                    .format(loss, 100.*correct/total, 100.*onepercent/total))

            acc = 100. * correct / total
            if acc > best_acc:
                best_acc = acc
                logging.info('Saving...')
                state = {
                    'net': self.model.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, './checkpoint/ckpt.pth')
            logging.info('Best accuracy: {}'.format(best_acc))                   

    
    # def _valid(self):
    #     test_loss = 0
    #     correct = 0
    #     onepercent = 0
    #     model.eval()
    #     bin_op.binarization()
    #     with torch.no_grad():
    #         for (inputs, ) in self.train_loader:
    #             inputs = inputs.to(self.device)
    #             outputs = model(inputs)
    #             test_loss += criterion(inputs, outputs).item()
    #             mask = inputs.eq(1).float()
    #             test_loss += criterion(outputs*mask, inputs*mask)
    #             mask = inputs.eq(-1).float()
    #             test_loss += 0.25 * criterion(outputs*mask, inputs*mask)
    #             correct += correct_calculate(inputs, outputs)
    #             onepercent += onepercent_calculate(outputs)

    #     print(outputs)
    #     bin_op.restore()
        
    #     acc = 100. * float(correct) / len(test_loader.dataset)
    #     return acc


if __name__=='__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch XNOR Testing Compression')

    parser.add_argument('--data_path', type=str, default='data/mlb.npy', help='The path of data')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
            help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=60, metavar='N',
            help='number of epochs to train (default: 60)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
            help='learning rate (default: 0.01)')
    parser.add_argument('--wd', default=1e-5, type=float,
            metavar='W', help='weight decay (default: 1e-5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
            help='random seed (default: 1)')
    parser.add_argument('--arch', action='store', default='fc_ae',
            help='the autoencoder structure: FCAutoEncoder')
    args = parser.parse_args()

    logging.info(args)
    
    bnn = BNNAutoEncoder(mlb_path=args.data_path)
    bnn.train()

    
