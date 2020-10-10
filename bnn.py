import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import sys
import csv
from models.ae import FCAutoEncoder
import util
from torchvision import datasets, transforms
import logging
logging.basicConfig(level=logging.INFO)

import matplotlib.pyplot as plt
plt.switch_backend('agg')

def create_mlb(num_id=415, num_cell=330):
    ###########################
    # 1. get the unique id list 
    num_sc = 0
    num_exp = 0
    logging.info('#' * 15)
    logging.info('Preproces the `TestCubes_ten_percent_11.csv` files.')
    id_list = []
    with open('data/TestCubes_ten_percent_11.csv', encoding='utf-8') as f:
        f_csv = csv.reader(f)
        row_count = 0 # One test cube's information is listed in 4 rows
        for row in f_csv:
            if row_count == 0:
                pass
                # assert int(row[0]) == num_exp, 'The test cube ID is not consistent'
            if row_count == 1:
                if len(row) != 0:
                    num_exp += 1
            
            row_count = (row_count + 1) % 4


    logging.info('The size of testing data is {}'.format(num_exp))

    ###########################
    # 2. Multi-Label Binarizer
    # Create the Multi-Label Matrix
    logging.info('Create Multi-Label Binarizer with Cells')
    mlb = np.zeros((num_exp, num_id, num_cell))
    with open('data/TestCubes_ten_percent_11.csv', encoding='utf-8') as f:
        f_csv = csv.reader(f)
        id = 0
        row_count = 0
        for row in f_csv:
            if row_count == 0:
                # assert int(row[0]) == id, 'The test cube ID is not consistent'
                pass
            elif row_count == 1:
                test_cube_id = np.array(list(map(int, row[:-1])))
                num_sc += len(row)
            elif row_count == 2:
                test_cbue_cell_id = np.array(list(map(int, row[:-1])))
                if len(row) != 0:
                    mlb[(id, test_cube_id, test_cbue_cell_id)] = 1
            elif row_count == 3:
                cell_value = np.array(list(map(int, row[:-1])))
                cell_value = 2 * cell_value - 1
                if len(row) != 0:
                    mlb[(id, test_cube_id, test_cbue_cell_id)] *= cell_value
                    id += 1
            
            row_count = (row_count + 1) % 4
    

    assert id == num_exp, 'The total number of test cubes does not match'

    logging.info('The # and percentage of activated scan chains are {:.2f} / {} and {:.2f}%.'.format(num_sc / num_exp, num_id, \
            100. * num_sc / (num_exp * num_id)))

    np.save('data/mlb_cell.npy', mlb)

# create_mlb(num_id=415, num_cell=330)


class BNNAutoEncoder(object):
    '''
    The class for BNN AutoEncoder.
    The basic idea is to use BNN as an approach to search the matrix A in EDT testing structure (might imposes stacked XOR network with some AND or OR ops).
    ``````````````
    Data: 25,093 test cubes with some blanks of test cubes.
    On average 7.82/415 (1.88%) scan chains are used.
    ``````````````
    Workflow:
    1. Merge the row data accoding to some constraints on specified scan chain percentage **for multiple times**.
    2. Train and evaluate the BNN on merged data.
    3. Fixed BNN structure and mege the row data to meet the encoding efficacy and low-power constraint in real scenario.
    ``````````````
    Matrics:
    1. Merged test cube count.
    2. Average specified scan chain percentage.

    ``````````````
    Training phase: recover all 1's in x, and minimize # 1's (or # 1's approached the constrant definend by users, e.g. 50%);
    Deployment phase:
        Encode: either analogy to the EDT solver to get the encoding bits, or directly use BNN encoder to get the encoding bits;
        Encode efficiency: get x^ from the BNN deocder, compare it with x. If x^ have all 1's of x, we successfully encode x.
    ``````````````
    Remaining problems:
    The operataions should be totally bit-wise, without floating operation.
    '''
    def __init__(self, mlb_path='data/mlb_cell.npy', num_ctrl=45, num_sc=415, num_merge=5, upper_bound_pre=0.2, upper_bound=0.5, arch='fc_ae', epoches=300, batch_size=16, lr=0.01, wd=1e-5, seed=0):
        self.mlb = np.load(mlb_path)
        self.num_ctrl = num_ctrl
        self.num_sc = num_sc
        self.num_merge = num_merge
        self.upper_bound_pre = upper_bound_pre
        self.upper_bound = upper_bound
        self.epoches = epoches
        self.seed = seed
        self._get_device()
        self._set_random_seed()
        self.writer = SummaryWriter('runs')

        # pre-merge to generate training data
        self.merge_pre()
        exit()
        self.data = np.load('data/data.npy')

        logging.info('The size of dataset is {}'.format(self.data.shape[0]))
        specified_percentage = self.data.sum() / (self.data.shape[0] * self.num_sc)
        logging.info('Specified scan chain percentage after merging is {:.2f}% {:.2f}.'.format(100.*specified_percentage, specified_percentage*self.num_sc))

        
        # Traininig dataset and its loader
        self.data = 2 * self.data - 1
        self.train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(self.data).float())
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

        ckpt = torch.load('checkpoint/ckpt.pth')
        self.model.load_state_dict(ckpt['net'])

    def _get_device(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def _set_random_seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device == 'cuda':
            torch.cuda.manual_seed(self.seed)

    def check_conflict(self, cube1, cube2):
        '''
        Check whether two cubes have a confliction
        '''
        # return (cube1 * cube2).sum() == 0
        # check scan chain level conflict
        cube1_sc = (cube1.sum(axis=1) != 0).astype(float)
        cube2_sc = (cube2.sum(axis=1) != 0).astype(float)
        if (cube1_sc * cube2_sc).sum() == 0:
            return True
        else:
            sc_index = ((cube1_sc * cube2_sc) == 1)
            if (cube1[sc_index] * cube2[sc_index]).sum() == (sc_index.sum()):
                return True
            else:
                return False
        # product = ((cube1 * cube2) == -1).astype(float)
        # return product.sum() == 0


    def merge_two_cube(self, cube1, cube2):
        '''
        Merge two testing cube.
        '''
        cube = np.zeros(cube1.shape)
        # cube = ((cube1 + cube2) > 0).astype(float)
        cube = np.sign(cube1 + cube2)
        return cube

    def calculate_specified_percentage(self, cube):
        # cube_with_cell = (cube.sum(axis=1) > 0).astype(float)
        cube_with_cell = (cube.sum(axis=1) != 0).astype(float)
        return cube_with_cell.sum()/cube.shape[0]


    def merge_pre(self):
        logging.info('*' * 15)
        logging.info('Start Pre-Merging.')
        mlb = copy.deepcopy(self.mlb)

        merged_array = []
        for merge_id in range(self.num_merge):
            np.random.shuffle(mlb)
            mask = np.zeros(mlb.shape[0])
            idx_now = 0
            print('Starting index', idx_now)
            mask[0] = 1
            merged_cube = copy.deepcopy(mlb[idx_now])

            while idx_now < (mlb.shape[0] - 1):
                for id in range(idx_now+1, mlb.shape[0]):
                    row = mlb[id]
                    if id == (mlb.shape[0] - 1):
                        if mask[id] != 1 and self.check_conflict(merged_cube, row):
                            merged_cube_candidate = self.merge_two_cube(merged_cube, row)
                            specified_percentage = self.calculate_specified_percentage(merged_cube_candidate)
                            if specified_percentage <= self.upper_bound_pre:
                                merged_cube = merged_cube_candidate
                                mask[id] = 1
                                # print('Merged Index', id)
                        merged_array.append(merged_cube)
                        while mask[idx_now] == 1 and idx_now < (mlb.shape[0] - 1):
                            idx_now += 1
                        mask[idx_now] = 1
                        print('Starting index', idx_now)
                        merged_cube = copy.deepcopy(mlb[idx_now])
                        # break
                    elif mask[id] == 1:
                        continue
                    elif self.check_conflict(merged_cube, row):
                        merged_cube_candidate = self.merge_two_cube(merged_cube, row)
                        specified_percentage = self.calculate_specified_percentage(merged_cube_candidate)
                        if specified_percentage <= self.upper_bound_pre:
                            merged_cube = merged_cube_candidate
                            mask[id] = 1
                            # print('Merged Index', id)

        merged_array = np.array(merged_array)
        self.data = (merged_array.sum(axis=2) != 0).astype(float)
        # Saving the data
        np.save('data/data_{}.npy'.format(self.num_merge), merged_array)
        logging.info('The size of dataset is {}'.format(self.data.shape[0]))
        specified_percentage = self.data.sum() / (self.data.shape[0] * self.num_sc)
        logging.info('Specified scan chain percentage after merging is {:.2f}%.'.format(100.*specified_percentage))


    
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
        n_iter = 0
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
                loss += 0.01 * self.criterion(outputs*mask, inputs*mask)
                loss.backward()
                
                # restore weights
                self.bin_op.restore()
                self.bin_op.updateBinaryGradWeight()

                self.optimizer.step()

                total += inputs.size(0)
                correct += self.correct_calculate(inputs, outputs)
                onepercent += self.onepercent_calculate(outputs)

                self.writer.add_scalar('loss', loss, n_iter)
                self.writer.add_scalar('ones', self.onepercent_calculate(outputs), n_iter)
                n_iter += 1

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
        logging.info('Saving Final Model...')
        state = {
            'net': self.model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_end.pth')               
    
    def visual(self):
        edt_eff = np.zeros(self.num_sc)
        edt_eff[:self.num_ctrl] = 1
        edt_eff[self.num_ctrl:] = np.power(0.5, range(self.num_sc - self.num_ctrl))

        bnn_correct = np.zeros(self.num_sc)
        bnn_total = np.zeros(self.num_sc)
        loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=1)
        self.model.eval()
        for i, (input, ) in enumerate(loader):
            scs = (input == 1).sum().item()
            bnn_total[scs] += 1
            # forward
            # process the weights including binarization
            self.bin_op.binarization()

            output = self.model(input).sign()
            mask = input.eq(1)
            count_inputs = (input * mask).sum()
            count_outputs = (output * mask).sum()
            correct = count_inputs.eq(count_outputs)
            if correct:
                bnn_correct[scs] += 1
        bnn_eff = bnn_correct / (bnn_total + 0.01)

        plt.figure()
        plt.bar(np.arange(self.num_sc)[37:50], bnn_total[37:50], alpha=0.9, width=0.2, label='Total')
        plt.bar(np.arange(self.num_sc)[37:50] + 0.2, bnn_correct[37:50], alpha=0.9, width=0.2, label='BNN')
        plt.bar(np.arange(self.num_sc)[37:50] + 0.4, bnn_total[37:50] * edt_eff[37:50], alpha=0.9, width=0.2, label='EDT')
        plt.xlabel('# Scan Chains')
        plt.ylabel('Test Cube Density')
        plt.legend()
        plt.savefig('encoding_dist.pdf')
        plt.close()

        # plt.figure()
        # plt.bar(np.arange(self.num_sc)[37:50], edt_eff[37:50], width=0.2, label='EDT')
        # plt.bar(np.arange(self.num_sc)[37:50] + 0.2, bnn_eff[37:50], width=0.2, label='BNN')
        # plt.legend()
        # plt.savefig('encoding_eff.pdf')
        # plt.close()

    def one_forward(self, merged_cube):
        test_input = (merged_cube.sum(axis=1) != 0).astype(float)
        test_input = 2 * test_input - 1
        test_input = torch.from_numpy(test_input).float()
        test_input.reshape((1, -1))
        test_input = test_input.unsqueeze(0)
        # print(test_input.size())
        output = self.model(test_input).sign()
        mask = test_input.eq(1)
        count_inputs = (test_input * mask).sum()
        count_outputs = (output * mask).sum()
        correct = count_inputs.eq(count_outputs)

        if (output.sum().item() / test_input.size(1)) <= self.upper_bound:
            return correct, output.sum().item()
        else:
            return False, None
        
    
    def merge_post(self):
        logging.info('*' * 15)
        logging.info('Start Post-Merging.')
        mlb = copy.deepcopy(self.mlb)
        activated_num = 0
        mask = np.zeros(mlb.shape[0])
        idx_now = 0
        mask[0] = 1
        merged_array = []
        merged_idx = [0]
        merged_idx_failed = []
        merged_cube = copy.deepcopy(mlb[idx_now])

        self.model.eval()
        self.bin_op.binarization()

        while idx_now < (mlb.shape[0] - 1):
            for id in range(idx_now+1, mlb.shape[0]):
                row = mlb[id]
                if id == (mlb.shape[0] - 1):
                    if mask[id] != 1 and self.check_conflict(merged_cube, row):
                        merged_cube_candidate = self.merge_two_cube(merged_cube, row)
                        specified_percentage = self.calculate_specified_percentage(merged_cube_candidate)
                        if specified_percentage <= self.upper_bound_pre:
                            merged_cube = merged_cube_candidate
                            mask[id] = 1
                            merged_idx.append(id)

                    encode_eff, ones = self.one_forward(merged_cube)
                    if encode_eff == True:
                        activated_num += ones
                        merged_array.append(merged_cube)
                    else:
                        merged_idx_failed.extend(merged_idx)
    
                    while mask[idx_now] == 1 and idx_now < (mlb.shape[0] - 1):
                        idx_now += 1
                    mask[idx_now] = 1
                    logging.info('Merging index:{}'.format(idx_now))
                    merged_idx = [idx_now]
                    merged_cube = copy.deepcopy(mlb[idx_now])
                    # break
                elif mask[id] == 1:
                    continue
                elif self.check_conflict(merged_cube, row):
                    merged_cube_candidate = self.merge_two_cube(merged_cube, row)
                    specified_percentage = self.calculate_specified_percentage(merged_cube_candidate)
                    if specified_percentage <= self.upper_bound_pre:
                        merged_cube = merged_cube_candidate
                        mask[id] = 1
                        merged_idx.append(id)

        merged_array = np.array(merged_array)
        self.data = (merged_array.sum(axis=2) > 0).astype(float)
        logging.info('The size of encoded dataset is {}'.format(self.data.shape[0]))
        activated_percentage = activated_num / (self.data.shape[0] * self.num_sc)
        logging.info('Acitvated scan chain percentage after merging is {:.2f}%.'.format(100.*activated_num))
        
        




if __name__=='__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch XNOR Testing Compression')

    parser.add_argument('--data_path', type=str, default='data/mlb_cell.npy', help='The path of data')
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
    # bnn.train()
    # bnn.visual()
    bnn.merge_post()

    
