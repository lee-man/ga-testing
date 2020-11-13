'''
Check the correctness of torch/numpy conversion.
'''

import numpy as np
import torch
from models.ae import FCAutoEncoder, MLPClassifer
import util
import pickle

# Some model parameters and experiment parameters
num_sc = 415
num_ctrl = 45

# Load the model
model = FCAutoEncoder(num_sc, num_ctrl)
ckpt = torch.load('checkpoint/ckpt.pth')
model.load_state_dict(ckpt['net'])


bin_op = util.BinOp(model)

print(model)
model.eval()
bin_op.binarization()

state_dict_np = {}  # The dictionary to store the weights of BNNs in numpy array format.
for k, v in model.state_dict().items():
    print(k)
    state_dict_np[k] = v.data.numpy()

# Store the weight dict
print('Store the weight...')
with open('checkpoint/weight_np.pkl', 'wb') as f:
    pickle.dump(state_dict_np, f)

# Do a forward in torch
fake_input = torch.rand(1, 415).sign()
output_torch = model(fake_input)
# print('Output from torch model', output_torch.sign())

fake_input_np = fake_input.sign().numpy()
# Do a forward in numpy
# encoder[0] linear
x = np.matmul(fake_input_np, state_dict_np['encoder.0.linear.weight'].T) + state_dict_np['encoder.0.linear.bias']
# encoder[1] bn
x = (x - state_dict_np['encoder.1.bn.running_mean']) /  np.sqrt(state_dict_np['encoder.1.bn.running_var'] + 1e-5) * state_dict_np['encoder.1.bn.weight'] + state_dict_np['encoder.1.bn.bias']
x = np.sign(x)
# encoder[1] linear
x = np.matmul(x, state_dict_np['encoder.1.linear.weight'].T) + state_dict_np['encoder.1.linear.bias']

x = np.sign(x)
# decoder[0] linear
x = np.matmul(x, state_dict_np['decoder.0.linear.weight'].T) + state_dict_np['decoder.0.linear.bias']
# decoder[1] bn
x = (x - state_dict_np['decoder.1.bn.running_mean']) /  np.sqrt(state_dict_np['decoder.1.bn.running_var'] + 1e-5) * state_dict_np['decoder.1.bn.weight'] + state_dict_np['decoder.1.bn.bias']
x = np.sign(x)
# decoder[1] linear
x = np.matmul(x, state_dict_np['decoder.1.linear.weight'].T) + state_dict_np['decoder.1.linear.bias']

# check the correctness
print(np.sum(np.abs(output_torch.data.numpy() - x))/np.prod(x.shape))
# This is the output of BNN
x = np.sign(x)
