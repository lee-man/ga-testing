'''
Check the correctness of torch/numpy conversion.
'''

import numpy as np
import torch
from models.ae import FCAutoEncoder, FCAutoEncoder1Layer, MLPClassifer
import util
import pickle

# Some model parameters and experiment parameters
num_sc = 415
num_ctrl = 100

# Load the model
# model = FCAutoEncoder(num_sc, num_ctrl)
model = FCAutoEncoder1Layer(num_sc, num_ctrl)
ckpt = torch.load('checkpoint/ckpt.pth')
model.load_state_dict(ckpt['net'])


bin_op = util.BinOp(model)

print(model)
model.eval()
bin_op.binarization()

state_dict_np = {}  # The dictionary to store the weights of BNNs in numpy array format.
for k, v in model.state_dict().items():
    # print(k)
    state_dict_np[k] = v.data.numpy()
# Store the weight dict
print('Store the weight...')
'''
with open('checkpoint/weight_np.pkl', 'wb') as f:
    pickle.dump(state_dict_np, f)
# Two-layers
np.save('checkpoint/encoder.0.linear.weight.npy', state_dict_np['encoder.0.linear.weight'])
np.save('checkpoint/encoder.0.linear.bias.npy', state_dict_np['encoder.0.linear.bias'])
np.save('checkpoint/encoder.1.bn.running_mean.npy', state_dict_np['encoder.1.bn.running_mean'])
np.save('checkpoint/encoder.1.bn.running_var.npy', state_dict_np['encoder.1.bn.running_var'])
np.save('checkpoint/encoder.1.bn.weight.npy', state_dict_np['encoder.1.bn.weight'])
np.save('checkpoint/encoder.1.bn.bias.npy', state_dict_np['encoder.1.bn.bias'])
np.save('checkpoint/encoder.1.linear.weight.npy',state_dict_np['encoder.1.linear.weight'])
np.save('checkpoint/encoder.1.linear.bias.npy', state_dict_np['encoder.1.linear.bias'])
np.save('checkpoint/decoder.0.linear.weight.npy', state_dict_np['decoder.0.linear.weight'])
np.save('checkpoint/decoder.0.linear.bias.npy', state_dict_np['decoder.0.linear.bias'])
np.save('checkpoint/decoder.1.bn.running_mean.npy', state_dict_np['decoder.1.bn.running_mean'])
np.save('checkpoint/decoder.1.bn.running_var.npy', state_dict_np['decoder.1.bn.running_var'])
np.save('checkpoint/decoder.1.bn.weight.npy', state_dict_np['decoder.1.bn.weight'])
np.save('checkpoint/decoder.1.bn.bias.npy', state_dict_np['decoder.1.bn.bias'])
np.save('checkpoint/decoder.1.linear.weight.npy', state_dict_np['decoder.1.linear.weight'])
np.save('checkpoint/decoder.1.linear.bias.npy', state_dict_np['decoder.1.linear.bias'])
'''
# One-layer
encoder_sign = np.sign(state_dict_np['encoder.0.bn.weight'])
decoder_sign = np.sign(state_dict_np['decoder.0.bn.weight'])
thred_encoder = -state_dict_np['encoder.0.bn.bias']/state_dict_np['encoder.0.bn.weight'] * np.sqrt(state_dict_np['encoder.0.bn.running_var'] + 1e-5) + state_dict_np['encoder.0.bn.running_mean']
thred_encoder = np.floor(thred_encoder) * encoder_sign
thred_decoder = -state_dict_np['decoder.0.bn.bias']/state_dict_np['decoder.0.bn.weight'] * np.sqrt(state_dict_np['decoder.0.bn.running_var'] + 1e-5) + state_dict_np['decoder.0.bn.running_mean']
thred_decoder = np.floor(thred_decoder) * decoder_sign
state_dict_np['encoder.0.linear.weight'] = (state_dict_np['encoder.0.linear.weight'] + 1.0) / 2.0 * np.expand_dims(encoder_sign, axis=1)
state_dict_np['decoder.0.linear.weight'] = (state_dict_np['decoder.0.linear.weight'] + 1.0) / 2.0 * np.expand_dims(decoder_sign, axis=1)
np.save('checkpoint/encoder.linear.weight.npy', state_dict_np['encoder.0.linear.weight'])
np.save('checkpoint/encoder.bn.thred.npy', thred_encoder)
np.save('checkpoint/decoder.linear.weight.npy', state_dict_np['decoder.0.linear.weight'])
np.save('checkpoint/encoder.bn.thred.npy', thred_decoder)

# exit()
# for i in state_dict_np:
#     print(i)
# exit()
# Do a forward in torch
# fake_input = torch.ones(1, 415)
fake_input = torch.randn(1, 415).sign()
output_torch = model(fake_input)
# print('Output from torch model', output_torch.sign())

fake_input_np = (fake_input.numpy() + 1.0) / 2.0  # fake_input_np in {0, 1}
# Do a forward in numpy
# encoder[0] linear 
# input (1, 415), (215, 415), (215, )
'''
# Two-layer
x = np.matmul(fake_input_np, state_dict_np['encoder.0.linear.weight'].T) + state_dict_np['encoder.0.linear.bias']
# encoder[1] bn
x = (x - state_dict_np['encoder.1.bn.running_mean']) /  np.sqrt(state_dict_np['encoder.1.bn.running_var'] + 1e-5) * state_dict_np['encoder.1.bn.weight'] + state_dict_np['encoder.1.bn.bias']
# => {-1, 1}
x = np.sign(x)
# encoder[1] linear
x = np.matmul(x, state_dict_np['encoder.1.linear.weight'].T) + state_dict_np['encoder.1.linear.bias']

# get the encoding bits in (45,)
x = np.sign(x)

# decoder[0] linear
x = np.matmul(x, state_dict_np['decoder.0.linear.weight'].T) + state_dict_np['decoder.0.linear.bias']
# decoder[1] bn
x = (x - state_dict_np['decoder.1.bn.running_mean']) /  np.sqrt(state_dict_np['decoder.1.bn.running_var'] + 1e-5) * state_dict_np['decoder.1.bn.weight'] + state_dict_np['decoder.1.bn.bias']
x = np.sign(x)
# decoder[1] linear
x = np.matmul(x, state_dict_np['decoder.1.linear.weight'].T) + state_dict_np['decoder.1.linear.bias']

# check the correctness
# print(np.sum(np.abs(output_torch.data.numpy() - x))/np.prod(x.shape))
# This is the output of BNN
x = np.sign(x)
'''
######
# One-layer
print('Check the correctness')
# decoder linear
x = np.matmul(fake_input_np, state_dict_np['encoder.0.linear.weight'].T)
# encoder[1] bn
# caculate the threshold
# thred_encoder = -state_dict_np['encoder.0.bn.bias']/state_dict_np['encoder.0.bn.weight'] * np.sqrt(state_dict_np['encoder.0.bn.running_var'] + 1e-5) + state_dict_np['encoder.0.bn.running_mean']
x = (x >= thred_encoder).astype(float)
# x = (((x * 2.0 - 1.0)* np.sign(state_dict_np['encoder.0.bn.weight'])) + 1.0) / 2.0

# decoder linear
x = np.matmul(x, state_dict_np['decoder.0.linear.weight'].T)
# decoder bn
# caculate the threshold
# thred_decoder = -state_dict_np['decoder.0.bn.bias']/state_dict_np['decoder.0.bn.weight'] * np.sqrt(state_dict_np['decoder.0.bn.running_var'] + 1e-5) + state_dict_np['decoder.0.bn.running_mean']
# print(thred_decoder)
x = (x >= thred_decoder).astype(float)
# x = (((x * 2.0 - 1.0)* np.sign(state_dict_np['decoder.0.bn.weight'])) + 1.0) / 2.0
# print(output_torch.data.sign())
# check the correctness
print('Error', np.sum(np.abs((output_torch.data.sign().numpy()+1.0)/2.0 - x))/np.prod(x.shape))
# This is the output of BNN