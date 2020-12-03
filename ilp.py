'''
The sovler for learnd BNN structure
'''
import cvxpy as cp
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
# model = FCAutoEncoder1Layer(num_sc, num_ctrl)
# ckpt = torch.load('checkpoint/ckpt.pth')
# model.load_state_dict(ckpt['net'])


# bin_op = util.BinOp(model)

# print(model)
# model.eval()
# bin_op.binarization()

# state_dict_np = {}  # The dictionary to store the weights of BNNs in numpy array format.
# for k, v in model.state_dict().items():
#     # print(k)
#     state_dict_np[k] = v.data.numpy()
# # Store the weight dict
# print('Extract weights...')
# One-layer
# thred_decoder = -state_dict_np['decoder.0.bn.bias']/state_dict_np['decoder.0.bn.weight'] * np.sqrt(state_dict_np['decoder.0.bn.running_var'] + 1e-5) + state_dict_np['decoder.0.bn.running_mean']
# thred_decoder = np.floor(thred_decoder)
# state_dict_np['decoder.0.linear.weight'] = (state_dict_np['decoder.0.linear.weight'] + 1.0) / 2.0
# thred_sign = np.sign(state_dict_np['decoder.0.bn.weight'])

# Solving binary programming
# Notation:
# A: the complete connection matrix; A_hat: the selected connection matrix;
# b: the complete threshold; b_hat: the selected threshold
# x: the encoding bits; y: the merged test cube to be encoded; y_hat: the decoded/re-constructed test cube
y = np.random.choice(2, (num_sc), p=[0.95, 0.05])
x = cp.Variable(num_ctrl, boolean=True)
# A = state_dict_np['decoder.0.linear.weight'] * np.expand_dims(thred_sign, axis=1)
# b = thred_decoder * thred_sign
A = np.load('checkpoint/decoder.linear.weight.npy')
b = np.load('checkpoint/encoder.bn.thred.npy')
A_hat = A[y.astype(bool)]
b_hat = b[y.astype(bool)]
# print(A_hat)
# print(b_hat)
# exit()
# constraints
# cost = cp.norm1(A @ x - b)   # This cost is too computational complex to solve.
cost = cp.norm1(x)
objective = cp.Minimize(cost)
constraint = [A_hat @ x >= b_hat]
prob = cp.Problem(objective, constraint)
prob.solve()

print("Status: ", prob.status)
print("The optimal value is", prob.value)
print("A solution x is")
print(x.value)

y_recover = np.matmul(A, x.value) >= b
# print(y_recover)
print('OnePercent: ', np.sum(y_recover) / num_sc)

