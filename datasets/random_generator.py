import numpy as np

# Some arguement variables
# length of training dataset
size_training = 10000
# length of testing dataset
# We actually don't really care about the testing dataset
size_testing = 1000

# length of scan chains
len_sc = 1000

# the representation of scan chain stats
state_sc = [1., -1.]

# probability of scan chain states
prob_sc = [0.2, 0.8]



# set the seed for reproduction
np.random.seed(0)

# generate training dataset
train_dataset = np.choice(state_sc, size=(size_training, len_sc), p=prob_sc)
test_dataset = np.choice(state_sc, size=(size_testing, len_sc), p=prob_sc)

np.save('train.npy', train_dataset)
np.save('test.npy', test_dataset)


