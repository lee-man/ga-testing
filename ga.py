'''
Author: LI Min
'''

import numpy as np
from xor_solver import XORSolver
from functools import reduce
import matplotlib.pyplot as plt
plt.switch_backend('agg')



class GAforXOR(object):
    '''
    Fitnesss: Encoding Scucess Rate
    Secondary Metric: Activated Scan Chain Rate
    '''

    def __init__(self, num_sc=415, num_ctrl=40, num_generation=20, num_pop=20, num_parent=10, num_crossover=10, num_mutation=5, mutation_rate=0.1, connection_percentage=0.2, freq_sc_file='data/freq_sc.npy', specified_percentage=0.1, num_test=100):
        self.num_sc = num_sc
        self.num_ctrl = num_ctrl
        self.num_generation = num_generation
        self.num_pop = num_pop
        ### To Do: change the number into percentage
        self.num_parent = num_parent
        self.num_crossover = num_crossover
        self.num_mutation = num_mutation
        self.mutation_rate = mutation_rate
        self.connection_percentage = connection_percentage
        self.freq_sc = np.load(freq_sc_file)
        self.test_data = self.initialize_testdata(specified_percentage, num_test)
        self.pop = self.initialize_pop()
        self.generation_idx = 0
        self.fitness_history = {}
        self.act_history = {}


        

    def initialize_testdata(self, specified_percentage, num_test):
        test_data = np.zeros((num_test, self.num_sc))
        for (i, row) in enumerate(test_data):
            generate_row = np.random.choice(self.num_sc, size=int(self.num_sc *  specified_percentage), replace=False, p=self.freq_sc)
            test_data[i][generate_row] = 1
        test_data = test_data.astype(dtype=bool)
        return test_data

    def initialize_pop(self):
        pop = np.random.choice(2, size=(self.num_pop, self.num_sc, self.num_ctrl), p=[1-self.connection_percentage, self.connection_percentage]).astype(dtype=bool)
        return pop

    def cal_pop_fitness(self):
        fitness_pop = []
        act_pop = []
        for (i, A) in enumerate(self.pop):
            fitness_i, act_i = self.xor_solving(A)
            fitness_pop.append(fitness_i)
            act_pop.append(act_i)
            # print('A index:', i)
        self.fitness_history[self.generation_idx] = fitness_pop
        self.act_history[self.generation_idx] = act_pop
        # self.generation_idx += 1

    def xor_solving(self, A):
        # A being boolean matrix
        encoded_count = 0.0
        total = 0.0
        activated_rate_acculmulate = 0.0
        for (i, cube) in enumerate(self.test_data):
            total += 1
            A_hat = A[cube.astype(dtype=bool)]
            b_hat = cube[cube.astype(dtype=bool)].astype(dtype=bool)
            equation = XORSolver(A_hat, b_hat)
            equation.gaussian_elimination()
            if equation.status:
                encoded_count += 1
                activated_rate_acculmulate += self.calculate_activated_rate(A, equation.x)
        return encoded_count/(total + 1.0), activated_rate_acculmulate/(encoded_count + 1.0)

    def calculate_activated_rate(self, A, x):
        b = np.zeros(np.shape(A)[0]).astype(dtype=bool)
        for (i, A_i) in enumerate(A):
            if np.sum(A_i) == 0:
                continue
            x_valid = x[A_i]
            b[i] = reduce(np.logical_xor, x_valid)
        return np.sum(b) / len(b)
    

    def select_mating_pool(self):
        parents = np.empty((self.num_parent, self.num_sc, self.num_ctrl))
        ind = np.argpartition(self.fitness_history[self.generation_idx], -self.num_parent)[-self.num_parent:]
        parents = self.pop[ind]
        return parents

    def crossover(self, parents):
        # There are different types of crossover. For XORNet, more flexible solution is to crossover under the granularity of each row of A.
        # kind of like Uniform Crossover
        offspring = np.empty((self.num_crossover, self.num_sc, self.num_ctrl))
        # The source of each row of offspring coming from
        source_crossover = np.random.choice(2, size=(self.num_crossover, self.num_sc)).astype(dtype=bool)
        for k in range(self.num_crossover):
            parents_idx = np.random.choice(self.num_parent, size=2, replace=False)
            offspring[k][source_crossover[k]] = parents[parents_idx[0]][source_crossover[k]]
            offspring[k][np.invert(source_crossover[k])] = parents[parents_idx[1]][np.invert(source_crossover[k])]
        return offspring.astype(dtype=bool)

    def mutation(self, offspring):
        # Mutation. For XORNet, mutation happends elemently in A.
        selected_idx = np.random.choice(self.num_crossover, size=self.num_mutation, replace=False)
        # Mutation points
        mu_idx = np.random.choice(2, size=(self.num_mutation, self.num_sc, self.num_ctrl), p=[1-self.mutation_rate, self.mutation_rate]).astype(dtype=bool)
        offspring[selected_idx][mu_idx] = np.invert(offspring[selected_idx][mu_idx])

        return offspring.astype(dtype=bool)


    
    def GALoop(self):
        new_pop = np.empty((self.num_pop, self.num_sc, self.num_ctrl))
        self.cal_pop_fitness()
        parents = self.select_mating_pool()
        offspring = self.crossover(parents)
        offspring = self.mutation(offspring)
        self.pop[:self.num_parent] = parents
        self.pop[-self.num_crossover:] = offspring
        self.generation_idx += 1
    
    def GA(self):
        for i in range(self.num_generation):
            print('###### No. {} generation ####'.format(i))
            self.GALoop()
            print('max ', i, ' :',  np.max(self.fitness_history[i]))
            print('average: ', i, ':', np.average(self.fitness_history[i]))
    
    def visulization(self):
        fig, ax = plt.subplots(1, 1)
        for (key, values) in self.fitness_history.items():
            ax.plot([key] * len(values), values, '.', color='k')
            ax.plot(key, np.max(values), '*', color='r')
            ax.plot(key, np.average(values), 'o', color='b')
        plt.xlabel('# Generation')
        plt.ylabel('Encoding rate')
        plt.title('GA for Testing')
        plt.savefig('figs/GA_fitness.pdf')


        
        





def main():
    ga = GAforXOR()
    ga.GA()
    ga.visulization()

if __name__ == '__main__':
    main()