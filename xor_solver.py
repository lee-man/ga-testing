'''
Author: LI Min
'''
import numpy as np 
from functools import reduce


class XORSolver(object):

    def __init__(self, A, b):
        self.A = A
        self.b = b
        self.x = None
        self.status = None

    def find_free_variables(self):
        free_idx = None
        pivot_idx = None
        (m, n) = np.shape(self.A)
        for i in range(m):
            for j in range(i, n):
                if self.A[i, j] != True:
                    continue
                else:
                    if pivot_idx == None:
                        pivot_idx = [j]
                    else:
                        pivot_idx.append(j)
                    if (len(pivot_idx) >= 2) and (pivot_idx[-1] - pivot_idx[-2]) != 1:
                        if free_idx == None:
                            free_idx = list(range(pivot_idx[-2] + 1, pivot_idx[-1]))
                        else:
                            free_idx.append(range(pivot_idx[-1] + 1, pivot_idx[-1]))
                    break
        if m < n:
            if free_idx == None:
                free_idx = list(range(pivot_idx[-1] + 1, n))
            else:
                free_idx.append(range(pivot_idx[-1] + 1, n))
        return pivot_idx, free_idx
    
    def random_free_varibles(self, free_idx):
        x_free = np.random.choice(2, size=len(free_idx))
        self.x[free_idx] = x_free
        

    def gaussian_elimination(self):
        (m, n) = np.shape(self.A)
        if m > n:
            print('A is a tall matrix (m > n).')
        elif m == n:
            print('A is a square matrix (m = n).')
        else:
            print('A is a fat matrix (m < n).')

        # Swap rows to get echelon form of matrix
        for i in range (m-1):
            find_pivot = False
            for p in range(i, n):
                if self.A[i, p] == False:
                    for j in range(i+1, m):
                        if self.A[j, p] == True:
                            find_pivot = True
                            self.A[[i,j]] = self.A[[j,i]]
                            self.b[[i,j]] = self.b[[j,i]]
                            break
                else:
                    find_pivot = True
                if find_pivot:
                    break    
            for k in range (i+1, m):
                if self.A[k, p] == False: continue
                self.A[k, :] = np.logical_xor(self.A[k, :], self.A[i, :]) 
                self.b[k] = np.logical_xor(self.b[k], self.b[i])


        # Check Solvability
        print('#### Check Solvability ####')
        for i in range(m-1, -1, -1):
            if (self.b[i] == True) and (np.sum(self.A[i, :]) == 0):
                self.status = False # No solution
                print('No solution...')
                break
        if self.status == None:
            self.status = True # One or infinite solutions
            print('One or inifite solutions...')
        
        # print('#### Row reduced A #### \n', self.A)

        # Find free variables and Assign random values for free variables
        if self.status:
            self.x = np.zeros(n).astype(dtype=bool)
            pivot_idx, free_idx = self.find_free_variables()

            print('Pivot variables: ', pivot_idx)
            print('Free variables: ', free_idx)
            if free_idx == None:
                print('A is full rank.')
            else:
                self.random_free_varibles(free_idx)
            print(self.x)
            # exit()

            # Backsubsition
            print('#### Start backsubstition####')
            # Find the first effective equation
            for i in range(m-1, -1, -1):
                if np.sum(self.A[i, :]) != 0:
                    m_max = i
                    break
            # if np.sum(A[m_max, :]) == 1:
            #     self.x[m_max] = self.b[m_max]
            # else:
            for i in range(m_max, -1, -1):
                left_most_idx = np.where(self.A[i, :] == True)[0][0]
                if np.sum(self.A[i, :]) == 1:
                    self.x[left_most_idx] = self.b[m_max]
                else:
                    xor_idx = self.A[i, left_most_idx+1:]
                    self.x[left_most_idx] = reduce(np.logical_xor, np.concatenate((self.x[left_most_idx+1:][xor_idx], [self.b[i]])))

            # return self.A, self.b, self.x


def main():
    # Square matrix
    # A = np.array([
    #     [1, 1, 1, 0],
    #     [1, 1, 0, 1],
    #     [1, 0, 1, 1],
    #     [0, 1, 1, 1]]).astype(dtype=bool)

    # Tall matrix
    A = np.array([
        [1, 1, 1, 0],
        [1, 1, 0, 1],
        [1, 0, 1, 1],
        [0, 1, 1, 1],
        [1, 1, 1, 1]]).astype(dtype=bool)

    # Fat matrix
    # A = np.array([
    #     [1, 1, 1, 0, 1],
    #     [1, 1, 0, 1, 1],
    #     [1, 0, 1, 1, 1],
    #     [0, 1, 1, 1, 1]]).astype(dtype=bool)
    # A = np.array([
    #     [1, 1, 1, 0, 1],
    #     [1, 1, 1, 0, 1],
    #     [1, 1, 1, 0, 1],
    #     [1, 1, 1, 0, 1]]).astype(dtype=bool)

    # b = np.array([1, 1, 0, 1]).astype(dtype=bool)
    b = np.array([1, 1, 0, 1, 0]).astype(dtype=bool)
    equation = XORSolver(A, b)
    equation.gaussian_elimination()

    print('Status: ', equation.status)
    print('Reduced A: ', equation.A)
    print('Reduced b: ', equation.b)
    print('solved x: ', equation.x)

if __name__ == '__main__':
    main()
