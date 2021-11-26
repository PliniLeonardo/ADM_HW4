import numpy as np      
import matplotlib.pyplot as plt 
import pickle

from tqdm.notebook import tqdm

from AudioSignals import *


'''
library of function useful for minhashing
'''


def generate_hash_parameters(n, number_addings):
    '''
    randomly exctract an hash function from the universal hash family
    of number_addings parameters associated with the prime n
    '''
    
    params = np.random.randint(0, n, number_addings)
    
    return (n, params)


def retrieve_hash_function(parameters):
    '''
    return the hash function associated to the input parameters
    
    parameters : [n, a1, a2, a3, ...]
    '''
    
    n = parameters[0]
    a = parameters[1]
    
    def hash(*x):
        value = np.sum(np.array(x) * a) % n
        return(value)
    
    return(hash)