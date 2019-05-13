import numpy as np 
import matplotlib.pyplot as plt 
import random

def give_piecewise(start=0, end=15, x1=5, x2=10, slope=1, bias=0, num_samples=1000):
    x_1 = np.linspace(start, x1, num_samples)
    y_1 = [bias] * num_samples 
    x_2 = np.linspace(x1, x2, num_samples)
    y_2 = [i - x1 for i in x_2]
    slope = [slope] * num_samples
    bias = [bias] * num_samples
    y_2 = np.add(np.multiply(y_2, slope), bias)
    x_3 = np.linspace(x2, end, num_samples)
    y_3 = [y_2[-1]] * num_samples 
    plt.scatter(x_1, y_1)
    plt.scatter(x_2, y_2)
    plt.scatter(x_3, y_3)
    plt.show()


if __name__ == '__main__':
    give_piecewise()
    
