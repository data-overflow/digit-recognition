"""
#TYPE THE FOLLOWING COMMANDS

>>> ml = NeuralNetwork([784, 200, 80, 10],0.1)
>>> ml.weights_matrices = old_weights
>>> break
"""

import numpy as np
from scipy.special import expit as activation_function
from scipy.stats import truncnorm
import pickle

print('STARTING PROGRAM...')


def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


class NeuralNetwork:
    def __init__(self, network_structure, learning_rate, bias=None):
        self.structure = network_structure
        self.learning_rate = learning_rate
        self.bias = bias
        self.create_weight_matrices()

    def create_weight_matrices(self):       
        bias_node = 1 if self.bias else 0
        self.weights_matrices = []        
        layer_index = 1
        no_of_layers = len(self.structure)
        while layer_index < no_of_layers:
            nodes_in = self.structure[layer_index-1]
            nodes_out = self.structure[layer_index]
            n = (nodes_in + bias_node) * nodes_out
            rad = 1 / np.sqrt(nodes_in)
            X = truncated_normal(mean=2, sd=1, low=-rad, upp=rad)
            wm = X.rvs(n).reshape((nodes_out, nodes_in + bias_node))
            self.weights_matrices.append(wm)
            layer_index += 1
        
    def train(self, input_vector, target_vector):        
        no_of_layers = len(self.structure)
        input_vector = np.array(input_vector, ndmin=2).T
        layer_index = 0        
        res_vectors = [input_vector]
        while layer_index < no_of_layers - 1:
            in_vector = res_vectors[-1]
            if self.bias:                
                in_vector = np.concatenate((in_vector, [[self.bias]]))
                res_vectors[-1] = in_vector
            x = np.dot(self.weights_matrices[layer_index], in_vector)
            out_vector = activation_function(x)            
            res_vectors.append(out_vector)    
            layer_index += 1
        
        layer_index = no_of_layers - 1
        target_vector = np.array(target_vector, ndmin=2).T         
        output_errors = target_vector - out_vector
        
        while layer_index > 0:
            out_vector = res_vectors[layer_index]
            in_vector = res_vectors[layer_index-1]
            if self.bias and not layer_index==(no_of_layers-1):
                out_vector = out_vector[:-1,:].copy()
            tmp = output_errors * out_vector * (1.0 - out_vector)     
            tmp = np.dot(tmp, in_vector.T)
            self.weights_matrices[layer_index-1] += self.learning_rate * tmp            
            output_errors = np.dot(self.weights_matrices[layer_index-1].T, output_errors)
            if self.bias:
                output_errors = output_errors[:-1,:]
            layer_index -= 1
                            
    def predict(self, input_vector):
        no_of_layers = len(self.structure)
        if self.bias:            
            input_vector = np.concatenate((input_vector, [self.bias]))
        in_vector = np.array(input_vector, ndmin=2).T
        layer_index = 1        
        while layer_index < no_of_layers:
            x = np.dot(self.weights_matrices[layer_index-1], 
                       in_vector)
            out_vector = activation_function(x)            
            in_vector = out_vector
            if self.bias:
                in_vector = np.concatenate((in_vector, [[self.bias]]))
            layer_index += 1

        print('\n===== PREDICTION RESULT =====')
        result = {}
        for i in range(len(out_vector)):            
            result[i] = float(out_vector[i][0])
        for i in result:
            print('probability of',i,':',float(round(result[i],4)))
        print('\nPREDICTED DIGIT : ',max(result, key=result.get))
        
        return out_vector    

print('LOADING DATASET...')
try:
    with open("mnist.pkl", "br") as fh:
        data = pickle.load(fh)
    train_data = data[0]
    train_labels = data[1]
    lr = np.arange(10)
    train_labels_one_hot = (lr==train_labels).astype(np.float)
    train_imgs = np.asfarray(train_data[:,1:])*(0.99/255)+0.01
except FileNotFoundError:
    print("ERROR: Missing mnist.pkl file. Cant train data")

with open("weights.bin", "br") as fh:
    old_weights = pickle.load(fh)
    
print('ALL DATA LOADED SUCCESSFULLY')

print('COMMAND PROMPT STARTED')
while True:
    cmd = input('>>> ')
    if cmd == 'break':
        break
    exec(cmd)

print('STARTING GUI PREDICTION...')
import pygame
from pygame.locals import *

pygame.init()
disp = pygame.display.set_mode((224,224))
pygame.display.set_caption('test')
run = True

draw_on = False
last_pos = (0, 0)
color = (255, 255, 255)
radius = 8

def reduce(mat, fac):
    x = np.array([[0.0 for i in range(len(mat)//fac)] for i in range(len(mat)//fac)])
    i = j = 0	
    for r in range(0,len(mat),fac):
        j = 0
        for c in range(0,len(mat),fac):
            avg = (mat[r:r+fac, c:c+fac].mean() + mat[r:r+fac, c:c+fac].max()) / 2
            x[i, j] = avg
            j += 1
        i += 1
    return x

def roundline(srf, color, start, end, radius=1):
    dx = end[0]-start[0]
    dy = end[1]-start[1]
    distance = max(abs(dx), abs(dy))
    for i in range(distance):
        x = int( start[0]+float(i)/distance*dx)
        y = int( start[1]+float(i)/distance*dy)
        pygame.draw.circle(srf, color, (x, y), radius)

while run:
    for event in pygame.event.get():
        if event.type == QUIT:
            run = False
        if event.type == KEYUP:
            if event.key == K_a:                
                inp=[]
                for x in range(224):
                    for y in range(224):
                        temp = disp.get_at((x,y))[0]/255                    
                        inp += [temp]
                inp = np.array(inp)
                inp.resize(224,224)
                #y = inp[::8,::8]
                y = reduce(inp, 8).T
                inp = y.flatten()
                ml.predict(inp)
                print('PREDICTION COMPLETE')
            if event.key == K_c:
                disp.fill((0, 0, 0))
                
        if event.type == pygame.MOUSEBUTTONDOWN:            
            pygame.draw.circle(disp, color, event.pos, radius)
            draw_on = True
        if event.type == pygame.MOUSEBUTTONUP:
            draw_on = False
        if event.type == pygame.MOUSEMOTION:
            if draw_on:
                pygame.draw.circle(disp, color, event.pos, radius)
                roundline(disp, color, event.pos, last_pos,  radius)
            last_pos = event.pos
    pygame.display.flip()
        
pygame.quit()

"""
>>> ml = NeuralNetwork([784, 200, 80, 10],0.1)
>>> ml.weights_matrices = old_weights
>>> break
"""
