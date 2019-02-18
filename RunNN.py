from nn_utils import *
import numpy as np

# a = np.array([[0.8,0.1],[0.2,0.7],[0.2,0.1]])
# b = np.array([[1,0],[0,1],[0,0]])
#
# a1 = np.array([[0.9,0.05],[0.1,0.8],[0.2,0.1]])
# b1 = np.array([[1,0],[0,1],[0,0]])
#
# cost = cost_func_cross_entropy(a,b,2)
# cost1 = cost_func_cross_entropy(a1,b1,2)
# print(cost)
# print(cost1)

m = 10
cla = 10
layer = [5,20,15,cla]
learning_rate = 0.01
layer_num = len(layer)
x = np.random.rand(5,m)
y = np.random.randint(0,cla,m)
y = one_hot(y,cla,m)
weights = init_weights(layer,layer_num)

for i in range(500):
    output,catches = forwardprop(x,weights,layer_num)
    cost = cost_func_cross_entropy(output,y,m)
    print(cost)
    weights_derivatives = backprop(output,y,weights,catches,layer_num)
    weights = update_weights(weights,weights_derivatives,learning_rate,layer_num)
