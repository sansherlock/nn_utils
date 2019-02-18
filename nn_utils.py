import numpy as np

def relu(z_data):
    return np.maximum(z_data,0)

def relu_derivative(data):
    return 1.0*(data > 0)

def sigmoid(z_data):
    return 1/(1.0+np.exp(-z_data))

def sigmoid_derivative(data):
    return sigmoid(data) * (1 - sigmoid(data))

def tanh(z_data):
    return (np.exp(z_data) - np.exp(-1 * z_data)) / \
             (np.exp(z_data) + np.exp(-1 * z_data))

def softmax(z_data):
    sub_sum = np.sum(np.exp(z_data),axis = 0)
    a_data = np.exp(z_data) / sub_sum
    return a_data

def fc_layer(input,w,b):
    return np.dot(w,input) + b

def forwardprop(x,weights,layer_num):
    catches = {}
    catches['A0'] = x
    for i in range(layer_num-2):
        catches['Z'+str(i+1)] = fc_layer(catches['A'+str(i)],weights['w'+str(i+1)],weights['b'+str(i+1)])
        catches['A'+str(i+1)] = relu(catches['Z'+str(i+1)])

    catches['Z'+str(layer_num-1)] = fc_layer(catches['Z'+str(layer_num-2)],weights['w'+str(layer_num-1)],weights['b'+str(layer_num-1)])
    catches['A'+str(layer_num-1)] = softmax(catches['Z'+str(layer_num-1)])
    return catches['A'+str(layer_num-1)],catches

def cost_func_cross_entropy(y,label,m):
    return np.sum(-1/m * np.nan_to_num(label * np.log(y) + (1 - label) * np.log(1 - y)))

def backprop(output,y,weights,catches,layer_num):
    weights_derivatives = {}

    delta = output - y
    weights_derivatives['dw'+str(layer_num-1)] = np.dot(delta,catches['A'+str(layer_num-2)].transpose())
    weights_derivatives['db'+str(layer_num-1)] = delta
    for l in range(layer_num-2):
        Z = catches['Z'+ str((layer_num-2)-l)]
        sp = relu_derivative(Z)
        delta = np.dot(weights['w'+str((layer_num-2)-l+1)].transpose(), delta) * sp
        weights_derivatives['db'+str((layer_num-2)-l)] = delta
        weights_derivatives['dw'+str((layer_num-2)-l)] = np.dot(delta, catches['A'+str((layer_num-2)-l-1)].transpose())
    return weights_derivatives
    
def update_weights(weights,weights_derivatives,learning_rate,layer_num):
    for i in range(layer_num-1):
        weights['w'+str(i+1)] = weights['w'+str(i+1)] - learning_rate * weights_derivatives['dw'+str(i+1)]
        weights['b'+str(i+1)] = weights['b'+str(i+1)] - learning_rate * weights_derivatives['db'+str(i+1)]
    return weights

def init_weights(layer,layer_num):
    weights = {}
    for i in range(layer_num-1):
        weights['w'+str(i+1)] = np.random.rand(layer[i+1],layer[i])
        weights['b'+str(i+1)] = np.random.rand(layer[i+1],1)
    return weights

def one_hot(label,cla,m):
    one_hot_label = np.zeros((cla,m))
    # 方式1
    for i in range(m):
        one_hot_label[label[i],i] = 1
    return one_hot_label
    # 方式2
    one_hot_label = np.eye(cla)[label].T
    return one_hot_label
