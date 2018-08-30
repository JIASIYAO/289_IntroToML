import matplotlib.pyplot as plt
import numpy as np

from sklearn.neural_network import MLPRegressor
from starter import *

def calc_layers(num_layers):
    k = num_layers
    l = np.roots([k-1, 9, -10000])
    idx = np.where(l>0)[0]
    l = int(l[idx])
    layers = []
    for i in range(k):
        layers.append(l)
    return layers

def neural_network(X, Y, X_test, Y_test, num_layers, activation):
    """
    This function performs neural network prediction.
    Input:
        X: independent variables in training data.
        Y: dependent variables in training data.
        X_test: independent variables in test data.
        Y_test: dependent variables in test data.
        num_layers: number of layers in neural network
        activation: type of activation, ReLU or tanh
    Output:
        mse: Mean square error on test data.
    """
    ## YOUR CODE HERE
    #################
    if activation == 'ReLU':
        activation = 'relu'
    clf = MLPRegressor(hidden_layer_sizes = calc_layers(num_layers), activation = activation, solver='lbfgs')
    clf.fit(X, Y)

    mses = []
    for i, x_test in enumerate(X_test):
        y_test = Y_test[i]
        y_pred = clf.predict([x_test])
        mse = np.mean(np.sqrt(np.sum((y_pred - y_test)**2, axis=1)))
        mses.append(mse)
 
    return np.mean(mses)


#############################################################################
#######################PLOT PART 2###########################################
#############################################################################
def generate_data(sensor_loc, k=7, d=2, n=1, original_dist=True, noise=1):
    return generate_dataset(
        sensor_loc,
        num_sensors=k,
        spatial_dim=d,
        num_data=n,
        original_dist=original_dist,
        noise=noise)


np.random.seed(0)
n = 200
num_layerss = [1, 2, 3, 4]
mses = np.zeros((len(num_layerss), 2))

# for s in range(replicates):
sensor_loc = generate_sensors()
X, Y = generate_data(sensor_loc, n=n)  # X [n * 2] Y [n * 7]
X_test, Y_test = generate_data(sensor_loc, n=1000)
for t, num_layers in enumerate(num_layerss):
    ### Neural Network:
    mse = neural_network(X, Y, X_test, Y_test, num_layers, "ReLU")
    mses[t, 0] = mse

    mse = neural_network(X, Y, X_test, Y_test, num_layers, "tanh")
    mses[t, 1] = mse

    print('Experiment with {} layers done...'.format(num_layers))

### Plot MSE for each model.
plt.figure()
activation_names = ['ReLU', 'Tanh']
for a in range(2):
    plt.plot(num_layerss, mses[:, a], label=activation_names[a])

plt.title('Error on validation data verses number of neurons')
plt.xlabel('Number of layers')
plt.ylabel('Average Error')
plt.legend(loc='best')
plt.yscale('log')
plt.savefig('num_layers.png')
