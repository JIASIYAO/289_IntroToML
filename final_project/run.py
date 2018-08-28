import numpy as np
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import matplotlib.cm as cm  

params = {'legend.fontsize': '18', 
 'axes.labelsize': '28', 
 'axes.titlesize': '22', 
 'xtick.labelsize': '28', 
 'ytick.labelsize': '28'}                                                       
plt.rcParams.update(params)

def read_data(file_name):
    """
    This function transfer the original txt file into a matrix txt file
    We only call this function once and it creates a matrix file
    We will use that matrix file as the input file after
    """
    with open(file_name) as f:
        content = f.readlines()

    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content] 
    content = np.array(content)

    # get the first H position
    idx1 = np.arange(1, len(content), 4)
    pos1 = content[idx1] 
    xyz1 = np.array([i.split() for i in pos1])
    x1 = [float(i) for i in xyz1.T[0]]
    y1 = [float(i) for i in xyz1.T[1]]
    z1 = [float(i) for i in xyz1.T[2]]
    
    # get the second H position
    idx2 = np.arange(2, len(content), 4)
    pos2 = content[idx2] 
    xyz2 = np.array([i.split() for i in pos2])
    x2 = [float(i) for i in xyz2.T[0]]
    y2 = [float(i) for i in xyz2.T[1]]
    z2 = [float(i) for i in xyz2.T[2]]

    # get the psi - the wave function
    idx3 = np.arange(3, len(content), 4)
    temp = content[idx3]
    psi = np.array([float(i.split()[1]) for i in temp])

    # write to a matrix file
    X = np.matrix([x1, y1, z1, x2, y2, z2, psi])
    np.savetxt('matrix_'+file_name+'.txt', X)

class fit_wfn():
    def __init__(self, data):
        # read the data: which is a n*7 matrix
        self.data = data 

    def fit(self, verbose=False):
        data = self.data

        # split into train and test data
        train = data.T[0:150000]
        test = data.T[150000:]
        x_train = train.T[0:6].T
        y_train = train.T[6].T
        x_test = test.T[0:6].T
        y_test = test.T[6].T
    
        # train the model
        mlp = MLPRegressor()
        mlp.hidden_layer_sizes = self.hidden_layer_sizes
        mlp.activation = self.activation
        mlp.solver = self.solver
        mlp.alpha = self.alpha
        mlp.fit(x_train, y_train)    
        self.mlp = mlp
        
        # predict
        y_train_p = mlp.predict(x_train)
        y_test_p = mlp.predict(x_test)
    
        # print training and test error
        train_error = np.mean((y_train - y_train_p)**2)
        test_error = np.mean((y_test - y_test_p)**2)
        if verbose:
            print(mlp.get_params())
            print("average training error is : %f" %(train_error))
            print("average test error is : %f" %(test_error))
        return train_error, test_error

    def predict(self,x):
        mlp = self.mlp
        return mlp.predict(x)

data = np.loadtxt('h2_matrix.txt')
wfn = fit_wfn(data)
wfn.hidden_layer_sizes = (100,1)
wfn.activation = 'relu'
wfn.solver = 'adam'
wfn.alpha = 0.0001
train_error, test_error = wfn.fit()

# First let's loop through differnt layer sizes
###########
neurons = [20,40,60,80,100]
layers = [1,2,3,4]
loops = 10
train_errors = np.zeros((len(neurons), loops, len(layers)))
test_errors = np.zeros((len(neurons), loops, len(layers)))
colors = cm.rainbow(np.linspace(0, 1, len(neurons)))

plt.figure(figsize=(10,10))
for i,neuron in enumerate(neurons):
    for j in range(loops):
        for k,layer in enumerate(layers):
            wfn.hidden_layer_sizes = (neuron,layer)
            train_errors[i,j,k], test_errors[i,j,k] = wfn.fit() 
    plt.plot(layers, np.mean(train_errors[i], axis=0), 'o-', c=colors[i], 
                label='neurons per layer: %d' %neuron)
    plt.plot(layers, np.mean(test_errors[i], axis=0), 'o--', c=colors[i])

lgd1 = plt.legend(loc='upper left')
l1, = plt.plot([],'r-', label='trainning error')
l2, = plt.plot([],'r--', label='test error')
plt.gca().add_artist(lgd1)
plt.legend(handles=[l1,l2], loc='upper right')
plt.xlabel('number of layers')
plt.ylabel('error')
plt.tight_layout()
plt.savefig('error_layer_size_h2.png')

## we decide to choose layer_size=(100,2)
###########

## let's try to plot electron cloud
###########
wfn.hidden_layer_sizes = (100,3)
wfn.fit()

# fit the data: Put both electrons only on x axis, range from -5 to 5
x1 = np.linspace(-5,5,500)
x2 = np.linspace(-5,5,500)
xx1, xx2 = np.meshgrid(x1, x2)
xx1_1d = xx1.ravel()
xx2_1d = xx2.ravel()
y = np.zeros(len(xx1_1d))
z = np.zeros(len(xx1_1d))
X = np.array([xx1_1d,y,z,xx2_1d,y,z]).T

# predict their psi
y_1d = wfn.predict(X)
y = np.reshape(y_1d,(len(x1), len(x2)))

# plot the contour of psi
plt.figure(figsize=(11,10))
cmap = cm.PRGn
norm = cm.colors.Normalize(vmax=abs(y).max(), vmin=-abs(y).max())
levels = np.arange(y.min(), y.max(), 0.01)
plt.contourf(xx1, xx2, y, levels,cmap=cm.get_cmap(cmap, len(levels) - 1), norm=norm, extend='both')
plt.xlabel('x1')
plt.ylabel('x2')
cb = plt.colorbar()
cb.set_label('wave function')
plt.tight_layout()
plt.savefig('wf.png')
plt.close()


