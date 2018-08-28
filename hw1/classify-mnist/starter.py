import numpy as np
import matplotlib.pyplot as plt

# Load the training dataset
train_features = np.load("train_features.npy")
train_labels = np.load("train_labels.npy").astype("int8")

n_train = train_labels.shape[0]

def visualize_digit(features, label):
    # Digits are stored as a vector of 400 pixel values. Here we
    # reshape it to a 20x20 image so we can display it.
    plt.imshow(features.reshape(20, 20), cmap="binary")
    plt.xlabel("Digit with label " + str(label))
    plt.show()

# Visualize a digit
# visualize_digit(train_features[0,:], train_labels[0])

# TODO: Plot three images with label 0 and three images with label 1
idx_0 = np.where(train_labels==0)[0][:3]
for i in idx_0:
    plt.figure()
    visualize_digit(train_features[i,:], train_labels[i])
    plt.savefig('fig_0_%d' %i)
    plt.close()
idx_1 = np.where(train_labels==1)[0][:3]
for i in idx_1:
    plt.figure()
    visualize_digit(train_features[i,:], train_labels[i])
    plt.savefig('fig_1_%d' %i)
    plt.close()


# Linear regression

# TODO: Solve the linear regression problem, regressing
# X = train_features against y = 2 * train_labels - 1
X_train = np.matrix(train_features)
y_train = np.matrix(2 * train_labels - 1).T
w = (X_train.T @ X_train).I @ X_train.T @ y_train


# TODO: Report the residual error and the weight vector
print('weight vector')
print(w)

RE = np.linalg.norm(X_train @ w - y_train) **2
print('residual error: %.2f' %RE)

# Load the test dataset
# It is good practice to do this after the training has been
# completed to make sure that no training happens on the test
# set!
test_features = np.load("test_features.npy")
test_labels = np.load("test_labels.npy").astype("int8")

n_test = test_labels.shape[0]

# TODO: Implement the classification rule and evaluate it
# on the training and test set
print('######## when regression targets -1/1 without bias ###########')

print('On training set:')
y_train = X_train @ w 
idx_0 = np.where(y_train<0)[0]
idx_1 = np.where(y_train>0)[0]
idx_0_right = np.where(train_labels[idx_0] == 0)[0]
print('for %d digits that are recognized as 0, %d are really 0' %(len(idx_0), len(idx_0_right))) 
idx_1_right = np.where(train_labels[idx_1] == 1)[0]
print('for %d digits that are recognized as 1, %d are really 1' %(len(idx_1), len(idx_1_right))) 
print('correct persentage: %.5f' %((len(idx_0_right)+len(idx_1_right))/(len(idx_0)+len(idx_1))))

print('On test set:')
X_test = np.matrix(test_features)
y_test = X_test @ w 
idx_0 = np.where(y_test<0)[0]
idx_1 = np.where(y_test>0)[0]
idx_0_right = np.where(test_labels[idx_0] == 0)[0]
print('for %d digits that are recognized as 0, %d are really 0' %(len(idx_0), len(idx_0_right))) 
idx_1_right = np.where(test_labels[idx_1] == 1)[0]
print('for %d digits that are recognized as 1, %d are really 1' %(len(idx_1), len(idx_1_right))) 
print('correct persentage: %.5f' %((len(idx_0_right)+len(idx_1_right))/(len(idx_0)+len(idx_1))))

# TODO: Try regressing against a vector with 0 for class 0
# and 1 for class 1
print('######## when regression targets 0/1 without bias ###########')

print('On training set:')
y_train = np.matrix(train_labels).T
w = (X_train.T @ X_train).I @ X_train.T @ y_train
y_train = X_train @ w 
idx_0 = np.where(y_train<0.5)[0]
idx_1 = np.where(y_train>0.5)[0]

idx_0_right = np.where(train_labels[idx_0] == 0)[0]
print('for %d digits that are recognized as 0, %d are really 0' %(len(idx_0), len(idx_0_right))) 
idx_1_right = np.where(train_labels[idx_1] == 1)[0]
print('for %d digits that are recognized as 1, %d are really 1' %(len(idx_1), len(idx_1_right))) 
print('correct persentage: %.5f' %((len(idx_0_right)+len(idx_1_right))/(len(idx_0)+len(idx_1))))


print('On test set:')
X_test = np.matrix(test_features)
y_test = X_test @ w 
idx_0 = np.where(y_test<0.5)[0]
idx_1 = np.where(y_test>0.5)[0]

idx_0_right = np.where(test_labels[idx_0] == 0)[0]
print('for %d digits that are recognized as 0, %d are really 0' %(len(idx_0), len(idx_0_right))) 
idx_1_right = np.where(test_labels[idx_1] == 1)[0]
print('for %d digits that are recognized as 1, %d are really 1' %(len(idx_1), len(idx_1_right))) 
print('correct persentage: %.5f' %((len(idx_0_right)+len(idx_1_right))/(len(idx_0)+len(idx_1))))


# TODO: Form a new feature matrix with a column of ones added
# and do both regressions with that matrix
print('######## when regression targets 0/1 with bias ###########')

print('On training set:')
X_train = np.matrix(np.hstack((train_features, np.ones((len(train_features),1)))))
y_train = np.matrix(train_labels).T
w = (X_train.T @ X_train).I @ X_train.T @ y_train
y_train = X_train @ w 
idx_0 = np.where(y_train<0.5)[0]
idx_1 = np.where(y_train>0.5)[0]
idx_0_right = np.where(train_labels[idx_0] == 0)[0]
print('for %d digits that are recognized as 0, %d are really 0' %(len(idx_0), len(idx_0_right))) 
idx_1_right = np.where(train_labels[idx_1] == 1)[0]
print('for %d digits that are recognized as 1, %d are really 1' %(len(idx_1), len(idx_1_right))) 
print('correct persentage: %.5f' %((len(idx_0_right)+len(idx_1_right))/(len(idx_0)+len(idx_1))))

print('On test set:')
X_test = np.matrix(np.hstack((test_features, np.ones((len(test_features),1)))))
y_test = X_test @ w 
idx_0 = np.where(y_test<0.5)[0]
idx_1 = np.where(y_test>0.5)[0]
idx_0_right = np.where(test_labels[idx_0] == 0)[0]
print('for %d digits that are recognized as 0, %d are really 0' %(len(idx_0), len(idx_0_right))) 
idx_1_right = np.where(test_labels[idx_1] == 1)[0]
print('for %d digits that are recognized as 1, %d are really 1' %(len(idx_1), len(idx_1_right))) 
print('correct persentage: %.5f' %((len(idx_0_right)+len(idx_1_right))/(len(idx_0)+len(idx_1))))

print('######## when regression targets -1/1 with bias ###########')
print('On training set:')
X_train = np.matrix(np.hstack((train_features, np.ones((len(train_features),1)))))
y_train = np.matrix(train_labels*2 - 1).T
w = (X_train.T @ X_train).I @ X_train.T @ y_train
y_train = X_train @ w 
idx_0 = np.where(y_train<0)[0]
idx_1 = np.where(y_train>0)[0]
idx_0_right = np.where(train_labels[idx_0] == 0)[0]
print('for %d digits that are recognized as 0, %d are really 0' %(len(idx_0), len(idx_0_right))) 
idx_1_right = np.where(train_labels[idx_1] == 1)[0]
print('for %d digits that are recognized as 1, %d are really 1' %(len(idx_1), len(idx_1_right))) 
print('correct persentage: %.5f' %((len(idx_0_right)+len(idx_1_right))/(len(idx_0)+len(idx_1))))

print('On test set:')
X_test = np.matrix(np.hstack((test_features, np.ones((len(test_features),1)))))
y_test = X_test @ w 
idx_0 = np.where(y_test<0)[0]
idx_1 = np.where(y_test>0)[0]
idx_0_right = np.where(test_labels[idx_0] == 0)[0]
print('for %d digits that are recognized as 0, %d are really 0' %(len(idx_0), len(idx_0_right))) 
idx_1_right = np.where(test_labels[idx_1] == 1)[0]
print('for %d digits that are recognized as 1, %d are really 1' %(len(idx_1), len(idx_1_right))) 
print('correct persentage: %.5f' %((len(idx_0_right)+len(idx_1_right))/(len(idx_0)+len(idx_1))))



# Logistic Regression

# You can also compare against how well logistic regression is doing.
# We will learn more about logistic regression later in the course.

import sklearn.linear_model

lr = sklearn.linear_model.LogisticRegression()
lr.fit(X, train_labels)

test_error_lr = 1.0 * sum(lr.predict(test_features) != test_labels) / n_test
