# You may want to install "gprof2dot"
import io
from collections import Counter

import numpy as np
import scipy.io
import sklearn.model_selection
import sklearn.tree
from numpy import genfromtxt
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin
import pdb

import pydot

eps = 1e-5  # a small number


class DecisionTree:
    def __init__(self, max_depth=3, feature_labels=None):
        self.max_depth = max_depth
        self.features = feature_labels
        self.left, self.right = None, None  # for non-leaf nodes
        self.split_idx, self.thresh = None, None  # for non-leaf nodes
        self.data, self.pred = None, None  # for leaf nodes

    @staticmethod
    def information_gain(X, y, thresh):
        # TODO implement information gain function
        # calculate entropy before cut
        p0 = len(np.where(y==0)[0])/(len(y)+0.001)
        p1 = 1- p0
        entropy_bef = p0 * np.log(p0) + p1 * np.log(p1)  

        # calculate entropy after cut
        # calculate for X> thresh
        idx_large = np.where(X>thresh)[0]
        y_large = y[idx_large]
        p0 = len(np.where(y_large==0)[0])/(len(y_large)+0.001)
        p1 = 1- p0
        entropy_aft_large = p0 * np.log(p0) + p1 * np.log(p1)

        idx_small = np.where(X<=thresh)[0]
        y_small = y[idx_small]
        p0 = len(np.where(y_small==0)[0])/(len(y_small)+0.001)
        p1 = 1 - p0
        entropy_aft_small = p0 * np.log(p0) + p1 * np.log(p1)

        # calculate the gain
        w_large = len(idx_large)/len(y)
        w_small = len(idx_small)/len(y)
        gain = entropy_bef - w_large*entropy_aft_large -  w_small*entropy_aft_small
        return gain

    @staticmethod
    def gini_impurity(X, y, thresh):
        # TODO implement gini_impurity function
        # calculate G before cut
        p0 = len(np.where(y==0)[0])/(len(y)+0.001)
        p1 = 1- p0
        G_bef = 1 - p0**2 - p1**2  

        # calculate G after cut
        # calculate for X> thresh
        idx_large = np.where(X>thresh)[0]
        y_large = y[idx_large]
        p0 = len(np.where(y_large==0)[0])/(len(y_large)+0.001)
        p1 = 1- p0
        G_aft_large = 1- p0**2 - p1**2

        idx_small = np.where(X<=thresh)[0]
        y_small = y[idx_small]
        p0 = len(np.where(y_small==0)[0])/(len(y_small)+0.001)
        p1 = 1 - p0
        G_aft_small = p0 * np.log(p0) + p1 * np.log(p1)

        # calculate the gain
        w_large = len(idx_large)/len(y)
        w_small = len(idx_small)/len(y)
        gain = G_bef - w_large*G_aft_large -  w_small*G_aft_small
 
        return gain

    def split(self, X, y, idx, thresh):
        X0, idx0, X1, idx1 = self.split_test(X, idx=idx, thresh=thresh)
        y0, y1 = y[idx0], y[idx1]
        return X0, y0, X1, y1

    def split_test(self, X, idx, thresh):
        idx0 = np.where(X[:, idx] < thresh)[0]
        idx1 = np.where(X[:, idx] >= thresh)[0]
        X0, X1 = X[idx0, :], X[idx1, :]
        return X0, idx0, X1, idx1

    def fit(self, X, y):
        if self.max_depth > 0:
            # compute entropy gain for all single-dimension splits,
            # thresholding with a linear interpolation of 10 values
            gains = []
            # The following logic prevents thresholding on exactly the minimum
            # or maximum values, which may not lead to any meaningful node
            # splits.
            thresh = np.array([
                np.linspace(np.min(X[:, i]) + eps, np.max(X[:, i]) - eps, num=10)
                for i in range(X.shape[1])
            ])
            for i in range(X.shape[1]):
                gains.append([self.information_gain(X[:, i], y, t) for t in thresh[i, :]])

            gains = np.nan_to_num(np.array(gains))
            self.split_idx, thresh_idx = np.unravel_index(np.argmax(gains), gains.shape)
            self.thresh = thresh[self.split_idx, thresh_idx]
            X0, y0, X1, y1 = self.split(X, y, idx=self.split_idx, thresh=self.thresh)
            if X0.size > 0 and X1.size > 0:
                self.left = DecisionTree(
                    max_depth=self.max_depth - 1, feature_labels=self.features)
                self.left.fit(X0, y0)
                self.right = DecisionTree(
                    max_depth=self.max_depth - 1, feature_labels=self.features)
                self.right.fit(X1, y1)
            else:
                self.max_depth = 0
                self.data, self.labels = X, y
                self.pred = stats.mode(y).mode[0]
        else:
            self.data, self.labels = X, y
            self.pred = stats.mode(y).mode[0]
        return self

    def predict(self, X):
        if self.max_depth == 0:
            return self.pred * np.ones(X.shape[0])
        else:
            X0, idx0, X1, idx1 = self.split_test(X, idx=self.split_idx, thresh=self.thresh)
            yhat = np.zeros(X.shape[0])
            yhat[idx0] = self.left.predict(X0)
            yhat[idx1] = self.right.predict(X1)
            return yhat


class BaggedTrees(BaseEstimator, ClassifierMixin):
    def __init__(self, params=None, n=200):
        if params is None:
            params = {}
        self.params = params
        self.n = n
        self.decision_trees = [
            sklearn.tree.DecisionTreeClassifier(random_state=i, **self.params)
            for i in range(self.n)
        ]

    def fit(self, X, y):
        Ndata = len(y)
        # TODO implement function
        for dt in self.decision_trees:
            idx = np.random.randint(0, Ndata-1, size=Ndata)
            Xi = X[idx]
            yi = y[idx]
            dt.fit(Xi, yi)

    def predict(self, X):
        # TODO implement function
        y = np.zeros(X.shape[0]) 
        for dt in self.decision_trees:
            y += dt.predict(X)
        y /= self.n
        return np.round(y)


class RandomForest(BaggedTrees):
    def __init__(self, params=None, n=200, m=5):
        if params is None:
            params = {}
        self.params = params
        self.n = n
        self.decision_trees = [
            sklearn.tree.DecisionTreeClassifier(random_state=i, max_features=m, **self.params)
            for i in range(self.n)
        ]
        # TODO implement function
        pass


class BoostedRandomForest(RandomForest):
    def fit(self, X, y):
        self.w = np.ones(X.shape[0]) / X.shape[0]  # Weights on data
        self.a = np.zeros(self.n)  # Weights on decision trees
        w = self.w

        # TODO implement function
        # record the number of data points
        ndata = len(y)

        # loop over decision trees
        for i in range(self.n):
            # sample randomly with weights
            idx = np.random.choice(ndata, ndata, p=w)
            Xi = X[idx]
            yi = y[idx]

            # train tree and predict
            self.decision_trees[i].fit(Xi, yi)
            yp = self.decision_trees[i].predict(X)

            # calculate ej and aj
            idx_wrong = np.where(yp!=y)[0]
            ej = np.sum(w[idx_wrong])/np.sum(w)
            aj = 0.5 * np.log((1-ej)/ej)
            self.a[i] = aj

            # update weight
            w = w * np.exp(-aj)
            w[idx_wrong] = w[idx_wrong] * np.exp(aj)

            # normalize w
            w /= np.sum(w)

        self.w = w
        idx_max = np.argmax(self.w)
        #print('max')
        #print(X[idx_max])
        idx_min = np.argmin(self.w)
        #print('min')
        #print(X[idx_min])
        return self

    def predict(self, X):
        # TODO implement function
        # initialize z for c=0/1
        z0 = np.zeros(len(X))
        z1 = np.zeros(len(X))

        # loop over decision trees
        for i in range(self.n):
            yp = self.decision_trees[i].predict(X)

            # calculate z
            idx0 = np.where(yp==0)[0]
            z0[idx0] += self.a[i]
            idx1 = np.where(yp==1)[0]
            z1[idx1] += self.a[i]

        labels = np.zeros(len(X))
        idx = np.where(z0<z1)[0]
        labels[idx] = 1
        return labels

def preprocess(data, fill_mode=True, min_freq=10, onehot_cols=[]):
    # fill_mode = False

    # Temporarily assign -1 to missing data
    data[data == b''] = '-1'

    # Hash the columns (used for handling strings)
    onehot_encoding = []
    onehot_features = []
    for col in onehot_cols:
        counter = Counter(data[:, col])
        for term in counter.most_common():
            if term[0] == b'-1':
                continue
            if term[-1] <= min_freq:
                break
            onehot_features.append(term[0])
            onehot_encoding.append((data[:, col] == term[0]).astype(np.float))
        data[:, col] = '0'
    onehot_encoding = np.array(onehot_encoding).T
    data = np.hstack([np.array(data, dtype=np.float), np.array(onehot_encoding)])

    # Replace missing data with the mode value. We use the mode instead of
    # the mean or median because this makes more sense for categorical
    # features such as gender or cabin type, which are not ordered.
    if fill_mode:
        for i in range(data.shape[-1]):
            mode = stats.mode(data[((data[:, i] < -1 - eps) +
                                    (data[:, i] > -1 + eps))][:, i]).mode[0]
            data[(data[:, i] > -1 - eps) * (data[:, i] < -1 + eps)][:, i] = mode

    return data, onehot_features


def evaluate(clf):
    print("Cross validation", sklearn.model_selection.cross_val_score(clf, X, y))
    if hasattr(clf, "decision_trees"):
        counter = Counter([t.tree_.feature[0] for t in clf.decision_trees])
        first_splits = [(features[term[0]], term[1]) for term in counter.most_common()]
        print("First splits", first_splits)

    accu = 1 - np.sum(abs(y - clf.predict(X)))/len(y)
    print("accuracy: %f" %accu)


if __name__ == "__main__":
    #dataset = "titanic"
    dataset = "spam"
    params = {
        "max_depth": 5,
        # "random_state": 6,
        "min_samples_leaf": 10,
    }
    N = 100

    if dataset == "titanic":
        # Load titanic data
        path_train = 'titanic_training.csv'
        data = genfromtxt(path_train, delimiter=',', dtype=None)
        path_test = 'titanic_testing_data.csv'
        test_data = genfromtxt(path_test, delimiter=',', dtype=None)
        y = data[1:, 0]  # label = survived
        class_names = ["Died", "Survived"]

        labeled_idx = np.where(y != b'')[0]
        y = np.array(y[labeled_idx], dtype=np.int)
        print("\n\nPart (b): preprocessing the titanic dataset")
        X, onehot_features = preprocess(data[1:, 1:], onehot_cols=[1, 5, 7, 8])
        X = X[labeled_idx, :]
        Z, _ = preprocess(test_data[1:, :], onehot_cols=[1, 5, 7, 8])
        assert X.shape[1] == Z.shape[1]
        features = list(data[0, 1:]) + onehot_features

    elif dataset == "spam":
        features = [
            "pain", "private", "bank", "money", "drug", "spam", "prescription", "creative",
            "height", "featured", "differ", "width", "other", "energy", "business", "message",
            "volumes", "revision", "path", "meter", "memo", "planning", "pleased", "record", "out",
            "semicolon", "dollar", "sharp", "exclamation", "parenthesis", "square_bracket",
            "ampersand"
        ]
        assert len(features) == 32

        # Load spam data
        path_train = 'spam_data.mat'
        data = scipy.io.loadmat(path_train)
        X = data['training_data']
        y = np.squeeze(data['training_labels'])
        Z = data['test_data']
        class_names = ["Ham", "Spam"]

    else:
        raise NotImplementedError("Dataset %s not handled" % dataset)

    print("Features:", features)
    print("Train/test size:", X.shape, Z.shape)

    print("\n\nPart 0: constant classifier")
    print("Accuracy", 1 - np.sum(y) / y.size)

    # Basic decision tree
    print("\n\nPart (a-b): simplified decision tree")
    #dt = DecisionTree(max_depth=3, feature_labels=features)
    #dt.fit(X, y)
    #print("Predictions", dt.predict(Z)[:100])

    print("\n\nPart (c): sklearn's decision tree")
    clf = sklearn.tree.DecisionTreeClassifier(random_state=0, **params)
    clf.fit(X, y)
    evaluate(clf)
    out = io.StringIO()
    sklearn.tree.export_graphviz(
        clf, out_file=out, feature_names=features, class_names=class_names)
    graph = pydot.graph_from_dot_data(out.getvalue())
    pydot.graph_from_dot_data(out.getvalue())[0].write_pdf("%s-tree.pdf" % dataset)

    # TODO implement and evaluate parts c-h
    bt = BaggedTrees()
    bt.fit(X,y)
    #evaluate(bt)
    zp = bt.predict(Z)
    np.savetxt('zp_spam', zp, fmt='%d')

    #bt = RandomForest(m=6)
    #bt.fit(X,y)
    #evaluate(bt)

    #bt = BoostedRandomForest()
    #bt.fit(X,y)
    #evaluate(bt)

    
