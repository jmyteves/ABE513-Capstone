import sys
# sys.path.append ('../utils')

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from decisionRegions import *

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt




df_wine = pd.read_csv('wine.data', header=None)
X,y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values
X_train, X_test, y_train, y_test= train_test_split (X,y, test_size=0.3, random_state=0)

# StandardScalear centers the data.
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform (X_test)

cov_mat = np.cov (X_train_std.T)
# Use eigh for semetric matricies...
eigen_vals, eigen_vecs = np.linalg.eigh(cov_mat)
print ("\nEigneValues \n%s" % eigen_vals)

#Look at how the variance is distributed.
# Lanbda_i / sum over lambda.

tot = sum(eigen_vals)
var_exp = [ i/tot for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)


if True:
    plt.figure()
    plt.bar(range(1,14), var_exp, alpha=0.5, align='center', label='individual expained variance')
    plt.step(range(1,14), cum_var_exp, where='mid', label = 'cumulative  explained variance')
    plt.ylabel ("Explained Variance Ratio")
    plt.xlabel ('Principle Components')
    plt.legend(loc='best')
    plt.savefig('Variance.png')


eigen_pairs = [ (np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(reverse=True)



w = np.hstack ((eigen_pairs[0][1][:,np.newaxis],
                eigen_pairs[1][1][:,np.newaxis]))


print ('Matrix W:\n', w)

X_train_pca = X_train_std.dot(w)

if True:
    colors = ['r', 'b','g']
    markers = ['s', 'x','o']
    plt.figure()
    for l,c,m in zip (np.unique(y_train), colors, markers):
        plt.scatter(X_train_pca[y_train==l,0],
                X_train_pca[y_train==l, 1],
                c=c, label = l, marker =m , hold=True, alpha=0.3)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(loc='Lower Left')
    plt.grid('on')
 #   plt.savefig ('ProjectedData.png')
    # plt.show()


if True:
    pca = PCA (n_components=2)
    lr =  LogisticRegression()
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform (X_test_std)
    lr.fit (X_train_pca, y_train)    

    # Plot the training set.
    plt.figure()
    plot_decision_regions( X_train_pca, y_train, classifier=lr)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(loc='lower left' )
#    plt.savefig ('TrainClassified.png')

    # Plot the testing set.
    plt.figure()
    plot_decision_regions( X_test_pca, y_test, classifier=lr)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(loc='lower left' )
#    plt.savefig ('TestClassified.png')


    
    plt.show()
