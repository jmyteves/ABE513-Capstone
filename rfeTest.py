# Recursive feature elimination.

import numpy as np, pandas as pd , matplotlib.pyplot as plt

import sklearn.linear_model as sk_lm
import sklearn.preprocessing as sk_preprocess
import sklearn.utils as sk_utils
import sklearn.model_selection as sk_ms
import sklearn.feature_selection as sk_fs

import tabulate
   

df_wine = pd.read_csv('wine.data', header=None)
rIndex = sk_utils.shuffle( range(len(df_wine)))
X,y = df_wine.iloc[:,1:].values[rIndex], df_wine.iloc[:,0].values[rIndex]


lr = sk_lm.LogisticRegression( penalty='l1', C=10000 )

if True:
     data = []
     for nToSelect in range (1,14):
          rfe = sk_fs.RFE(lr, n_features_to_select= nToSelect)
          rfe.fit(X,y)
          # RFE has ranking of selected features.
          # rfe.ranking_
          data.append(rfe.ranking_)
     xx = pd.DataFrame (data)
     xx.index = range(1,14)
     with ( open ("../tex/RFE_Features.tbl", 'w')) as f: 
          print >> f, tabulate.tabulate ( xx, tablefmt='latex', floatfmt=".3f" , headers="keys")
     print (" %d Selected Features : %s " % (nToSelect,  rfe.ranking_))

# print "Features sorted by their rank:"
# # print sorted(zip(map(lambda x: round(x, 4), rfe.ranking_),))



rfevc = sk_fs.RFECV(estimator=lr, step=1, cv=sk_ms.StratifiedKFold(10), scoring ='accuracy')
rfevc.fit(X,y)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfevc.grid_scores_) + 1), rfevc.grid_scores_)
plt.grid(True)
plt.savefig("../tex/XValScores.png")
plt.show()
