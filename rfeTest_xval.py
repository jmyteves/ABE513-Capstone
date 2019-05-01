# Recursive feature elimination.

import numpy as np, pandas as pd , matplotlib.pyplot as plt

import sklearn.linear_model as sk_lm
import sklearn.preprocessing as sk_preprocess
import sklearn.utils as sk_utils
import sklearn.model_selection as sk_ms
import sklearn.feature_selection as sk_fs

import statsmodels.api as sm
import scipy.stats as stats


import tabulate
import pickle


df_wine = pd.read_csv('wine.data', header=None)
rIndex = sk_utils.shuffle( range(len(df_wine)))
X,y = df_wine.iloc[:,1:].values[rIndex], df_wine.iloc[:,0].values[rIndex]




if False:
     data = []
     lr = sk_lm.LogisticRegression( penalty='l1', C=10000 )
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

def calcFoldScores ( nFold = 10):
     # kFold = sk_ms.KFold ( nFold).split(X,y)
     foldReturns = []
     lr = sk_lm.LogisticRegression( penalty='l1', C=10000 )
     for trainIndex,  testIndex in sk_ms.KFold(nFold).split(X,y):
          xTrain, yTrain = X[trainIndex], y[trainIndex]
          xTest,yTest = X[testIndex], y[testIndex]
          featureReturns   = []
          for nFeatures in range(1,14):
              
               rfe = sk_fs.RFE( lr, n_features_to_select = nFeatures )
               rfe.fit(xTrain,yTrain)
               score = rfe.score( xTest, yTest)
               featureReturns.append(( nFeatures, score)) 
          foldReturns.append(featureReturns)
     returns = np.array(foldReturns)
     return returns


if False:
     foldReturns = calcFoldScores(5)
     pickle.dump ( foldReturns, open('foldReturns.pkl', 'wb'))
else:
     foldReturns =pickle.load( open('foldReturns.pkl', 'rb'))
     
avgScores= np.array([foldReturns[:,nFeature,1].mean() for nFeature in range (foldReturns.shape[1])])
stdScores=  np.array([foldReturns[:,nFeature,1].std() for nFeature in range (foldReturns.shape[1])])

scores = foldReturns.reshape(-1,2)
indexes = scores[:,0].argsort()
sortedScores = scores[indexes]

if True:
     yy = np.log(stdScores)
     xx = np.array(range(len(stdScores))) +1.0

     # One variable.
     term_1 = np.zeros([len(stdScores),2])
     term_1[:,0] = 1; term_1[:,1] = xx 
     ols_1  = sm.OLS ( yy, term_1 ).fit()

     # Quadratic  case.
     term_2 = np.zeros([len(stdScores),3])
     term_2[:,0] = 1; term_2[:,1] = xx ; term_2 [:,2] = xx**2
     ols_2 = sm.OLS ( yy, term_2 ).fit()
     
     term_3 = np.zeros([len(stdScores),4])
     term_3[:,0] = 1; term_3[:,1] = xx ; term_3[:,2] = xx**2;  term_3[:,3] = xx**3
     ols_3  = sm.OLS ( yy, term_3 ).fit()
     
     plt.figure()
     l0 = plt.scatter( xx , yy )
     l1 = plt.plot( xx , ols_1.fittedvalues)
     l2 = plt.plot( xx , ols_2.fittedvalues)
     #l3 = plt.plot( xx , ols_3.fittedvalues)
     

     #leg = [  "Linear" , "Quadratic", "Cubic" ,"Log Std Err"]
     leg = [  "Linear" , "Quadratic" ,"Log Std Err"]

     # http://www.jerrydallal.com/lhsp/extra.htm
     # http://tex.stackexchange.com/questions/131867/using-multicolumn-in-latex
     plt.grid(True)
     plt.xlabel("Number of features selected")
     plt.ylabel("Log Std of no. feature  selected")
     plt.legend( leg , loc="upper right")
     plt.savefig("../tex/VarianceFitted.png")
     plt.show()
     
     mse_32 = (ols_3.ess - ols_2.ess) / (ols_3.df_model - ols_2.df_model) / ols_3.mse_resid
     mse_31 = (ols_3.ess - ols_1.ess) / (ols_3.df_model - ols_1.df_model)  / ols_3.mse_resid
     pval_32 = stats.f.sf(mse_32,1,ols_3.df_model)
     pval_31 = stats.f.sf(mse_31,2,ols_3.df_model)

     with open ('../tex/fittedValues.txt', 'w') as f:
          print >> f, "OLS Linear Fit:"
          print  >> f,  ols_1.summary2().tables[1]
          print >> f, ""
          print  >> f,"OLS Quadratic fit:"
          print  >> f, ols_2.summary2().tables[1]
          #print  >> f, "OLS Cubic Fit: "
          #print ols_3.summary2().tables[1] 

     

if True:
     plt.figure()
     plt.errorbar(range(1,len(avgScores)+1), avgScores ,  yerr = stdScores)
     #plt.plot( avgScores)
     
     plt.xlabel("Number of features selected")
     plt.ylabel("Cross validation score (nb of correct classifications)")
     #plt.plot(range(1, len(rfevc.grid_scores_) + 1), rfevc.grid_scores_)
     plt.grid(True)
     plt.savefig("../tex/XValScores.png")
     plt.show()








# if False:
#      nFold = 3
#      lr = sk_lm.LogisticRegression( penalty='l1', C=10000 )
#      rfevc = sk_fs.RFECV(estimator=lr, step=1, \
#                          cv=sk_ms.StratifiedKFold(nFold), scoring ='accuracy')
#      rfevc.fit(X,y)

#      #Plot number of features VS. cross-validation scores
#      plt.figure()
#      plt.xlabel("Number of features selected")
#      plt.ylabel("Cross validation score (nb of correct classifications)")
#      plt.plot(range(1, len(rfevc.grid_scores_) + 1), rfevc.grid_scores_)
#      plt.grid(True)
#      #plt.savefig("../tex/XValScores.png")
#      plt.show()
