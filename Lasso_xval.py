# Lasso_xval.
import numpy as np, pandas as pd , matplotlib.pyplot as plt

import sklearn.linear_model as sk_lm
import sklearn.preprocessing as sk_preprocess
import sklearn.utils as sk_utils
import sklearn.model_selection as sk_ms


# Determines numerical count for disagreement.
def calcDisagree ( yEstimate, yTest ) :
     nDisagree = (yTest == yEstimate) == False
     totDisagree = nDisagree.sum().astype(float)
     return totDisagree/len(yTest)
    

# Determine number of features.
def countFeatures ( lm ):
     arr = np.zeros ( lm.coef_.shape[1])
     for k in range(lm.coef_.shape[0]):
          arr += np.abs(lm.coef_[k,:]) > .0001
     return (arr !=0).sum()




class LTest:
    
    def __init__(self, x,y,nSplits=10):
        self.x, self.y , self.nSplits = x,y,nSplits
        # Good values - .1
        self.pStart , self.pStop , self.pStep = 1e-6, 0.200 , 0.005
        #self.pStart , self.pStop , self.pStep = 1e-6, 1.000 , 0.02
        #self.pStart , self.pStop , self.pStep = 1e-6, 6.000 , 0.01
        
        

    def __iter__ (self):
        self.kfold = sk_ms.KFold( self.nSplits).split(self.x)
        return self
    # Run cross Validation
    # http://scikit-learn.org/stable/modules/cross_validation.html
    # def runXVal ( self, curPenelty):
    #      lr = sk_lm.LogisticRegression( penalty='l1', C=curPenelty )
    #      scores = sk_ms.cross_val_score(lr, self.x, self.y, cv=self.nSplits)
    #      return scores
       

    def runXVal ( self ):
         results = np.array([ xx for xx in self ])
         # For each value over the tuning range
         res = []
         for k in range ( results.shape[1]) :
              pErrors, nFeatures  = results[:,k,1], results[:,k,2]
              meanErrors , stdErrors= pErrors.mean(), pErrors.std()
              meanFeatures, stdFeatures = nFeatures.mean(), nFeatures.std()
              res.append( (results[0,k,0], meanErrors, meanFeatures,  stdErrors, stdFeatures))
         rv = np.array ( res)
         return rv
        
    # Returns a vector of the number of disagreements and the number of features.
    def next( self):
        trainIndex, testIndex =  self.kfold.next()
        #print "Train", trainIndex, "Test", testIndex
        xTrain , yTrain = self.x[trainIndex], self.y[trainIndex]
        xTest  , yTest  = self.x[testIndex], self.y[testIndex]

        returns = []
        curPenelty = self.pStart
        while curPenelty <= self.pStop:
            lr = sk_lm.LogisticRegression( penalty='l1', C=curPenelty )
            lr.fit( xTrain, yTrain)
            yEstimate = lr.predict( xTest)
            nDisagree = calcDisagree( yEstimate, yTest)
            nFeatures = countFeatures ( lr)
            returns.append( [ curPenelty, 1.0-nDisagree , nFeatures ])      
            curPenelty += self.pStep 
        return np.array(returns)
    

# Given an input vector of disagreemments vs featurs, plot the results.
def plotVectors ( v ):
     hasStd  = v.shape[1]==5
     hasStd = True
     fig, ax1 = plt.subplots()
     ax2 = ax1.twinx()
     ins1=ax1.plot( v[:,0], v[:,1]*100, 'b-', label='Percent Agreement')
     if hasStd:
          ax1.errorbar(  v[:,0], v[:,1]*100, yerr = v[:,3]*100 ,
                         ecolor='b', fmt='o')
     ins2=ax2.plot( v[:,0], v[:,2], 'g-', label='Number Features')
     if hasStd:
          ax2.errorbar(  v[:,0], v[:,2], ecolor='g' , color='g', yerr = v[:,4] , fmt='o')
     ax1.set_ylabel("Percent Agreement")
     ax2.set_ylabel("Average Number of Features")
     ax1.set_xlabel ('Penalty Factor')
     
     lns = ins1+ins2
     labs = [l.get_label() for l in lns ]
     ax1.legend( lns, labs, loc='center right')
     ax1.grid(True)
    
    
    

df_wine = pd.read_csv('wine.data', header=None)
rIndex = sk_utils.shuffle( range(len(df_wine)))
X,y = df_wine.iloc[:,1:].values[rIndex], df_wine.iloc[:,0].values[rIndex]

lTest = LTest ( X,y,3)
xVals = lTest.runXVal( )
plotVectors(xVals)
#plt.savefig ('../tex/LassoSelection_xval.png')

