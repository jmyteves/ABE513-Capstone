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
        self.pStart , self.pStop , self.pStep = .001, .060 , 0.002 
        
        

    def __iter__ (self):
        self.kfold = sk_ms.KFold( self.nSplits).split(self.x)
        return self

    # Returns a vector of the number of disagreements and the number of features.
    def next( self):
        trainIndex, testIndex =  self.kfold.next()
        print "Train", trainIndex, "Test", testIndex
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
            returns.append( [ curPenelty, nDisagree , nFeatures ])      
            curPenelty += self.pStep 
        return np.array(returns)
    

# Given an input vector of disagreemments vs featurs, plot the results.
def plotVectors ( v ):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ins1=ax1.plot( v[:,0], v[:,1], 'b-', label='Percent Disagreement')
    ins2=ax2.plot( v[:,0], v[:,2], 'g-', label='Number Features')
    ax1.set_ylabel("Percent Disagreement")
    ax2.set_ylabel("Number of Features")
    ax1.set_xlabel ('Penelty Factor')

    lns = ins1+ins2
    labs = [l.get_label() for l in lns ]
    ax1.legend( lns, labs, loc='upper center')
    ax1.grid(True)
    
    
    

df_wine = pd.read_csv('wine.data', header=None)
rIndex = sk_utils.shuffle( range(len(df_wine)))
X,y = df_wine.iloc[:,1:].values[rIndex], df_wine.iloc[:,0].values[rIndex]

lTest = LTest ( X,y,10)
res = []
for xx in lTest:
     res.append(xx)
     # plotVectors(xx)

#plotVectors(res[4] )
#plt.savefig ('../tex/LassoSelection.png')
   
