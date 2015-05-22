# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 01:55:47 2015

@author: marios michailidis

"""
# licence: FreeBSD

"""
Copyright (c) 2015, Marios Michailidis
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
 list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import random

import numpy as np

import scipy as spss
from scipy.sparse import csr_matrix
import sys
sys.path.append("../../xgboost/wrapper")
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier

thre=20
num_round=1150
lr=0.05
max_de=7
subsam=0.4
colsample_bytree=0.5
gamma =0.001
min_child_weight=0.05
seed=1


objective='multi:softprob'
param = {}
param['booster']= 'gbtree'#gblinear
param['objective'] = objective
param['bst:eta'] = lr
param['seed']=seed	   
param['bst:max_depth'] = max_de
param['eval_metric'] = 'auc'
param['bst:min_child_weight']=min_child_weight
param['silent'] = 1
param['nthread'] = thre
param['bst:subsample'] = subsam 
param['num_class'] = 9
param['gamma'] = gamma
param['colsample_bytree']=colsample_bytree


def transform2dtos(D2,y2):
   # transform a 2d array of predictions to single array
   # we also change
   d1=[]
   y1=[]
   for i in range (0,len(D2)):
      for j in range (0,len(D2[0])):
            d1.append(float(D2[i][j]))
            if y2[i]==float(j):
                y1.append(1.0)
            else:
                y1.append(0.0)     
                
   return d1,y1
   
""" print predictions in file"""
    
def printfilewithtarget(X, name):
    print("start print the training file with target")
    wfile=open(name + ".csv", "w")
    for i in range (0, len(X)):
        wfile.write(str(X[i][0]) )
        for j in range (1, len(X[i])):
           wfile.write("," +str(X[i][j]) ) 
        wfile.write("\n")
    wfile.close()
    print("done")            
    
""" the metric we are being tested on"""

def logloss_metric(p, y):
     logloss=0
     for i in range (0, len(p)):
        for j in range (0,len(p[i])):
                if y[i]==float(j):   
                     logloss+= np.log(spss.maximum(spss.minimum(p[i][j],1-(1e-15) ),1e-15 ))
     return -logloss/float(len(y))

"""Load a csv file"""

def load(name):
    print("start  reading file with target")
    
    wfile=open(name , "r")
    
    line=wfile.readline().replace("\n","")
    splits=line.split(",")
    datalen=len(splits)
    wfile.close()
    X = np.loadtxt(open( name), delimiter=',',usecols=range(0, datalen), skiprows=0)       
    print("done") 
    return np.array(X) 
    
""" use to concatebate the various kfold sets together"""
def cving(x1, x2, x3, x4,x5, y1 ,y2, y3, y4, y5, ind1, ind2, ind3, ind4 ,ind5, num):
    if num==0:
        xwhole=np.concatenate((x2,x3,x4,x5), axis=0) 
        yhol=np.concatenate((y2,y3,y4,y5), axis=0) 
        return x1,y1 ,ind1,xwhole,yhol
    elif num==1:
        xwhole=np.concatenate((x1,x3,x4,x5), axis=0) 
        yhol=np.concatenate((y1,y3,y4,y5), axis=0) 
        return x2,y2 ,ind2,xwhole,yhol    
    elif num==2:
        xwhole=np.concatenate((x1,x2,x4,x5), axis=0) 
        yhol=np.concatenate((y1,y2,y4,y5), axis=0) 
        return x3,y3 ,ind3,xwhole,yhol             
    elif num==3:
        xwhole=np.concatenate((x1,x2,x3,x5), axis=0) 
        yhol=np.concatenate((y1,y2,y3,y5), axis=0) 
        return x4,y4 ,ind4,xwhole,yhol      
    else :
        xwhole=np.concatenate((x1,x2,x3,x4), axis=0) 
        yhol=np.concatenate((y1,y2,y3,y4), axis=0) 
        return x5,y5 ,ind5,xwhole,yhol   
        
""" Splits data to 5 kfold sets"""        
def split_array_in_5(array, seed):
    random.seed(seed)
    
    new_arra1=[]
    new_arra2=[]
    new_arra3=[]
    new_arra4=[] 
    new_arra5=[] 
    
    indiceds1=[]
    indiceds2=[]
    indiceds3=[]
    indiceds4=[]    
    indiceds5=[] 
    
    for j in range (0,len(array)):
        rand=random.random()
        if rand <0.2:
            new_arra1.append(array[j])
            indiceds1.append(j)
        elif rand <0.4:
            new_arra2.append(array[j]) 
            indiceds2.append(j)            
        elif rand <0.6:
            new_arra3.append(array[j]) 
            indiceds3.append(j)            
        elif rand <0.8:
            new_arra4.append(array[j]) 
            indiceds4.append(j)  
        else :
            new_arra5.append(array[j]) 
            indiceds5.append(j)               
    #convert to numpy        
    new_arra1=np.array(new_arra1)
    new_arra2=np.array(new_arra2)
    new_arra3=np.array(new_arra3)
    new_arra4=np.array(new_arra4)  
    new_arra5=np.array(new_arra5) 
     
    #return arrays and indices
    return new_arra1,new_arra2,new_arra3,new_arra4,new_arra5,indiceds1,indiceds2,indiceds3,indiceds4,indiceds5     

        
def scalepreds(prs):
    
    for i in range  (0, len(prs)):
        suum=0.0
        for j in range (0,9):
            suum+=prs[i][j]
        for j in range (0,9):
            prs[i][j]/=suum 
    

"""loads first columns of a file"""

def loadfirstcolumn(filename):
    pred=[]
    op=open(filename,'r')
    op.readline() #header
    for line in op:
        line=line.replace('\n','')
        sp=line.split(',')
        #load always the last columns
        pred.append(sp[0])
    op.close()
    return pred  
    
"""loads last columns of a file"""

def loadlastcolumn(filename):
    pred=[]
    op=open(filename,'r')
    op.readline() #header
    for line in op:
        line=line.replace('\n','')
        sp=line.split(',')
        #load always the last columns
        pred.append(float(sp[len(sp)-1])-1.0)
    op.close()
    return pred 



""" This is the main method"""
            
def main():
    directory=''
    train_file="old4gramtrain.csv"
    test_file="old4gramtest.csv" 
    SEED= 15 
    outset="4gram"
    y= loadlastcolumn(directory+"trainLabels.csv")
    ids=loadfirstcolumn(directory+"sampleSubmission.csv")    
    include_inpretrain=True    
    model=ExtraTreesClassifier(n_estimators=100, criterion='entropy', max_depth=16, min_samples_split=2,min_samples_leaf=1, max_features=0.5,n_jobs=20, random_state=1)   
    trainini_file=  ["old1gramtrain.csv"] 
    testini_file =  ["old1gramtest.csv"]     

    X=load(train_file)
    print ("train samples: %d columns: %d " % (len(X) , len(X[0])))
    X_test=load(test_file)
    print ("train samples: %d columns: %d" % (len(X_test) , len(X_test[0])))
    if include_inpretrain:
        for t in range(0,len(trainini_file)):
            Xini=load(trainini_file[t])
            print ("train samples: %d columns: %d " % (len(Xini) , len(Xini[0])))
            X_testini=load(testini_file[t])
            print ("train samples: %d columns: %d" % (len(X_testini) , len(X_testini[0])))  
            X=np.column_stack((X,Xini))
            X_test=np.column_stack((X_test,X_testini)) 
            print ("train after merge samples: %d columns: %d" % (len(X) , len(X[0])))
            print ("train  after merge samples: %d columns: %d" % (len(X_test) , len(X_test[0])))
         
    number_of_folds=5 # repeat the CV procedure 10 times to get more precise results
    train_stacker=[ [0.0 for d in range (0,9)] for k in range (0,len(X)) ]
    test_stacker=[[0.0 for d in range (0,9)] for k in range (0,len(X_test))]
    #label_stacker=[0 for k in range (0,len(X))]
    #split trainingg
    x1,x2,x3,x4,x5,in1,in2,in3,in4,in5=split_array_in_5(X, SEED)
    y1,y2,y3,y4,y5,iny1,iny2,iny3,iny4,iny5=split_array_in_5(y, SEED)         
    #create target variable        
    mean_log = 0.0
    for i in range(0,number_of_folds):
        X_cv,y_cv,indcv,X_train,y_train=cving(x1, x2, x3, x4,x5, y1 ,y2, y3, y4, y5,in1, in2, in3, in4 ,in5, i)
        print (" train size: %d. test size: %d, cols: %d " % (len(X_train) ,len(X_cv) ,len(X_train[0]) ))

        """ model XGBOOST classifier"""             
        xgmat = xgb.DMatrix( csr_matrix(X_train), label=y_train, missing =-999.0  )
        bst = xgb.train( param.items(), xgmat, num_round );
        xgmat_cv = xgb.DMatrix( csr_matrix(X_cv), missing =-999.0)
        preds =bst.predict( xgmat_cv ).reshape( len(X_cv), 9).tolist()
        scalepreds(preds)
        """now model scikit classifier"""          
        model.fit(X_train, y_train) 
        predsextra = model.predict_proba(X_cv)
        scalepreds(predsextra) 
        for pr in range (0,len(preds)):  
            for d in range (0,9):            
                preds[pr][d]=preds[pr][d]*0.8 +  predsextra[pr][d]*0.2            
        # compute Loglikelihood metric for this CV fold
        loglike = logloss_metric( preds,y_cv)
        print "size train: %d size cv: %d Loglikelihood (fold %d/%d): %f" % (len(X_train), len(X_cv), i + 1, number_of_folds, loglike)
     
        mean_log += loglike
        #save the results
        no=0
        for real_index in indcv:
            for d in range (0,9):
                 train_stacker[real_index][d]=(preds[no][d])
            no+=1
   
    if (number_of_folds)>0:
        mean_log/=number_of_folds
        print (" Average M loglikelihood: %f" % (mean_log) )

    xgmat = xgb.DMatrix( csr_matrix(X), label=y, missing =-999.0  )
    bst = xgb.train( param.items(), xgmat, num_round );
    xgmat_cv = xgb.DMatrix(csr_matrix(X_test), missing =-999.0)
    preds =bst.predict( xgmat_cv ).reshape( len(X_test), 9 ).tolist() 
    scalepreds(preds)  
    
    #predicting for test
    model.fit(X, y)
    predsextra = model.predict_proba(X_test) 
    scalepreds(predsextra) 
    
    for pr in range (0,len(preds)):  
        for d in range (0,9):            
            test_stacker[pr][d]=preds[pr][d]*0.8 +  predsextra[pr][d]*0.2  
            

    # === Predictions === #
    print (" printing datasets ")
    printfilewithtarget(train_stacker, outset + "train")
    printfilewithtarget(test_stacker,  outset + "test")
    
    print("Write results...")
    output_file = "submission_"+str( (mean_log ))+".csv"
    print("Writing submission to %s" % output_file)
    f = open(output_file, "w")   
    f.write("Id")# the header
    for b in range (1,10):
         f.write("," + str("Prediction" + str(b) ) )
    f.write("\n")    
    for g in range(0, len(test_stacker))  :
      f.write("%s" % ((ids[g])))
      for prediction in test_stacker[g]:
         f.write(",%f" % (prediction))    
      f.write("\n")
    f.close()
    print("Done.")           
        

    
    
if __name__=="__main__":
  main()