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

import numpy as np
import sys
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.utils import shuffle
sys.path.append("../../xgboost/wrapper")
import xgboost as xgb


#bestensemble of 6 models

num_round=11500
lr=0.005
max_de=5
subsam=0.4
colsample_bytree=0.3
gamma =0.001
min_child_weight=0.05
seed=1
nesti=10
thre=25

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


def load(name):
    print("start reading file ")
    
    wfile=open(name , "r")
    
    line=wfile.readline().replace("\n","")
    splits=line.split(",")
    datalen=len(splits)
    wfile.close()
    if datalen==9:
        X = np.loadtxt(open( name), delimiter=',',usecols=range(0, datalen), skiprows=0)     
    else :
        X = np.loadtxt(open( name), delimiter=',',usecols=range(1, 10), skiprows=1)         
    print("done") 
    return np.array(X) 

        
def scalepreds(prs):
    
    for i in range  (0, len(prs)):
        suum=0.0
        for j in range (0,9):
            suum+=prs[i][j]
        for j in range (0,9):
            prs[i][j]/=suum 

    
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
    


    
def bagged_set(X,y, size, seed, model, estimators, xt):

   baggedpred=[ [0.0 for d in range (0,9)] for d in range(0, len(xt))]
   for i in range (0, estimators):
       
        X_t,y_c=shuffle(X,y, random_state=seed + i)
        
        
        xgmat = xgb.DMatrix( X_t, label=y_c, missing =-999.0  )
        param['seed']=seed + i
        bst = xgb.train( param.items(), xgmat, num_round );
        xgmat_cv = xgb.DMatrix( xt, missing =-999.0)
        preds =bst.predict( xgmat_cv ).reshape( len(xt), 9).tolist() 
        
        scalepreds(preds)
        
        model.set_params(random_state=seed + i).fit(X_t,y_c)
        predsextra=model.predict_proba(xt)
        scalepreds(predsextra)
        
        for p in range (0,len(preds)):
            for g in range (0,9):
                preds[p][g]= preds[p][g]*0.6 +predsextra[p][g]*0.40    
                

        for j in range (0, len(xt)):
            for g in range (0,9):            
                baggedpred[j][g]+=preds[j][g]
                
   for j in range (0, len(baggedpred)):
            for g in range (0,9):       
                baggedpred[j][g]/=float(estimators)
                
   return baggedpred    
            
def main():

    y= loadlastcolumn("trainLabels.csv")
    ids=loadfirstcolumn("sampleSubmission.csv")
    trainini_file= ["fulllinetrain.csv","2gramtrain.csv","Gert282_258extra115ktrain.csv","Gert202xtra115ktrain.csv","Gert282xtra115ktrain.csv" ,"pred_45c.csv","4gramtrain.csv"]
    testini_file = ["fulllinetest.csv","2gramtest.csv","Gert282_258extra115ktest.csv","Gert202xtra115ktest.csv","Gert282xtra115ktest.csv","subm_45c.csv","4gramtest.csv"]             
    
    model= ExtraTreesClassifier(n_estimators=1000, criterion='entropy', max_depth=20, min_samples_split=2,min_samples_leaf=1, max_features=0.8,n_jobs=25, random_state=3) 

    print (" #####################loading ensemble 1##########################")
    X1=load(trainini_file[0])
    print ("train samples: %d columns: %d " % (len(X1) , len(X1[0])))
    X_test1=load(testini_file[0] )
    print ("train samples: %d columns: %d" % (len(X_test1) , len(X_test1[0])))
    for t in range(1,len(trainini_file)):
        Xini1=load(trainini_file[t] )
        print ("train samples: %d columns: %d " % (len(Xini1) , len(Xini1[0])))
        X_testini1=load(testini_file[t])
        print ("train samples: %d columns: %d" % (len(X_testini1) , len(X_testini1[0])))  
        X1=np.column_stack((X1,Xini1))
        X_test1=np.column_stack((X_test1,X_testini1)) 
        print ("train after merge samples: %d columns: %d" % (len(X1) , len(X1[0])))
        print ("train  after merge samples: %d columns: %d" % (len(X_test1) , len(X_test1[0])))  
                    
        
    number_of_folds=0 # repeat the CV procedure 10 times to get more precise results
    print("------------Start cross validation, times: %d-----------" % (number_of_folds))    


    # === Predictions === #
    # When making predictions, retrain the model on the whole training set
    print("making actual model---")     
    preds1=bagged_set(X1,y, len(X1), 3, model, nesti, X_test1)  

    
    print("Write results...")
    output_file = "final_submission.csv"
    print("Writing submission to %s" % output_file)
    f = open(output_file, "w")   
    f.write("Id")# the header
    for b in range (1,10):
         f.write("," + str("Prediction" + str(b) ) )
    f.write("\n")    
    for g in range(0, len(preds1))  :
      f.write("%s" % ((ids[g])))
      for prediction in preds1[g]:
         f.write(",%f" % (prediction))    
      f.write("\n")
    f.close()
    print("Done.")     
    
    
    
if __name__=="__main__":
  main()