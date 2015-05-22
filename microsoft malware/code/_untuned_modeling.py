######################################################
# _untuned_modeling.py
# author: Gert Jacobusse, gert.jacobusse@rogatio.nl
# licence: FreeBSD

"""
Copyright (c) 2015, Gert Jacobusse
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

#first run feature_extraction.py
#then run this file from the same directory

######################################################
# import dependencies

import csv
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.ensemble import GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.metrics import log_loss

######################################################
# list ids and labels

trainids=[]
labels=[]
with open('trainLabels.csv','r') as f:
    r=csv.reader(f)
    r.next() # skip header
    for row in r:
        trainids.append(row[0])
        labels.append(float(row[1]))

testids=[]
with open('sampleSubmission.csv','r') as f:
    r=csv.reader(f)
    r.next()
    for row in r:
        testids.append(row[0])

######################################################
# general functions

def readdata(fname,header=True,selectedcols=None):
    with open(fname,'r') as f:
        r=csv.reader(f)
        names = r.next() if header else None
        if selectedcols:
            assert header==True
            data = [[float(e) for i,e in enumerate(row) if names[i] in selectedcols] for row in r]
            names = [name for name in names if name in selectedcols]
        else:
            data = [[float(e) for e in row] for row in r]
    return data,names

def writedata(data,fname,header=None):
    with open(fname,'w') as f:
        w=csv.writer(f)
        if header:
            w.writerow(header)
        for row in data:
            w.writerow(row)

######################################################
# cross validation

"""
function docv
input: classifier, kfolds object, features, labels, number of data rows
output: holdout-set-predictions for all rows
* run cross validation
"""
def docv(clf,kf,x,y,nrow,nlab=9):
    pred = np.zeros((nrow,nlab))
    for trainidx, testidx in kf:
        clf.fit(x[trainidx],y[trainidx])
        pred[testidx] = clf.predict_proba(x[testidx])    
    return pred

"""
function runcv
input: name of train/ test file, classifier 1 and 2 to be used
output: writes holdout-set-predictions for all rows to file
* run cross validation by calling docv for both classifiers, combine and save results
"""
def runcv(filename,c1,c2):
    y=np.array(labels)
    nrow=len(y)
    x,_=readdata('train_%s'%filename)
    x=np.array(x)
    kf = KFold(nrow,10,shuffle=True)
    p1=docv(c1,kf,x,y,nrow)
    p2=docv(c2,kf,x,y,nrow)
    pcombi=0.667*p1+0.333*p2
    print '%.4f %.4f %.4f'%(log_loss(y,p1),log_loss(y,p2),log_loss(y,pcombi))
    with open('pred_%s'%filename,'w') as f:
        w=csv.writer(f)
        for row in pcombi:
            w.writerow(row)

######################################################
# submit and print feature importance

"""
function writesubm
input: name of train/ test file, classifier 1 and 2 to be used
output: writes testset predictions to file
* train classifiers using all traindata, create testset predictions, combine and save results
"""
def writesubm(filename,c1,c2):
    xtrain,names=readdata('train_%s'%filename)
    xtest,_=readdata('test_%s'%filename)
    c1.fit(xtrain,labels)
    c2.fit(xtrain,labels)
    p1=c1.predict_proba(xtest)
    p2=c2.predict_proba(xtest)
    p=0.667*p1+0.333*p2
    with open('subm_%s'%filename,'w') as f:
        w=csv.writer(f)
        w.writerow(['Id']+['Prediction%d'%num for num in xrange(1,10)])
        for inum,i in enumerate(testids):
            w.writerow([i]+list(p[inum]))

######################################################
# go

if __name__ == '__main__':
    gbm=GradientBoostingClassifier(
                                n_estimators=400, max_features=5)
    xtr=ExtraTreesClassifier(
                                n_estimators=400,max_features=None,
                                min_samples_leaf=2,min_samples_split=3,
                                n_jobs=7)
    for filename in [
            '45c.csv',
            ]:
        print filename
        runcv(filename,gbm,xtr)
        writesubm(filename,gbm,xtr)
        print ''
"""

45c.csv
0.0117 0.0168 0.0101

public LB: 0.008071379
private LB: 0.007615772

"""