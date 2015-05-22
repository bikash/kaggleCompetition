# -*- coding: utf-8 -*-
"""
Created on Sat Feb 14 23:15:24 2015

@author: marios

Basic script that performs online tfidf of the bytes file
in the malware kaggle competiton and prints 4 gram-bytes

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

from collections import defaultdict
import operator
import gc
"""

Method to detect different bytes values and update the given dictionary

The aim is to create a corpus with all unique words. 

"""   
    
def convertfiletotokenarray(id_to_open, ngram, worddict):

    words_file = defaultdict(lambda: 0)
    fop=open(id_to_open.replace('"','') +'.bytes', 'r') # open the bytes file
    for mline in fop:
       if ngram<=0: # take the whole line as gram
             mline= mline.replace("\n","")
             mline=mline[9:]
             if mline not in words_file: # if word not present in current dictionary...add the term of the word in  worddict
                        words_file[mline]+=1
                        worddict[mline]+=1                
       else :
           # use grams of single bytes
           split=mline.replace("\n","").split(" ") # break by space
           for ng in range(ngram-1,ngram):
               for j in range (ng+1,len(split) ):
                   str_to_pass=""
                   for s in range (0,-ng-1,-1 ): 
                       str_to_pass+=split[j+s]
                   if str_to_pass not in words_file: # if word not present in current dictionary...add the term of the word in  worddict
                        words_file[str_to_pass]+=1
                        worddict[str_to_pass]+=1     
                        
                        
 
"""
based on a dictionary of words and given ngrams, prints teh counts of bytes in the given file .
"""
                        
def convertfiletotokenarraybytetokens(id_to_open, ngram, indexesdict,files):

    words_file = defaultdict(lambda: 0)
    fop=open(id_to_open.replace('"','') +'.bytes', 'r') # open the bytes file
    row_length=0
    total_elements=0    
    for mline in fop:
           row_length+=1             
           split=mline.replace("\n","").split(" ")
           for ng in range(ngram-1,ngram):
               for j in range (ng+1,len(split) ):
                   str_to_pass=""
                   for s in range (0,-ng-1,-1 ): 
                       str_to_pass+=split[j+s]
                   #print(str_to_pass)
                   if str_to_pass in indexesdict : # if word not present in current dictionary...add the term of the word in  worddict
                        words_file[str_to_pass]+=1
                        total_elements+=1

    files.write(str(row_length) + "," + str(total_elements) )
    for word, ind in indexesdict.iteritems(): 
          if  word in words_file:
                  bvalue=float(words_file[word])
                  files.write("," + str(bvalue) )
          else :
                   files.write(",0" )             
    files.write("\n")   
   
"""
method that creates ngrams from bytes and includes 2 steps:

1) scan teh files and the fine the bytes ("words")
2) re-scan files and print new train and test sets with counts' distribution of the bytes' founds via teh scanning process
parameters:

trainfile= the labels' file (becasuse we can get the ids)
testfile= the sample submission file (becasuse we can get the ids for the tets cases)
ngrams= number of ngrams. This considers the number of grands strictly. e.g ngrams=2 considers only ngrams 2 (not 1 grams too)
output_name= prefix for the train and test files

"""
def makesets( trainfile="E:/trainLabels.csv", testfile="E:/sampleSubmission.csv" ,  ngrams=1,   output_name=""  ):
    
    #wordsother=loaddicts("E:/" + "dictspersubject"  , 1000000)
    words = defaultdict(lambda: 0)
    indices = defaultdict(lambda: 0)
    words_test = defaultdict(lambda: 0)
    
    print ("openning: " + str(trainfile) )        
    tr=open (trainfile,'r') # open training labels' file taht has the ids
    tr.readline() # headers        
    train_counter=0
    for line in tr : # for each line in the file
       splits=line.split(",")
       trid=splits[0] # e.g. a key
       convertfiletotokenarray('E:/train/train/' + trid, ngrams, words)
       train_counter+=1
       if train_counter%100==0:
           print ("we are at train : " + str(train_counter)+ " length: " + str(len(words) ) )                
       if len(words)>50000000.0:
           print(" reached maximum number of cases: " + str(len(words)) )
           break
    tr.close()
    print(" finished training tf with total distinct words: " + str(len(words) ) ) 
    
    print("sorting...")
    word_sorted = sorted(words.items(), key=operator.itemgetter(1) , reverse=True) # reverse sort word by count    
    words=None
    gc.collect() # call garbage collector to release some memory

    
    # do the same with test as this time we work with samples and want to make certain that the words exist in both
    print ("openning: " + str(testfile) )
    te=open (testfile,'r') # open training labels' file taht has the ids
    te.readline() # headers       
    test_counter=0
    for line in te : # for each line in the file
       splits=line.split(",")
       teid=splits[0] # e.g. a key
       convertfiletotokenarray('E:/test/test/' + teid, ngrams, words_test)  
       print(str(test_counter) + " case: " + str(splits[0]) )
       test_counter+=1
       if test_counter%100==0:
           print ("we are at test : " + str(test_counter)+ " length: " + str(len(words_test) ) )

       if len(words_test)>50000000.0:
           print(" reached maximum number of cases: " + str(len(words_test))  )               
           break    
    te.close()
    

            
    index=2
    mini_c=0
    thress=40000 # number of most popular ngrams to consider
    for iu in range(0,len(word_sorted)):
                word=word_sorted[iu][0]
                if mini_c>thress:
                    break
                if word in words_test: # if word is also in the test dictionary
                    indices[word]=index
                    index += 1
                    mini_c+=1

    #rest dictionaries
    words_test=None

    word_sorted=None
    gc.collect() # call garbage collector to release some memory
    
     
    print (" max index is: " + str(index) )
    
    #create train set elements

    trs=open("E:/" +output_name + "train.csv", "w")
    print ("openning: " + str(trainfile) )        
    tr=open (trainfile,'r') # open training labels' file taht has the ids
    tr.readline() # headers        
    train_counter=0
    for line in tr : # for each line in the file
       splits=line.split(",")
       trid=splits[0] # e.g. a key
       convertfiletotokenarraybytetokens('E:/train/train/' + trid, ngrams ,indices,trs) # print line in file   

       train_counter+=1
       if train_counter%100==0:
           print ("we are at : " + str(train_counter)   )
    print ("create file with rows: " + str(train_counter) )
    
    tr.close()
    trs.close()
  
    tes=open("E:/" +output_name + "test.csv", "w")   
    print ("openning: " + str(testfile) )    
    te=open (testfile,'r') # open training labels' file taht has the ids
    te.readline() # headers       
    test_counter=0
    for line in te : # for each line in the file
       splits=line.split(",")
       teid=splits[0] # e.g. a key
       convertfiletotokenarraybytetokens('E:/test/test/' + teid, ngrams ,indices,tes) # print line in file    
       test_counter+=1
       if test_counter%100==0:
        print ("we are at test : " + str(test_counter) )
    print ("create file with rows: " + str(test_counter) )

            
    te.close()   
    tes.close()      

                



def main():
    
    
    
    makesets( trainfile="E:/trainLabels.csv", testfile="E:/sampleSubmission.csv" , ngrams=4,  output_name="old4gram"  )


  
   

if __name__=="__main__":
  main()
