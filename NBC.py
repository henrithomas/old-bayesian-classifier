#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 13:22:27 2018

@author: henrithomas
"""
import numpy as np
import math
import matplotlib.pyplot as plt
setSize = 100
iterations = 20 #max is 155
tpROC = np.zeros(iterations)
fpROC = np.zeros(iterations)
Ham = np.genfromtxt('Ham.txt')
Spam = np.genfromtxt('Spam.txt')
def ROCCurveVals(expLab,actualLab,testSize):
    TP = 0
    FP = 0
    N = 0
    P = 0
    for i in range(testSize):
        if actualLab[i] == 1:
            P += 1
        else: 
            N += 1
        if expLab[i] == actualLab[i] and actualLab[i] == 1:
            TP += 1
        elif expLab[i] != actualLab[i] and expLab[i] == 1:
            FP += 1
    return (TP/P, FP/N)

def createLabels(spamP,hamP,testSet):
    #print('Classifying emails...')
    labels = np.zeros(testSet.shape[0])
    probs = np.zeros(2)
    for i in range(testSet.shape[0]):
        probs[0] = np.prod(hamP[np.argwhere(testSet[i] == 1)])
        probs[1] = np.prod(spamP[np.argwhere(testSet[i] == 1)])
        if np.argmax(probs) == 0:
            labels[i] = -1
        else:
            labels[i] = 1
    #print('Classifying complete...')
    return labels
        
def hamAndSpamProb(trainingSet):
    #returns arrays of probabilities for features in the training set
    #print('Constructing probabilities for ham and spam features...')
    hamProb = np.zeros(334)
    spamProb = np.zeros(334)
    hamCount = 0 
    spamCount = 0
    for i in range(trainingSet.shape[0]):
        if trainingSet[i][0] == 1:
            spamCount += 1
            spamProb += trainingSet[i,1:335]
        else:
            hamCount += 1 
            hamProb += trainingSet[i,1:335]
    spamProb *= 1/spamCount
    hamProb *= 1/hamCount
    #print('Probabilities complete.')
    return (spamProb, hamProb)

#preprocess
def preprocessNBC(data):
    #print('Preprocessing...')
    buffer = np.zeros((1,335))
    noHam = True
    noSpam = True
    for i in range(data.shape[0]):
        buffer[0,0] = int(data[i,1])
        buffer[0,1:335] = (np.fromstring(data[i,2],'i1') - 48)
        if buffer[0,0] == -1:
            if noHam == True:
                ham = buffer
                noHam = False
            ham = np.concatenate((ham, buffer),axis = 0)
        else:
            if noSpam == True:
                spam = buffer
                noSpam = False
            spam = np.concatenate((spam,buffer),axis = 0)
    #print('Preprocessing complete.')
    return (spam, ham)


for i in range(iterations):
    subsetSize = int(setSize/2)
    subset = np.concatenate((Ham[np.random.randint(Ham.shape[0], size=subsetSize)],
                             Spam[np.random.randint(Spam.shape[0], size=subsetSize)]),axis = 0)
    cutoff = math.floor(setSize * 0.8)
    np.random.shuffle(subset)
    training = subset[0:cutoff]
    validation = subset[cutoff:subset.shape[0]]
    
    spamProbs, hamProbs = hamAndSpamProb(training)
    
    testLabels = createLabels(spamProbs,hamProbs,validation[:,1:355])
    
    truePositives, falsePositives = ROCCurveVals(testLabels,validation[:,0],setSize - cutoff)
    tpROC[i] = truePositives
    fpROC[i] = falsePositives
    print('TP rate:',round(truePositives,3),'FP rate:',round(falsePositives,3))
    setSize += 100

plt.figure(figsize=(6,6))
plt.xlim(0,1.1)
plt.ylim(0,1.1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Space')
plt.scatter(fpROC, tpROC)
plt.show()

