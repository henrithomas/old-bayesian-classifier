#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 21:39:58 2018

@author: henrithomas
"""

import numpy as np
stringData = np.genfromtxt('/Users/henrithomas/Desktop/SpamInstances.txt',dtype = str)
#preprocess
def preprocessNBC(data):
    print('Preprocessing...')
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
    print('Preprocessing complete.')
    return (spam, ham)

Spam, Ham = preprocessNBC(stringData)
del stringData
print('Saving data...')
np.savetxt('Ham.txt',Ham)
np.savetxt('Spam.txt',Spam)
del Ham
del Spam
print('Preprocessing complete.')