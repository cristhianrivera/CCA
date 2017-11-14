#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 16:28:51 2017

@author: Cristhian
"""
import numpy as np
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA




def read_words(filename):
  wordEnVectors = []
  wordForVectors = []
  fileObject = open(filename, 'r')
  for lineNum, line in enumerate(fileObject):
    line = line.decode('utf-8').strip().lower()
    wordEn = line.split(' ||| ')[0]
    wordFor = line.split(' ||| ')[1]
    wordEnVectors.append(wordEn)
    wordForVectors.append(wordFor)
  return wordEnVectors,wordForVectors

def read_word_vectors_cca(filename):
  wordVectors = []
  fileObject = open(filename, 'r')
  for lineNum, line in enumerate(fileObject): #read vectors
      wordVectors.append(line.split()[1:]) 
  wordVectors = normalize(wordVectors)
  return wordVectors


subsetEnWords, subsetForWords = read_words('parallel.fwdxbwd-dict.de-en')

origEnVecs = read_word_vectors_cca('en-sample.txt')
origForeignVecs = read_word_vectors_cca('de-sample.txt')

subsetEnVecs = read_word_vectors_cca('out_new_aligned.txt')
subsetForeignVecs = read_word_vectors_cca('out_orig_subset.txt')

Or_subsetEnVecs = deepcopy(subsetEnVecs)
Or_subsetForeignVecs= deepcopy(subsetForeignVecs)

#-------------------------------sklearn CCA------------------------------------ 
cca = CCA(n_components=300, scale = False, max_iter = 100000 )

projected_subsetEnVecs,projected_subsetForVecs = cca.fit_transform(
       subsetEnVecs, subsetForeignVecs)

cca.score(subsetEnVecs, subsetForeignVecs)


projectedEnVecs, projectedForVecs = cca.transform(origEnVecs, origForeignVecs)



#-------------------------PCA for visualization--------------------------------


#original vectors
pcaO = PCA(n_components=2, svd_solver='arpack')
pcaO.fit(Or_subsetEnVecs)
original_En_pca = pcaO.transform(Or_subsetEnVecs)
print(pcaO.explained_variance_ratio_)  

pcaO.fit(Or_subsetForeignVecs)
original_Fo_pca = pcaO.transform(Or_subsetForeignVecs)
print(pcaO.explained_variance_ratio_)  



#projected vectors
pcaP = PCA(n_components=2, svd_solver='arpack')
pcaP.fit(projected_subsetEnVecs)
projected_En_pca = pcaP.transform(projected_subsetEnVecs)
print(pcaP.explained_variance_ratio_)  

pcaP.fit(projected_subsetForVecs)
projected_For_pca = pcaP.transform(projected_subsetForVecs)
print(pcaP.explained_variance_ratio_)  


#-------------------------Plot PCA for both VSM--------------------------------

matplotlib.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

ax[0].plot(original_En_pca[0:25,0], original_En_pca[0:25,1], 'o')
ax[0].set_title('VSM Original English' )
u = -1
for i,j in zip(original_En_pca[0:25,0],original_En_pca[0:25,1]):
    u = u + 1 
    ax[0].annotate(subsetEnWords[u],xy = (i,j))

u = 0
ax[1].plot(projected_En_pca[0:25,0], projected_En_pca[0:25,1], 'o')
ax[1].set_title('VSM transformed English')
u = -1
for i,j in zip(projected_En_pca[0:25,0], projected_En_pca[0:25,1]):
    u = u + 1 
    ax[1].annotate(subsetEnWords[u],xy = (i,j))
plt.show()



fig, ax = plt.subplots(1, 2, figsize=(15, 5))

ax[0].plot(original_Fo_pca[0:25,0], original_Fo_pca[0:25,1], 'o')
ax[0].set_title('VSM Original Foreign' )
u = -1
for i,j in zip(original_Fo_pca[0:25,0],original_Fo_pca[0:25,1]):
    u = u + 1 
    ax[0].annotate(subsetForWords[u],xy = (i,j))

u = 0
ax[1].plot(projected_For_pca[0:25,0], projected_For_pca[0:25,1], 'o')
ax[1].set_title('VSM transformed Foreign')
u = -1
for i,j in zip(projected_For_pca[0:25,0], projected_For_pca[0:25,1]):
    u = u + 1 
    ax[1].annotate(subsetForWords[u],xy = (i,j))
plt.show()


