# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 21:34:32 2020

@author: alexi
"""
from sklearn import datasets, neighbors 
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from pandas import DataFrame
import numpy as np
from sklearn import cluster
from math import sqrt
from sklearn.cluster import KMeans



white = pd.read_csv("winequality-white.csv",  sep=";")
red = pd.read_csv("winequality-red.csv",  sep=";")


#######     Clustering visualisation for white wine #########

#données réels
whiteQual=white.quality
whiteQualBin = []
qual, pasQual = 0 , 0
for i in range (len(whiteQual)):
    if whiteQual[i] < 6:
        whiteQualBin.append(1)
        pasQual = pasQual+1
    else:
        whiteQualBin.append(0)
        qual=qual+1
whiteQualBin = DataFrame(whiteQualBin, columns=['Quality'])    



#2D
pca = PCA(n_components=2)
PCA_white = pca .fit_transform(white)
df_white_PCA = pd.DataFrame (data = PCA_white, columns = ["PC1","PC2"])
whiteQual=whiteQual

kmeans = KMeans( n_clusters=2 , n_init=5, max_iter =300, random_state=0).fit(white)
    
prediction = kmeans.predict(white)  
predictionList=prediction.tolist()
prediction = DataFrame(predictionList,columns=['Prediction'])

for lab , col in zip((0, 1),
                     ('red', 'blue')):
    plt.scatter(PCA_white[prediction.Prediction==lab, 0],
                PCA_white[prediction.Prediction==lab, 1],
                label=lab,
                c=col)

plt.title ("White wine quality prediction")
plt.legend(loc='lower center')
plt.tight_layout()
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')
plt.show()

#3D

from mpl_toolkits.mplot3d import Axes3D

pca = PCA(n_components=3)
PCA_white = pca .fit_transform(white)
fig=plt.figure()
ax=Axes3D(fig, rect=(0,0,2,2), azim=60, elev=30)
ax.set_xlim3d(-150,200)
ax.set_ylim3d(0,50)
ax.set_zlim3d(-25,20)
for lab , col in zip((0, 1, 2),
                     ('red', 'blue', 'green')):
    ax.scatter(PCA_white[prediction.Prediction==lab, 0],
                PCA_white[prediction.Prediction==lab, 1],
                PCA_white[prediction.Prediction==lab, 2],
                label=lab,
                c=col)
plt.title ("White wine prediction 3D")  
plt.show()


#vrais donnés en 3D:
fig=plt.figure()
ax=Axes3D(fig, rect=(0,0,2,2), azim=-30, elev=30)
ax.set_xlim3d(0,50)
ax.set_ylim3d(-25,20)
ax.set_zlim3d(-150,200)
for lab , col in zip((0, 1),
                     ('blue', 'red', 'green', 'yellow', 'pink')):
    ax.scatter(PCA_white[whiteQualBin.Quality==lab, 1],
                PCA_white[whiteQualBin.Quality==lab, 2],
                PCA_white[whiteQualBin.Quality==lab, 0],
                label=lab,
                c=col)
    
plt.title ("White wine real data 3D") 
plt.show()


plt.figure()
for lab , col in zip((0, 1),
                     ('red', 'blue', 'green', 'yellow', 'pink')):
    plt.scatter(PCA_white[whiteQualBin.Quality==lab, 0],
                PCA_white[whiteQualBin.Quality==lab, 1],
                label=lab,
                c=col)
plt.title ("White wine real data")  
plt.legend(loc='lower center')
plt.tight_layout()
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')
plt.show()
plt.figure()

#vérification des résultats
a=0
for i in range (len(predictionList)):
    b = predictionList[i]
    c = whiteQualBin.Quality[i]
    if b == c:
        a = a+1
    
print("The clustering precision with the K-mean algorithm for white wine is of", ((a/len(predictionList))*100) , " %")

#######     Clustering visualisation for red wine #########

#Données réels: 

#on considére un vin de qualité si sa note > ou = 6
redQual=red.quality
redQualBin = []
qual, pasQual = 0 , 0
for i in range (len(redQual)):
    if redQual[i] < 6:
        redQualBin.append(1)
        pasQual = pasQual+1
    else:
        redQualBin.append(0)
        qual=qual+1
redQualBin = DataFrame(redQualBin, columns=['Quality'])    

pca = PCA(n_components=2)
PCA_red = pca .fit_transform(red)
     

plt.figure()
for lab , col in zip((0, 1),
                     ('red', 'blue', 'green', 'yellow', 'pink')):
    plt.scatter(PCA_red[redQualBin.Quality==lab, 0],
                PCA_red[redQualBin.Quality==lab, 1],
                label=lab,
                c=col)
plt.title ("Red wine real data")  
plt.legend(loc='lower center')
plt.tight_layout()
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')
plt.show()
plt.figure()


pca = PCA(n_components=3)
PCA_red = pca .fit_transform(red)
fig=plt.figure()
ax=Axes3D(fig, rect=(0,0,2,2), azim=60, elev=30)
ax.set_xlim3d(-35,40)
ax.set_ylim3d(-4, 9)
ax.set_zlim3d(-50,80)

for lab , col in zip((0, 1),
                     ('blue', 'red', 'green', 'yellow', 'pink')):
    ax.scatter(PCA_red[redQualBin.Quality==lab, 1],
                PCA_red[redQualBin.Quality==lab, 2],
                PCA_red[redQualBin.Quality==lab, 0],
                label=lab,
                c=col)
    
plt.title ("Red wine real data 3D") 
plt.show()


#Données prédits
#2D
pca = PCA(n_components=2)
PCA_red = pca .fit_transform(red)
kmeans = KMeans( n_clusters=2 , n_init=5, max_iter =300, random_state=0).fit(PCA_red)
prediction = kmeans.predict(PCA_red)  
predictionList = prediction.tolist()
prediction = DataFrame(prediction,columns=['Prediction'])

plt.figure()
for lab , col in zip((0, 1),
                     ('blue', 'red', 'green', 'yellow', 'pink')):
    plt.scatter(PCA_red[prediction.Prediction==lab, 0],
                PCA_red[prediction.Prediction==lab, 1],
                label=lab,
                c=col)
plt.title("Red wine clustering")   
plt.legend(loc='lower center')
plt.tight_layout()
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')
plt.show()
plt.figure()

#3D
pca = PCA(n_components=3)
PCA_red = pca .fit_transform(red)
fig=plt.figure()
ax=Axes3D(fig, rect=(0,0,2,2), azim=60, elev=30)
ax.set_xlim3d(-35,40)
ax.set_ylim3d(-4, 9)
ax.set_zlim3d(-50,80)

for lab , col in zip((0, 1),
                     ('blue', 'red', 'green', 'yellow', 'pink')):
    ax.scatter(PCA_red[prediction.Prediction==lab, 1],
                PCA_red[prediction.Prediction==lab, 2],
                PCA_red[prediction.Prediction==lab, 0],
                label=lab,
                c=col)
plt.title("Red wine clustering")    
plt.show()

#vérification des résultats
a=0
for i in range (len(predictionList)):
    b = predictionList[i]
    c = redQualBin.Quality[i]
    if b == c:
        a = a+1
    
print("The clustering precision with the K-mean algorithm for red wine is of", ((a/len(predictionList))*100) , " %")
        

from scipy.stats import chi2_contingency
def cramers(confusion_matrix):

    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().iloc[4]
    phi2 = chi2/n
    r = confusion_matrix.shape[0]
    k = confusion_matrix.shape[1]
    if 0> (phi2 - ((k-1)*(r-1))/(n-1)):
        phi2corr = 0
    else:
        phi2corr = phi2 - ((k-1)*(r-1))/(n-1)
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    if ((kcorr-1)>(rcorr-1)):
        return np.sqrt(abs(phi2corr/(rcorr-1)))
    else: 
        return np.sqrt(abs(phi2corr/(kcorr-1)))

wine = pd.concat((white,red))
wineType=[]
for i in range (len(red)):
    wineType.append(0)
for i in range (len(white)):
    wineType.append(1)

wine['wineType']= wineType



for k in wine.columns:
    for l in wine.columns:
        print("Le coefficient de Cramers entre ", k, " et ", l , "est: " 
              , cramers(pd.crosstab(wine[l], wine[k])))
