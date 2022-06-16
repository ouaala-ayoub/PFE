from PIL import Image
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
import os


def showImage(path):
    img = Image.open(path)
    img.show()

def getDataSetNames(path):    
    os.chdir(path)
    #print(os.getcwd())
    # list des noms des images
    dataSet = []

    files = os.scandir(path)
    for file in files:
        if file.name.endswith('.jpg') or file.name.endswith('.png'):
        # ajouter seulement les images
            dataSet.append(file.name)

    return dataSet

def getFeatureHistogram(path):
    img = Image.open(path)
    img = img.convert('RGB')
    return img.histogram()

def getFeatureMean(path):
    img = Image.open(path)
    img = img.convert('RGB')
    a = np.array(img)
    w, h, c = a.shape
    a1 = a.reshape(w * h, c)
    mr, mg, mb = np.mean(a1[:, 0]), np.mean(a1[:, 1]), np.mean(a1[:, 2])
    return [mr, mg, mb]



def getStandardDev(path):
    img = Image.open(path)
    img = img.convert('RGB')
    a = np.array(img)
    r, g, b = img.split()
    r, g, b = np.array(r), np.array(g), np.array(b)
    stdr, stdg, stdb = np.std(r), np.std(g), np.std(b)

    return [stdr, stdg, stdb]



def getFeatures(path):
    #h = getFeatureHistogram(path)
    m = getFeatureMean(path)
    s = getStandardDev(path)
    features = m + s
    return features

def view_cluster(cluster, name = 'default'):
    plt.figure(name, figsize = (15, 15))
    # avoir la liste des images
    files = cluster
    # only allow up to 30 images to be shown at a time
    if len(files) > 30:
        print("Clipping cluster size from", len(files), "to 30")
        files = files[:29]
    # plot each image in the cluster
    for index, file in enumerate(files):
        plt.subplot(10, 10, index+1)
        img = Image.open(file)
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')
    plt.show()    
    return len(cluster)


#a donner le path ou il y a les images a grouper manuelement et a changer apres pas l appel de la base des donnes
#exemple ===> data = getDataSetNames(r'C:\Users\Nomdutilisateur\Desktop\py\code\dataSet')
# le 'r' avant le path est obligatoire

data = getDataSetNames(r"C:\Users\ouaal\Desktop\pyyyy\dataSetFlowers")
#print(data)

#remplir les features
features = {}
for path in data:
    feature = getFeatures(path)
    features[path] = feature

#choisir n nombre des groupes 
n = 5
kMeans = KMeans(n_clusters=n)
v = np.array(list(features.values()))
print(v.shape)
#print(v)
kMeans.fit(v)
centroids = kMeans.cluster_centers_
labels = kMeans.labels_

#retourner les groupes resultats d algorithme k_means
clusters = []
for i in range(n):
    cluster = []
    for j in range(len(data)): 
        if labels[j] == i:
            cluster.append(data[j])
    clusters.append(cluster)   

#afficher les resultats
for i in range(n):
    view_cluster(clusters[i])

