from PIL import Image
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os

'''importation du package PIL et matplotlib 
commande pour installer PIL : pip install pillow
commande pour installer matplotlib : python -m pip install -U pip
                                     python -m pip install -U matplotlib
'''

'''Types d images : P, PA, L, LA, RBG, RBGA'''

'''retourner le code hexadecimal de chaque couleur'''

def getRed(redVal):
    return '#%02x%02x%02x' % (redVal, 0, 0)

def getGreen(greenVal):
    return '#%02x%02x%02x' % (0, greenVal, 0) 

def getBlue(blueVal):
    return '#%02x%02x%02x' % (0, 0, blueVal)

'''prendre les informations d une image comme la format et la taille en pixels 
et le mode (rbg, p, l, ...)'''

def getImageInfo(path):
    img = Image.open(path)
    print(img.format, img.size, img.mode)
    ''' img.show() 
    pour afficher un l image dans une fenetre independente'''

'''afficher le histograme de chaque couleur separament en affichant les barres avec la couleur correspondante'''

def getImageHistogram(path, red = True, green = True, blue = True):

    img = Image.open(path)
    img = img.convert('RGB')

    #print(img.histogram())

    """histogram = img.histogram()  
    r = histogram[0:256]    
    g = histogram[256:512]
    b = histogram[512:768]
    """ 

    r, g, b = img.split()
    r, g, b = r.histogram(), g.histogram(), b.histogram()
    
    

    if red:
        plt.figure(0)
        for i in range(0, 256):
            plt.bar(i, r[i], color = getRed(i), edgecolor=getRed(i), alpha=0.3)

    if green:
        plt.figure(1)
        for i in range(0, 256):
            plt.bar(i, g[i], color = getGreen(i), edgecolor=getGreen(i), alpha=0.3)

    if blue:
        plt.figure(2)
        for i in range(0, 256):
            plt.bar(i, b[i], color = getBlue(i), edgecolor=getBlue(i), alpha=0.3)

    plt.show()
    return img.histogram()

"""Application de K_means"""

def cluster(n_clusters, hist):
    kMeans = KMeans(n_clusters = 3)
    kMeans.fit(hist)
    centroids = kMeans.cluster_centers_
    labels = kMeans.labels_

    '''print(hist)
    print(centroids)
    print(labels)'''


    plt.plot(hist[labels == 0,0],hist[labels == 0,1],'r.', label='cluster 1')
    plt.plot(hist[labels == 1,0],hist[labels == 1,1],'b.', label='cluster 2')
    plt.plot(hist[labels == 2,0],hist[labels == 2,1],'g.', label='cluster 3')

    plt.plot(centroids[:,0],centroids[:,1],'mo',markersize=8, label='centroids')

    plt.legend()
    plt.show()

    pass

'''a changer par les images dans la base de donnes'''

def randomGreyPlt():
    digits = load_digits()
    data = digits.data

    data = 255 - data
    print(data[1700])
    np.random.seed(1)
    n = 10
    kMeans = KMeans(n_clusters=n, init='random')
    kMeans.fit(data)
    Z = kMeans.predict(data)

    for i in range(0,n):

        row = np.where(Z==i)[0]  # row in Z for elements of cluster i
        num = row.shape[0]       #  number of elements for each cluster
        r = np.floor(num/10.)    # number of rows in the figure of the cluster 
        r = int(r)
        
        print("cluster "+str(i))
        print(str(num)+" elements")

        plt.figure(figsize=(10,10))
        for k in range(0, num):
            plt.subplot(r+1, 10, k+1)
            image = data[row[k], ]
            image = image.reshape(8, 8)
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.show()

def reduceColors(n_clusters, path):

    img = Image.open(path)
    img = img.convert('RGB')
    a  = np.asarray(img, dtype=np.float32)/255
    x, y, z = a.shape
    a1 = a.reshape(x*y, z)

    n = n_clusters
    kMeans = KMeans(n_clusters=n)
    kMeans.fit(a1)
    centroids = kMeans.cluster_centers_
    labels = kMeans.labels_
    a2 = centroids[labels]
    a3 = a2.reshape(x,y,z)
    a4 = np.floor(a3*255)
    a5 = a4.astype(np.uint8)
    I1 = Image.fromarray(a5)
    I1.save("test.jpg")

    plt.figure(figsize=(12,12))
    plt.imshow(a3)
    plt.axis('off')
    plt.show()

def getDarkColorArea(path, totalArea, show = True):

    img = Image.open(path)
    img = img.convert('L')
    a = np.asarray(img, float)
    x, y = a.shape
    a1 = a.reshape(x * y, 1)

    n = 3
    kMeans = KMeans(n_clusters=n)
    kMeans.fit(a1)
    centroids = kMeans.cluster_centers_
    labels = kMeans.labels_
    error = kMeans.inertia_
    #print(error)
    a2 = centroids[labels]
    
    a3 = a2.reshape(x, y)
    a4 = (a3 - np.min(a3))/(np.max(a3)-np.min(a3))*255
    a5 = a4.astype(np.uint8)
    I2 = Image.fromarray(a5)
    w, h =I2.size
    colors = I2.getcolors(w * h)
    
    i = datetime.datetime.now()
    imageName = 'pic' + str(i) + '.PNG'
    imageName = imageName.replace(':', '.')
    I2.save(imageName)

    if show:
        I2.show()

    return float(totalArea) * float(colors[0][0])/float(w * h)

#reduceColors(3, 'lac.jpg')

x = getDarkColorArea('test1111.PNG', 100, False)
print(x)

