import pandas as pd 
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle 
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import cv2
import random
import keras
from keras.models import Sequential
from keras.layers import Convolution2D,Flatten,Dense
from keras.optimizers import Adam


def spliter(pathname):
   return pathname.split("\\")[-1]



def importdata(path):
    columns = ["centre","left","right","a1","a2","a3","a4"]
    path = os.path.join(path,"driving_log.csv")
    data = pd.read_csv(path,names= columns)
    data["centre"] = data["centre"].apply(spliter)
    # print(data.head())
    return data
    
    
def balancedata(data,display = True):
    nbins = 31
    sampleperbin = 500
    
    hist,bins =np.histogram(data["a1"],nbins) 
    if display:
        center = (bins[:-1]+bins[1:])*0.5
        # plt.hist(data["a1"],bins)
        
        print(center)
        plt.bar(center,hist,width=0.06)
        plt.plot((-1,1),(sampleperbin,sampleperbin))
        plt.show()
    
    removelist = []
    for j in range(nbins):
        bindatalist = []
        for i in range(len(data['a1'])):
            if data['a1'][i]>=bins[j] and data['a1'][i]<= bins[j+1]:
                bindatalist.append(i)
        bindatalist = shuffle(bindatalist)
        bindatalist = bindatalist[sampleperbin:]
        removelist.extend(bindatalist)
    print("removed imagee :",len(removelist))    
    data.drop(data.index[removelist],inplace = True)
    print("remain", len(data))
    if display:
        hist,bins =np.histogram(data["a1"],nbins) 
        center = (bins[:-1]+bins[1:])*0.5
        # plt.hist(data["a1"],bins)
        
        # print(center)
        plt.bar(center,hist,width=0.06)
        plt.plot((-1,1),(sampleperbin,sampleperbin))
        plt.show()
    return data
     
def loaddata(path,data):
    imagepath = []
    stearing = []
    for i in range (len(data)):
        indexeddata = data.iloc[i]
        imagepath.append(os.path.join(path,'IMG',indexeddata[0]))
        stearing.append(indexeddata[3])
    imagepath = np.array(imagepath)
    stearing = np.array(stearing)
    return imagepath, stearing

def augmentimage(imgpath,stearing):
    img = mpimg.imread(imgpath)
    if np.random.rand()<0.5:
        
    #translation left right move  
        pan = iaa.Affine(translate_percent={'x':(-0.1,0.1),'y':(-0.1,0.1)})
        img = pan.augment_image(img)
    
    if np.random.rand()<0.5:
    #zoom
        zoom = iaa.Affine(scale= (1,1.2))
        img = zoom.augment_image(img)
    if np.random.rand()<0.5:
    # brightness 
        brightness = iaa.Multiply((0.6,1.2))
        img = brightness.augment_image(img)
    if np.random.rand()<0.5:
    #flip
        img = cv2.flip(img,1)
        stearing = -stearing
        
    return img , stearing

def preprocessing(img):
    
    img = img[60:135,:,:]
    img = cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
    img =cv2.GaussianBlur(img,(3,3),0)#iteration
    img= cv2.resize(img,(200,66))
    img =img/255
    return img
    
def batchgen(imagepath , stearingm,batchsize ,flag):
    while True:
        imgbatch = []
        stearingbatch = []
        for i in range(batchsize):
            index = random.randint(0,len(imagepath)-1)
            if flag:
                img ,stearing =augmentimage(imagepath[index],stearingm[index])
            else:
                img = mpimg.imread(imagepath[index])
                stearing = stearingm[index]
                
            img=preprocessing(img)
            imgbatch.append(img)
            stearingbatch.append(stearing)
        yield(np.asarray(imgbatch),np.asarray(stearingbatch))
        
def Model():
    model = Sequential()
    
    model.add(Convolution2D(24,(5,5),(2,2),input_shape=(66,200,3)))
    model.add(Convolution2D(36,(5,5),(2,2),activation='elu'))
    model.add(Convolution2D(48,(5,5),(2,2),activation='elu'))
    model.add(Convolution2D(64,(3,3),activation='elu'))
    model.add(Convolution2D(64,(3,3),activation='elu'))
    model.add(Flatten())
    model.add(Dense(100,activation ='elu'))
    model.add(Dense(50,activation ='elu'))
    model.add(Dense(10,activation ='elu'))
    model.add(Dense(1))
    
    model.compile(Adam(0.0001),loss = 'mse')
    
    return model