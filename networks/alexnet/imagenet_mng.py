#!/usr/bin/python3
import sys
#sys.path.append('../../convnets-keras/')
sys.path.append('convnets-keras/') #added in order to work with non-targeted
from convnetskeras.convnets import preprocess_image_batch, preprocess_in_values
from tqdm import tqdm
import os
import numpy as np

from keras import backend as K
from keras.optimizers import SGD

class manager_imagenet():
    def __init__(self, pathLabels, pathImgs, cache=False):
        self.pathLabels = pathLabels
        self.pathImgs = pathImgs
        self.listDirs = os.listdir(self.pathImgs)
        self.cache = cache
        self.dictLabels = {}
        self.dictOfImages = {}
        self.listImgsClassLabel = {}
        self._getLabelsImagesFromFile()
        if self.cache:
            self._getFullListImagesFromDirectory()

    def _getLabelsImagesFromFile(self):
        fd = open(self.pathLabels, "r")
        for elem in fd:
            elem = elem.rstrip()
            clas = elem.split('\t')[0]
            label = elem.split('\t')[1]
            if clas not in self.dictLabels:
                self.dictLabels[clas] = label
        fd.close()
        return

    def _getFullListImagesFromDirectory(self):
        i = 0
        list_path = []
        for elem in self.listDirs:
            list_path.append(self.pathImgs + elem + "/" + os.listdir(self.pathImgs + elem)[0])
            img = preprocess_image_batch(list_path, img_size=(256,256), crop_size=(227,227), color_mode="rgb")
            self.listImgsClassLabel[i] = img
            self.dictOfImages[i] = elem
            i+=1
            list_path.clear()
        return None

    def _readImageFormDirectoryNoCache(self, num):
        list_path = []
        elem = self.listDirs[num]
        list_path.append(self.pathImgs + elem + "/" + os.listdir(self.pathImgs + elem)[0])
        img = preprocess_image_batch(list_path, img_size=(256,256), crop_size=(227,227), color_mode="rgb")
        self.listImgsClassLabel[num] = img
        self.dictOfImages[num] = elem
        return img

    def _validNum(self, num):
        return num >= 0 and num < len(self.listDirs)

    def getImgByNum(self, num):
        if self._validNum(num):
            if self.cache:
                return self.listImgsClassLabel[num]
            else:
                return self._readImageFormDirectoryNoCache(num)
        else:
            print("Error: the parameter 'number' is out bound of range: [0 ... 1000]")
            return None

    def preprocessing(self, imgs, num_images):
        preprocess_in_values(imgs, num_images) #side effect!!

    def getListNums(self):
        return list(self.dictOfImages.keys())

    def getClassByNum(self, num):
        if self._validNum(num):
            if self.cache or (num in list(self.dictOfImages.keys())) :
                return self.dictOfImages[num]
            else:
                self._readImageFormDirectoryNoCache(num)
                return self.dictOfImages[num]
        else:
            print("Error: the parameter 'number' is out bound of range: [0 ... 1000]")
            return None

    def getNumByClass(self, cls):
        if self.dictLabels[cls]:
            return int(list(self.dictLabels.keys()).index(cls))
        else:
            print("Error: no num associated with passed class")
            return None

    def getLabelByClass(self, cls):
        if self.dictLabels[cls]:
            return self.dictLabels[cls]
        else:
            print("Error: no image of this class has been read")
            return None

    def clear():
        self.dictOfImages.clear()
        self.listImgsClassLabel.clear()
        self.dictLabels.clear()
        if self.cache:
            self.cache = False
        return None
