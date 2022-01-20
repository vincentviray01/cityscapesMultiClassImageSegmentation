import tensorflow as tf
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from skimage.io import imread, imshow
from sklearn.cluster import KMeans
from sklearn.utils import class_weight
import random

np.random.seed(1)

from model import ConvPoolBlockWithoutTranspose, ConvPoolBlockWithTranspose
def getImages(path):
    imagesList = []

    for directory, _, files in os.walk(path):
        for file in files:
            imageFile = f"{directory}/{file}"
            # print(imageFile)
            # imshow(imageFile)
            # plt.show()
            imagesList.append(imageFile)

    return imagesList

def showImage(imageFilePath):
    imshow(imageFilePath)
    # plt.show()

def imagePathsToNumpyArrays(imageFilePaths, image_size=128):
    return list(map(lambda x: (resize(x, output_shape=(image_size,2*image_size))*256).astype('int'), list(map(imread, imageFilePaths))))

def splitImageAndMask(images):
    leftImages = (np.asarray(list(map(lambda x: x[:,:int(images[0].shape[1]/2),:], images)))/255.0).astype("float32")
    rightMasks = np.asarray(list(map(lambda x: x[:,int(images[0].shape[1]/2):,:], images))).astype("int16")

    return leftImages, rightMasks    

def showSampleImagesAndMasks(imagesWithoutMask, imagesWithMask):
    randomIndices = np.random.choice(np.arange(len(imagesWithoutMask)), 9)
    plt.figure(figsize=(100,100))
    for i in range(9):
        plt.subplot(9, 2,(i*2)+1)
        plt.imshow(imagesWithoutMask[randomIndices[i]])

        plt.subplot(9, 2, (i*2)+2)
        plt.imshow(imagesWithMask[randomIndices[i]])

    plt.show()


def getNewLabelsForTrainAndTestMasks(trainMasks, testMasks, num_classes=10):
    imageSize = trainMasks[0].shape[1]
    label_model = KMeans(n_clusters = num_classes)
    dataToFit = trainMasks[:10].reshape(-1, 3)
    label_model.fit(dataToFit)
    trainLabels=[]
    for i in tqdm(range(len(trainMasks))):
        label_class = label_model.predict(trainMasks[i].reshape(-1,3)).reshape(imageSize,imageSize)
        trainLabels.append(label_class)

    testLabels=[]
    for i in tqdm(range(len(testMasks))):
        label_class = label_model.predict(testMasks[i].reshape(-1,3)).reshape(imageSize,imageSize)
        testLabels.append(label_class)

    return label_model.cluster_centers_, trainLabels, testLabels 


def getModel(inputShape = (128, 128, 3), classes=10, verbose=False):
    inputs = tf.keras.Input(shape=(inputShape))

    block1 = ConvPoolBlockWithoutTranspose(3, 32, 32) # input size: 256, output size: 128
    block2 = ConvPoolBlockWithoutTranspose(3, 64, 64) # input size: 128, output size: 64
    block3 = ConvPoolBlockWithoutTranspose(3, 128, 128) # input size: 64, output size: 32
    block4 = ConvPoolBlockWithoutTranspose(3, 256, 256, useMaxPool=False)
    block5 = ConvPoolBlockWithTranspose(3, 128, 128, 128) # input size: 32, output size: 64
    block6 = ConvPoolBlockWithTranspose(3, 64, 64, 64) # input size: 64, output size: 128
    block7 = ConvPoolBlockWithTranspose(3, 32, 32, 32) # input size: 128, output size: 256
    
    c1, c1_pooled = block1(inputs) # c1 size = 256, c1_pooled size = 128
    c2, c2_pooled = block2(c1_pooled) # c2 size = 128, c2_pooled size = 64
    c3, c3_pooled = block3(c2_pooled) # c3 size = 64, c3_pooled size = 32
    c4 = block4(c3_pooled) # c4 size = 32
    c5 = block5(c4, c3) # c5 size = 64 
    c6 = block6(c5, c2) # c6 size = 128
    c7 = block7(c6, c1) # c7 size = 256
    
    outputsPenultimate = tf.keras.layers.Conv2D(classes, (1, 1), padding='same', activation='softmax')(c7)
    outputs = tf.keras.layers.Reshape((inputShape[0]*inputShape[1], classes))(outputsPenultimate)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

    if verbose:
        print(model.summary())

    return model

def printModelLayers(model):
    for layer in model.layers:
        try:
            for layer in layer.layers:
                print(layer)
        except:
            print(layer)

def getCallbacks(patience=500):
    checkpointCallback = tf.keras.callbacks.ModelCheckpoint('seg_model.h5', monitor='loss', verbose=1, save_best_only=True, save_weights_only=True)
    earlyStoppingCallback=tf.keras.callbacks.EarlyStopping(monitor='loss',patience=patience)
    tensorBoardCallback = tf.keras.callbacks.TensorBoard(log_dir='logs')
    callbacksList=[checkpointCallback, earlyStoppingCallback, tensorBoardCallback]
    return callbacksList

def getSampleWeights(trainingImagesWithMaskLabels, num_classes=10, numTrainingImages=2975, image_size=128):
    class_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(np.asarray(trainingImagesWithMaskLabels[:100]).reshape(-1)), y = np.asarray(trainingImagesWithMaskLabels[:100]).reshape(-1))
    class_weights = {x : class_weights[x] for x in range(len(class_weights))}
    _, class_weights_freq = np.unique(np.asarray(trainingImagesWithMaskLabels).reshape(-1), return_counts=True)
    class_weights_freq_normalized = (1/class_weights_freq)/np.sum(1/class_weights_freq)
    class_weights_freq_normalized_temp = class_weights_freq_normalized * (1/np.min(class_weights_freq_normalized))
    class_weights_freq_normalized_final = class_weights_freq_normalized_temp / (1/min(class_weights.values()))

    trainingImagesWithMaskLabelsTemp = np.asarray(trainingImagesWithMaskLabels).reshape(-1, 16384)
    sample_weights = np.random.rand(numTrainingImages, image_size**2)
    for x in range(num_classes):
        indicesToChange = trainingImagesWithMaskLabelsTemp==x
        sample_weights[indicesToChange] = class_weights_freq_normalized_final[x]

    return sample_weights

def plotModelPNG(model):
    modelImagePath = './model.png'
    tf.keras.utils.plot_model(model, to_file=modelImagePath, show_shapes=True, show_layer_names=True)

def showPredictionsAndTruths(model, kMeansClusterCenters, X, y, num_classes=10):
    randomImageIndexes = random.sample(range(X.shape[0]), 10)
    imagesToPredict = X[randomImageIndexes]
    pred = model.predict(imagesToPredict).reshape(-1, 128, 128, num_classes)
    
    for index, randomImageIndex in enumerate(randomImageIndexes):
        plt.imshow(kMeansClusterCenters[np.argmax(pred[index], axis=2)].astype("int"))
        plt.show()
        plt.imshow(kMeansClusterCenters[y[randomImageIndex]].astype("int"))
        plt.show()
    
