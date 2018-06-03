'''''''''
Author: the.desert.eagle
Application: i-Guessture, A Hand-Detection and Gesture-Recognition System
Version: 1.0
'''''''''


### Import Dependencies
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.models import load_model
from collections import deque
from sys import argv
K.set_image_data_format('channels_last')

### Global Declarations
numberOfSamplesForEachGesture = 700;
numberOfSamples = numberOfSamplesForEachGesture*5;
numberOfGestures = 5
imageDimension = 25
trainingDataFeatures = []
trainingDataLabels = []
fileName = None # For Loading Purposes
trainingDataLabel = None # For Loading Purposes

### Utility Function to Return Gesture Index
def returnGesture(gestureIndex):
    if gestureIndex == 1: return 'One'
    elif gestureIndex == 2: return 'Two'
    elif gestureIndex == 3: return 'Yes'
    elif gestureIndex == 4: return 'No'
    elif gestureIndex == 5: return 'Fist'

### Testing Phase (or) One-Instance Lite Testing
if len(argv) > 1:
    print('\n<IMAGE TESTING INITIATED>\n')
    print('[LOADING CNN MODEL]')
    neuralNetwork = load_model('classifierModel.h5')
    print('[MODEL IS PREDICTION-READY]')
    print('[WAITING FOR USER TEST-IMAGE PATH/FILENAME]')
    path = input()
    prediction = neuralNetwork.predict(cv2.imread(path, 0).reshape(1, imageDimension, imageDimension, 1))
    print('[IMAGE LOADED: {}]\n[PREDICTED GESTURE: "{}"]'.format(path, returnGesture(np.argmax(prediction, axis=1)+1)))    
    print('\n<IMAGE TESTING COMPLETE>\n')
else:
    print('\n<TRAINING PHASE INTIATED>\n')
    ## Loading Data for Training
    for gestureIndex in range(1, numberOfGestures+1):
        fileName = returnGesture(gestureIndex)
        if not trainingDataLabel: trainingDataLabel = deque([1] + [0]*(numberOfGestures-1))
        else: trainingDataLabel.rotate(1) # Shift the One to the Next Element of the Vector
        
        for fileNumber in range(1, numberOfSamplesForEachGesture+1):
            imageRecord = cv2.imread('images/{}/{}{}.png'.format(fileName, fileName.lower(), fileNumber), 0).flatten()
            trainingDataFeatures.append(imageRecord)
            trainingDataLabels.append(np.array(trainingDataLabel))

        print('[{} IMAGE FILES READ FROM LOCATION - images/{}]'.format(numberOfSamplesForEachGesture, fileName))

    trainingDataFeatures = np.array(trainingDataFeatures)
    trainingDataLabels = np.array(trainingDataLabels)
    trainingDataFeatures = trainingDataFeatures.reshape((numberOfSamples, imageDimension, imageDimension, 1))
    print('[PREPARING TRAINING-DATA BY RESHAPING DATA-FEATURES TO DIMENSIONS ({}, {}, {}, {})]'.format(numberOfSamples, imageDimension, imageDimension, 1))


    ## Global Declarations for CNN Model
    kernelSize = 5
    convolutionalLayerDepth1 = 32
    convolutionalLayerDepth2 = 64
    dropoutPoolingProbability = 0.25
    dropoutFullyConnectedProbability = 0.25
    poolSize = 2
    fullyConnectedLayerSize = 512
    numberOfEpochs = 10
    batchSize = 100
    learningRate = 0.001


    ## Convolutional Neural Network (CNN) Definition
    def convolutionalNeuralNetwork():
        neuralNetwork = Sequential()

        neuralNetwork.add(Conv2D(convolutionalLayerDepth1, (kernelSize,kernelSize), padding='same', input_shape=(imageDimension, imageDimension, 1), activation='relu'))
        neuralNetwork.add(MaxPooling2D(pool_size=(poolSize,poolSize)))
            
        neuralNetwork.add(Conv2D(convolutionalLayerDepth2, (kernelSize,kernelSize), padding='same', activation='relu'))
        neuralNetwork.add(MaxPooling2D(pool_size=(poolSize,poolSize)))

        neuralNetwork.add(Dropout(dropoutPoolingProbability))
        
        neuralNetwork.add(Conv2D(convolutionalLayerDepth2, (kernelSize,kernelSize), padding='same', activation='relu'))
        neuralNetwork.add(MaxPooling2D(pool_size=(poolSize,poolSize)))

        neuralNetwork.add(Conv2D(convolutionalLayerDepth1, (kernelSize,kernelSize), padding='same', activation='relu'))
        neuralNetwork.add(MaxPooling2D(pool_size=(poolSize,poolSize)))

        neuralNetwork.add(Dropout(dropoutPoolingProbability))

        neuralNetwork.add(Flatten())
        neuralNetwork.add(Dense(fullyConnectedLayerSize, activation='relu'))
        neuralNetwork.add(Dropout(dropoutFullyConnectedProbability))

        neuralNetwork.add(Dense(numberOfGestures, activation='softmax'))
        return neuralNetwork

    neuralNetwork = convolutionalNeuralNetwork()


    ## Getting the model ready
    stochasticGradientDescentOptimizer = SGD(lr=learningRate, decay=1e-6, momentum=0.9, nesterov=True)
    print('[MODEL COMPILATION INITIATED]')
    neuralNetwork.compile(loss='categorical_crossentropy', optimizer=stochasticGradientDescentOptimizer, metrics=['accuracy'])
    print('[MODEL COMPILATION COMPLETE]')


    ## Begin Training Phase
    print('[TRAINING CNN MODEL WITH BATCH SIZE - {}]'.format(batchSize))
    neuralNetwork.fit(trainingDataFeatures, trainingDataLabels ,batch_size=batchSize, epochs=numberOfEpochs, validation_split=0.1 , callbacks=[ModelCheckpoint('classifierModel.h5', save_best_only=True)])
    print('\n<TRAINING PHASE COMPLETE>\n<READY TO TEST + DEPLOY>')
