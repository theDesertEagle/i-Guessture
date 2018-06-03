'''''''''
Author: the.desert.eagle
Application: i-Guessture, A Hand-Detection and Gesture-Recognition System
Version: 1.0
'''''''''


### Import Dependencies
import warnings
warnings.simplefilter('ignore')
import cv2
import numpy as np
import pyautogui
from sklearn.mixture import GaussianMixture
from keras.models import load_model
from getpass import getuser
from sys import argv
from math import floor

### Global Declarations
debug = False
whitePixelValue = 255;
possibleSkinSaturationHighValue = floor(0.75*whitePixelValue); # HS-Skin Color Index values
possibleSkinSaturationLowValue = floor(0.10*whitePixelValue);

### Gesture Mapping Function
def mapGesture(gestureIndex):
    global debug
    if gestureIndex == 0:
        pyautogui.press('volumedown')
        debug and print('\n[GESTURE-RECOGNITION] One')
    elif gestureIndex == 1:
        pyautogui.press('volumeup')
        debug and print('\n[GESTURE-RECOGNITION] Two')
    elif gestureIndex == 2:
        pyautogui.scroll(-10)
        debug and print('\n[GESTURE-RECOGNITION] Yes')
    elif gestureIndex == 3:
        pyautogui.press('q') 
        debug and print('\n[GESTURE-RECOGNITION] No')
    elif gestureIndex == 4:
        pyautogui.screenshot('screenshot.png')
        debug and print('\n[GESTURE-RECOGNITION] Fist')

### Main Method Definition
def main():
    
    ## Program Mode (Default: Non-Debug)
    global debug
    if len(argv) > 1: debug = True
    print('Welcome {}. Place your hand in front of the camera and gesture. Ready to serve you! {}'.format(getuser(), '\n[DEBUG-MODE]' if debug else '\n'))#if debug else '\n'))
     
    
    ## Video Capture Object Declaration
    videoCaptureObj = cv2.VideoCapture(0);
    imageCounter = 0 # For Counting/Saving Clustered Image Sequences for Training-Set Development 
    model = load_model('classifierModel.h5') # Loading Trained Keras CNN Model


    ## Video Capture Loop
    while True:        
        [retStatus, frame] = videoCaptureObj.read()
        if debug: 
            if retStatus: print('\n[SUCCESS] Frame Detected') 
            else:
                print('\n<FATAL-ERROR> Camera Disconnected. Aborting ...')
                exit()
                
        # Image Sampling and Window-of-Interest Extraction
        frame = cv2.resize(frame, (50, 50))
        [completeFrameHeight, completeFrameWidth, _] = frame.shape 
        frame = frame[floor(completeFrameHeight/4): floor(3*completeFrameHeight/4), floor(completeFrameWidth/4): floor(3*completeFrameWidth/4), :]

        # Dimensionality Reduction to HS Color Space
        [heightOfFrame, widthOfFrame, _] = frame.shape 
        sizeOfFrame = heightOfFrame * widthOfFrame
        hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hueChannel = hsvFrame[:, :, 0]
        saturationChannel = hsvFrame[:, :, 1]
        imageDataset = np.concatenate((np.column_stack((hueChannel.flatten().T, saturationChannel.flatten().T)), np.arange(sizeOfFrame).T[:, None]), axis=1); #[:, None] for concatenate to work
        
        # Possible Skin-Color Filtering using HSV Color Index
        filteredImageDataset = imageDataset[(80 >= imageDataset[:, 0]) & ((imageDataset[:, 1] >= possibleSkinSaturationLowValue) & (possibleSkinSaturationHighValue >= imageDataset[:, 1]))]   

        # Skin and Background-Colors Clustering using Gaussian Mixture Model 
        gaussianMixtureClusteringModel = GaussianMixture(n_components=2, covariance_type='diag', warm_start=True, max_iter=200).fit(filteredImageDataset[:,0:2])
        clusterLabels = gaussianMixtureClusteringModel.predict(filteredImageDataset[:,0:2])
        handLabel = -1;

        # Estimated Skin-Color Cluster Detection
        numberOfClusterZeroLabels = np.count_nonzero(clusterLabels == 0)
        numberOfClusterOneLabels = len(clusterLabels) - numberOfClusterZeroLabels
        numberOfHandPixels = 0
        if numberOfClusterZeroLabels < numberOfClusterOneLabels:
            handLabel = 0
            numberOfHandPixels = numberOfClusterZeroLabels 
            debug and print('\n[SKIN-CLUSTERING] Possible Hand Detected as Class-Zero')
        elif numberOfClusterOneLabels < numberOfClusterZeroLabels:
            handLabel = 1
            numberOfHandPixels = numberOfClusterOneLabels
            debug and print('\n[SKIN-CLUSTERING] Possible Hand Detected as Class-One')

        # Hand-Detection and Segmentation
        segmentedImage = np.zeros((sizeOfFrame), 'uint8')
        handDetected = False
        if numberOfHandPixels > 150:
            handDetected = True
            for i, clusterLabel in enumerate(clusterLabels):
                if clusterLabel == handLabel: segmentedImage[filteredImageDataset[i, 2]] = whitePixelValue
        elif numberOfHandPixels > 100 and debug: print('\n<WARNING> Possible Hand Undetected due to Threshold Failure Constraint')

        # Webcam Output Display
        segmentedImage = np.resize(segmentedImage, (heightOfFrame, widthOfFrame))
        cv2.imshow('Live Video-Feed', cv2.resize(segmentedImage, (completeFrameHeight*4, completeFrameWidth*4)))
        debug and cv2.imshow('Original Capture', cv2.resize(frame, (completeFrameHeight*4, completeFrameWidth*4))) 

        # Gesture Mapping on Hand Detection
        if handDetected:
            predictedGestureIndex = np.argmax(model.predict(segmentedImage.flatten().reshape(1,25,25,1)), axis=1);
            mapGesture(predictedGestureIndex[0])

        # Special Keys 
        keyCode = cv2.waitKey(1)
        if keyCode & 0xFF == ord('q'): break; # Exit the Program 
        elif keyCode & 0xFF == ord('w'):
            imageCounter += 1
            cv2.imwrite('[.] Image saved as {}.png for training'.format(imageCounter), segmentedImage) # Save Image for Training
            print(imageCounter)

    ## End of Video Capture Loop and Program    
    videoCaptureObj.release()
    cv2.destroyAllWindows()

### Program Execution
if __name__ == '__main__': main()
