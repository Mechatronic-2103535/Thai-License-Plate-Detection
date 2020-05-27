## Introduction
First of all this is my first github writing so i might explaine something quite weird i sorry from here LOL. For someone who begin to make something like this don't worry i will walk you though step by step (If you have any question you can ask me thought my email Thanapol.ds@gmail.com I will try my best to answers all your questions

## Requirements

Python3, tensorflow 1.0, numpy, opencv 3, Raspi model 4B or othere model,Raspi CAM:

## Detail
Today i gonna show you how to do a realtime object detection using darkflow on Raspi 4.I gonna split it to 3 part 
1.) Train a custom object detection using darkflow (I used google colad cause my notebook don't have enough space for CONDA)
2.) Using KNN to do a characteristic recognition in Thai languages
3.) How to implement all model on raspi model 4B


## PART 1 Train model to detect License plate 
1.Before you do anything you will start with collect some image of plate license as mush as you can for me I took about 500 pic it took me  about a week to collect with this number but i suggest you to take more then that cause it will effect you accuracy in the end (2000 pic is quite good enough)

2.After that we will create annontation from the picture I ask you to take next(hope you get enough).you will need to label it using LabelImg go to study how to use it at https://github.com/tzutalin/labelImg

3.Then i suggest you to put you pic and annontation to you google drive and start with google colab

Getting start with code:

  1. Start by connecting gdrive into the google colab
     ```
     from google.colab import drive
     drive.mount('/content/drive')
     ```
     
  2. Place location of darkflow  in your Google drive
     ```
     %cd /content/ 
     !git clone https://github.com/thtrieu/darkflow.git
     %cd /content/darkflow
     !python setup.py build_ext --inplace
     ```
     
  3. Change tensorflow version to tensorflow 1.0 
    ```
    pip install tensorflow==1.0
    ```
    
  4. import all necessary lib
   ```
   import tensorflow as tf
   import numpy as np
   import cv2
   import imutils
   import pprint as pp
   %matplotlib inline
   import matplotlib.pyplot as plt
   from darkflow.net.build import TFNet
   ```
    
  5. Load Img and anontation up to where your clip or video is placed for me
   ```
  !unzip /content/dataset.zip
   ```
  6. Load model that already pretrain
  ```
  %cd /content/darkflow/
  %rm -rf weights
  %mkdir weights
  %cd weights/
  !wget https://oc.codespring.ro/s/Jgyo6N4Jen3ma2P/download
  %mv download tiny-yolo-voc.weights
  ```
  
  7. Config cfg file according to darkflow doc by dupicate tiny-yolo-voc.cfg and change name to tiny-yolo-voc-1c.cfg and 
     change classes in the [region] layer (the last layer) to the number of classes you are going to train for. In our case, 
     classes are set to 1 then change filters in the [convolutional] layer (last layer) to num * (classes + 5). In our case, 
     num is 5 and classes are 1 so 5 * (1 + 5) = 30 therefore filters are set to 30.than upload back to colab local 
  
  8. Change File /content/darkflow/label.txt >> plate (According to we only need 1 label in this project)
  
  9. Get Weights training
  ```
  options = {"model": "/content/darkflow/cfg/tiny-yolo-voc-1c.cfg", 
           "load": "/content/darkflow/weights/tiny-yolo-voc.weights", 
           "batch": 8,
           "epoch": 40,
           "gpu": 1.0,
           "train": True,
           "annotation": "place where you keep you annotation",
           "dataset": "place where you keep you picture"}
   ```
   
   10. Start train model
   ```
   tfnet = TFNet(options)
   tfnet.train()
   tfnet.savepb()
   ```
   
   11. Then download you datkflow folder to you computer
   
   # Extra
   To test you model is work or not 
   ```
   options = {"model": "/content/darkflow/cfg/tiny-yolo-voc-1c.cfg",
           "load": 992,
           "gpu": 1.0,
          "pbLoad":"/content/darkflow/build_graph/tiny-yolo-voc-1c.pb" ,
          "metaLoad":"/content/darkflow/build_graph/tiny-yolo-voc-1c.meta"}
   tfnet2 = TFNet(options)
   tfnet2.load_from_ckpt()
   original_img = cv2.imread("Picture you want to test")
   original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
   results = tfnet2.return_predict(original_img)
   # This is a python function used to create bounding boxes and confidence score around Plate
def boxing(original_img , predictions):
    newImage = np.copy(original_img)

    for result in predictions:
        top_x = result['topleft']['x']
        top_y = result['topleft']['y']

        btm_x = result['bottomright']['x']
        btm_y = result['bottomright']['y']

        confidence = result['confidence']
        label = result['label'] + " " + str(round(confidence, 3))
        
        if confidence > 0.2:
            newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255,0,0), 3)
            newImage = cv2.putText(newImage, label, (top_x, top_y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.8, (0, 230, 0), 1, cv2.LINE_AA)
        
    return newImage
    
    # This will show your original image on which you want to use prediction
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(original_img)
   ```
   
## Part 2 Character Recognition
# Part 2.1 Train computer to remember Thai character in txt file format from Thai Character in PNG file format

# CreateDATA.py
   ```
import sys
import numpy as np
import cv2
import os

# module level variables ##########################################################################
MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

###################################################################################################
def main():
    imgTrainingNumbers = cv2.imread("Thai3.png")            # read in training numbers image

    if imgTrainingNumbers is None:                          # if image was not read successfully
        print("error: image not read from file \n\n")       # print error message to std out
        os.system("pause")                                  # pause so user can see error message
        return                                              # and exit function (which exits program)
    # end if

    imgGray = cv2.cvtColor(imgTrainingNumbers, cv2.COLOR_BGR2GRAY)          # get grayscale image
    imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)                        # blur

                                                        # filter image from grayscale to black and white
    imgThresh = cv2.adaptiveThreshold(imgBlurred,                           # input image
                                      255,                                  # make pixels that pass the threshold full white
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # use gaussian rather than mean, seems to give better results
                                      cv2.THRESH_BINARY_INV,                # invert so foreground will be white, background will be black
                                      11,                                   # size of a pixel neighborhood used to calculate threshold value
                                      2)                                    # constant subtracted from the mean or weighted mean
    cv2.imshow("imgThresh", imgThresh)      # show threshold image for reference

    imgThreshCopy = imgThresh.copy()        # make a copy of the thresh image, this in necessary b/c findContours modifies the image
 # imgContours
    imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,        # input image, make sure to use a copy since the function will modify this image in the course of finding contours
                                                 cv2.RETR_EXTERNAL,                 # retrieve the outermost contours only
                                                 cv2.CHAIN_APPROX_SIMPLE)           # compress horizontal, vertical, and diagonal segments and leave only their end points

                                # declare empty numpy array, we will use this to write to file later
                                # zero rows, enough cols to hold all image data
    npaFlattenedImages =  np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

    intClassifications = []         # declare empty classifications list, this will be our list of how we are classifying our chars from user input, we will write to file at the end

                                    # possible chars we are interested in are digits 0 through 9, put these in list intValidChars
    intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                     ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
                     ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
                     ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z'),ord('a'), ord('b'), ord('c'), ord('d'), 
                     ord('e'), ord('f'), ord('g'), ord('h'), ord('i'), ord('j'),ord('k'), ord('l'), ord('m'), ord('n'), 
                     ord('o'), ord('p'), ord('q'), ord('r'), ord('s'), ord('t'),ord('u'), ord('v'), ord('w'), ord('x'), 
                     ord('y'), ord('z')]

    for npaContour in npaContours:                          # for each contour
        if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:          # if contour is big enough to consider
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour)         # get and break out bounding rect

                                                # draw rectangle around each contour as we ask user for input
            cv2.rectangle(imgTrainingNumbers,           # draw rectangle on original training image
                          (intX, intY),                 # upper left corner
                          (intX+intW,intY+intH),        # lower right corner
                          (0, 0, 255),                  # red
                          2)                            # thickness

            imgROI = imgThresh[intY:intY+intH, intX:intX+intW]                                  # crop char out of threshold image
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))     # resize image, this will be more consistent for recognition and storage
            cv2.imshow("imgROI", imgROI)                    # show cropped out char for reference
            cv2.imshow("imgROIResized", imgROIResized)      # show resized image for reference
            cv2.imshow("training_numbers.png", imgTrainingNumbers)      # show training numbers image, this will now have red rectangles drawn on it
            intChar = cv2.waitKey(0)                  # get key press

            if intChar == 27:                   # if esc key was pressed
                sys.exit()                      # exit program
            elif intChar in intValidChars:      # else if the char is in the list of chars we are looking for . . .
                if intChar == ord('A'):
                    intChar = ord('ก')
                    print(chr(intChar))
                if intChar == ord('B'):
                    intChar = ord('ข')
                    print(chr(intChar))
                if intChar == ord('C'):
                    intChar = ord('ฃ')
                    print(chr(intChar))
                if intChar == ord('D'):
                    intChar = ord('ค')
                    print(chr(intChar))
                if intChar == ord('E'):
                    intChar = ord('ฅ')
                    print(chr(intChar))
                if intChar == ord('F'):
                    intChar = ord('ฆ')
                    print(chr(intChar))
                if intChar == ord('G'):
                    intChar = ord('ง')
                    print(chr(intChar))
                if intChar == ord('H'):
                    intChar = ord('จ')
                    print(chr(intChar))
                if intChar == ord('I'):
                    intChar = ord('ฉ')
                    print(chr(intChar))
                if intChar == ord('J'):
                    intChar = ord('ช')
                    print(chr(intChar))
                if intChar == ord('K'):
                    intChar = ord('ซ')
                    print(chr(intChar))
                if intChar == ord('L'):
                    intChar = ord('ฌ')
                    print(chr(intChar))
                if intChar == ord('M'):
                    intChar = ord('ญ')
                    print(chr(intChar))
                if intChar == ord('N'):
                    intChar = ord('ฎ')
                    print(chr(intChar))
                if intChar == ord('O'):
                    intChar = ord('ฏ')
                    print(chr(intChar))
                if intChar == ord('P'):
                    intChar = ord('ฐ')
                    print(chr(intChar))
                if intChar == ord('Q'):
                    intChar = ord('ฑ')
                    print(chr(intChar))
                if intChar == ord('R'):
                    intChar = ord('ฒ')
                    print(chr(intChar))
                if intChar == ord('S'):
                    intChar = ord('ณ')
                    print(chr(intChar))
                if intChar == ord('T'):
                    intChar = ord('ด')
                    print(chr(intChar))
                if intChar == ord('U'):
                    intChar = ord('ต')
                    print(chr(intChar))
                if intChar == ord('V'):
                    intChar = ord('ถ')
                    print(chr(intChar))
                if intChar == ord('W'):
                    intChar = ord('ท')
                    print(chr(intChar))
                if intChar == ord('X'):
                    intChar = ord('ธ')
                    print(chr(intChar))     
                if intChar == ord('Y'):
                    intChar = ord('น')
                    print(chr(intChar))
                if intChar == ord('Z'):
                    intChar = ord('บ')  
                    print(chr(intChar))
                if intChar == ord('a'):
                    intChar = ord('ป')
                    print(chr(intChar)) 
                if intChar == ord('b'):
                    intChar = ord('ผ')
                    print(chr(intChar))
                if intChar == ord('c'):
                    intChar = ord('ฝ')
                    print(chr(intChar))
                if intChar == ord('d'): 
                    intChar = ord('พ')
                    print(chr(intChar))
                if intChar == ord('e'):
                    intChar = ord('ฟ')
                    print(chr(intChar))
                if intChar == ord('f'):
                    intChar = ord('ภ')
                    print(chr(intChar))
                if intChar == ord('g'):
                    intChar = ord('ม')
                    print(chr(intChar))
                if intChar == ord('h'):
                    intChar = ord('ย')
                    print(chr(intChar))
                if intChar == ord('i'):
                    intChar = ord('ร')
                    print(chr(intChar))
                if intChar == ord('j'):
                    intChar = ord('ล')
                    print(chr(intChar))
                if intChar == ord('k'):
                    intChar = ord('ว')
                    print(chr(intChar))
                if intChar == ord('l'):
                    intChar = ord('ศ')
                    print(chr(intChar))
                if intChar == ord('m'):
                    intChar = ord('ษ')
                    print(chr(intChar))
                if intChar == ord('n'):
                    intChar = ord('ส')
                    print(chr(intChar))
                if intChar == ord('o'):
                    intChar = ord('ห')
                    print(chr(intChar))
                if intChar == ord('p'):
                    intChar = ord('ฬ')
                    print(chr(intChar))
                if intChar == ord('q'):
                    intChar = ord('อ')
                    print(chr(intChar))
                if intChar == ord('r'):
                    intChar = ord('ฮ')
                    print(chr(intChar))
                if intChar == ord('0'):
                    print(chr(intChar))
                if intChar == ord('1'):
                    print(chr(intChar))
                if intChar == ord('2'):
                    print(chr(intChar))
                if intChar == ord('3'):
                    print(chr(intChar))
                if intChar == ord('4'):
                    print(chr(intChar))
                if intChar == ord('5'):
                    print(chr(intChar))
                if intChar == ord('6'):
                    print(chr(intChar))
                if intChar == ord('7'):
                    print(chr(intChar))
                if intChar == ord('8'):
                    print(chr(intChar))
                if intChar == ord('9'):
                    print(chr(intChar))

                    """
                intClassifications.append(intChar)                                                # append classification char to integer list of chars (we will convert to float later before writing to file)
                npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
                npaFlattenedImage = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))  # flatten image to 1d numpy array so we can write to file later
                npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0)                    # add current flattened impage numpy array to list of flattened image numpy arrays
 """
                intClassifications.append(intChar)                                                # append classification char to integer list of chars (we will convert to float later before writing to file
                npaFlattenedImage = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))  # flatten image to 1d numpy array so we can write to file later
                npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0)                    # add current flattened impage numpy array to list of flattened image numpy arrays
           
            # end if
        # end if
    # end for
    """
    npaClassifications = np.loadtxt("classifications.txt", np.float32)  
    #npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32) 
    fltClassifications = np.array(intClassifications, np.float32)                   # convert classifications list of ints to numpy array of floats
   
    npaClassifications = np.append(npaClassifications,fltClassifications.reshape((fltClassifications.size, 1)))  # flatten numpy array of floats to 1d so we can write to file later
    

    #np.savetxt("/content/OpenCV_3_KNN_Character_Recognition_Python-master/classifications.txt", npaClassifications)           # write flattened images to file
    #np.savetxt("/content/OpenCV_3_KNN_Character_Recognition_Python-master/flattened_images.txt", npaFlattenedImages)          #
    print("\n\ntraining complete !!\n")

    np.savetxt("classifications.txt", npaClassifications)           # write flattened images to file
    np.savetxt("flattened_images.txt", npaFlattenedImages)          #

    cv2.destroyAllWindows()             # remove windows from memory
        # remove windows from memory

    return"""
    fltClassifications = np.array(intClassifications, np.float32)                   # convert classifications list of ints to numpy array of floats

    npaClassifications = fltClassifications.reshape((fltClassifications.size, 1))   # flatten numpy array of floats to 1d so we can write to file later

    print("\n\ntraining complete !!\n")

    np.savetxt("classifications.txt", npaClassifications)           # write flattened images to file
    np.savetxt("flattened_images.txt", npaFlattenedImages)          #

    cv2.destroyAllWindows()             # remove windows from memory

    return

###################################################################################################
if __name__ == "__main__":
    main()
# end if



   ```
   
 #Part 2.2 Code for check txt file form match whether it work correctly or not
 
 # Test.py
 ```

# TrainAndTest.py

import cv2
import numpy as np
import operator
import os

# module level variables ##########################################################################
MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

###################################################################################################
class ContourWithData():

    # member variables ############################################################################
    npaContour = None           # contour
    boundingRect = None         # bounding rect for contour
    intRectX = 0                # bounding rect top left corner x location
    intRectY = 0                # bounding rect top left corner y location
    intRectWidth = 0            # bounding rect width
    intRectHeight = 0           # bounding rect height
    fltArea = 0.0               # area of contour

    def calculateRectTopLeftPointAndWidthAndHeight(self):               # calculate bounding rect info
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

    def checkIfContourIsValid(self):                            # this is oversimplified, for a production grade program
        if self.fltArea < MIN_CONTOUR_AREA: return False        # much better validity checking would be necessary
        return True

###################################################################################################
def main():
    allContoursWithData = []                # declare empty lists,
    validContoursWithData = []              # we will fill these shortly

    try:
        npaClassifications = np.loadtxt("classifications.txt", np.float32)                  # read in training classifications
    except:
        print("error, unable to open classifications.txt, exiting program\n")
        os.system("pause")
        return
    # end try

    try:
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)                 # read in training images
    except:
        print("error, unable to open flattened_images.txt, exiting program\n")
        os.system("pause")
        return
    # end try

    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))       # reshape numpy array to 1d, necessary to pass to call to train

    kNearest = cv2.ml.KNearest_create()                   # instantiate KNN object
    print(npaFlattenedImages)
    print(npaClassifications)
    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

    imgTestingNumbers = cv2.imread("T1.png")          # read in testing numbers image

    if imgTestingNumbers is None:                           # if image was not read successfully
        print("error: image not read from file \n\n")        # print error message to std out
        os.system("pause")                                  # pause so user can see error message
        return                                              # and exit function (which exits program)
    # end if

    imgGray = cv2.cvtColor(imgTestingNumbers, cv2.COLOR_BGR2GRAY)       # get grayscale image
    imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)                    # blur

                                                        # filter image from grayscale to black and white
    imgThresh = cv2.adaptiveThreshold(imgBlurred,                           # input image
                                      255,                                  # make pixels that pass the threshold full white
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # use gaussian rather than mean, seems to give better results
                                      cv2.THRESH_BINARY_INV,                # invert so foreground will be white, background will be black
                                      11,                                   # size of a pixel neighborhood used to calculate threshold value
                                      2)                                    # constant subtracted from the mean or weighted mean

    imgThreshCopy = imgThresh.copy()        # make a copy of the thresh image, this in necessary b/c findContours modifies the image
#imgContours 
    imgContours,npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,             # input image, make sure to use a copy since the function will modify this image in the course of finding contours
                                                 cv2.RETR_EXTERNAL,         # retrieve the outermost contours only
                                                 cv2.CHAIN_APPROX_SIMPLE)   # compress horizontal, vertical, and diagonal segments and leave only their end points

    for npaContour in npaContours:                             # for each contour
        contourWithData = ContourWithData()                                             # instantiate a contour with data object
        contourWithData.npaContour = npaContour                                         # assign contour to contour with data
        contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)     # get the bounding rect
        contourWithData.calculateRectTopLeftPointAndWidthAndHeight()                    # get bounding rect info
        contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)           # calculate the contour area
        allContoursWithData.append(contourWithData)                                     # add contour with data object to list of all contours with data
    # end for

    for contourWithData in allContoursWithData:                 # for all contours
        if contourWithData.checkIfContourIsValid():             # check if valid
            validContoursWithData.append(contourWithData)       # if so, append to valid contour list
        # end if
    # end for

    validContoursWithData.sort(key = operator.attrgetter("intRectX"))         # sort contours from left to right

    strFinalString = ""         # declare final string, this will have the final number sequence by the end of the program

    for contourWithData in validContoursWithData:            # for each contour
                                                # draw a green rect around the current char
        cv2.rectangle(imgTestingNumbers,                                        # draw rectangle on original testing image
                      (contourWithData.intRectX, contourWithData.intRectY),     # upper left corner
                      (contourWithData.intRectX + contourWithData.intRectWidth, contourWithData.intRectY + contourWithData.intRectHeight),      # lower right corner
                      (0, 255, 0),              # green
                      2)                        # thickness

        imgROI = imgThresh[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight,     # crop char out of threshold image
                           contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth]

        imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))             # resize image, this will be more consistent for recognition and storage

        npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))      # flatten image into 1d numpy array

        npaROIResized = np.float32(npaROIResized)       # convert from 1d numpy array of ints to 1d numpy array of floats

        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)     # call KNN function find_nearest

        strCurrentChar = str(chr(int(npaResults[0][0])))                                             # get character from results

        strFinalString = strFinalString + strCurrentChar            # append current char to full string
    # end for

    print("\n" + strFinalString + "\n")                  # show the full string
    cv2.imshow("imgTestingNumbers", imgTestingNumbers)
    cv2.waitKey(0)                                          # wait for user key press

    cv2.destroyAllWindows()             # remove windows from memory

    return

###################################################################################################
if __name__ == "__main__":
    main()
# end if



 ```












## Part 3 implement on Raspi model 4
   first of all start with install cv2 library follow this link : https://www.learnopencv.com/install-opencv-4-on-raspberry-pi/
   then this is you code
   ```
   import tensorflow as tf
#from tensorflow.keras import layers, models
import numpy as np
import cv2
import imutils
import pprint as pp
#matplotlib inline
import matplotlib.pyplot as plt
from darkflow.net.build import TFNet
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import os
import DetectChars
import DetectPlates
import PossiblePlate
import json
import requests

options = {"model": "/home/pi/Desktop/darkflow/cfg/tiny-yolo-voc-1c.cfg",
           "load": 992,
           "gpu": 1.0,
          "pbLoad":"/home/pi/Desktop/darkflow/build_graph/tiny-yolo-voc-1c.pb" ,
          "metaLoad":"/home/pi/Desktop/darkflow/build_graph/tiny-yolo-voc-1c.meta"}

tfnet2 = TFNet(options)
tfnet2.load_from_ckpt()


# This is a python function used to create bounding boxes and confidence score around Plate
def boxing(original_img , predictions):
    newImage = np.copy(original_img)

    for result in predictions:
        top_x = result['topleft']['x']
        top_y = result['topleft']['y']
        btm_x = result['bottomright']['x']
        btm_y = result['bottomright']['y']
        im   =  newImage[top_y:btm_y,top_x:btm_x]
        Content = main(im)
        confidence = result['confidence']
        label = result['label'] + " " + str(round(confidence, 3))
        if confidence > 0.2:
            newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255,0,0), 3)
            newImage = cv2.putText(newImage, label, (top_x, top_y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.8, (0, 230, 0), 1, cv2.LINE_AA)
        
    return newImage,Content
def main(x):

    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()         # attempt KNN training

    if blnKNNTrainingSuccessful == False:                               # if KNN training was not successful
        print("\nerror: KNN traning was not successful\n")  # show error message
        return                                                          # and exit program
    # end if

    imgOriginalScene  = x               # open image

    if imgOriginalScene is None:                            # if image was not read successfully
        print("\nerror: image not read from file \n\n")  # print error message to std out
        os.system("pause")                                  # pause so user can see error message
        return                                              # and exit program
    # end if

    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)           # detect plates

    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)        # detect chars in plates

    #cv2.imshow("imgOriginalScene", imgOriginalScene)            # show scene image

    if len(listOfPossiblePlates) == 0:                          # if no plates were found
        print("\nno license plates were detected\n")  # inform user no plates were found
    else:                                                       # else
                # if we get in here list of possible plates has at leat one plate

                # sort the list of possible plates in DESCENDING order (most number of chars to least number of chars)
        listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)

                # suppose the plate with the most recognized chars (the first plate in sorted by string length descending order) is the actual plate
        licPlate = listOfPossiblePlates[0]

        #cv2.imshow("imgPlate", licPlate.imgPlate)           # show crop of plate and threshold of plate
        #cv2.imshow("imgThresh", licPlate.imgThresh)

        if len(licPlate.strChars) == 0:                     # if no chars were found in the plate
            print("\nno characters were detected\n\n")  # show message
            return                                          # and exit program
        # end if

        drawRedRectangleAroundPlate(imgOriginalScene, licPlate)             # draw red rectangle around plate
        #print(licPlate.strChars) 
        #print("\nlicense plate read from image = " + licPlate.strChars + "\n")  # write license plate text to std out
        #print("----------------------------------------")

        writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)           # write license plate text on the image

        #cv2.imshow("imgOriginalScene", imgOriginalScene)                # re-show scene image

        #cv2.imwrite("imgOriginalScene.png", imgOriginalScene)           # write image out to file

    # end if else

    					# hold windows open until user presses a key

    return licPlate.strChars
# end main

###################################################################################################
# initialize the camera and grab a reference to the raw camera capture

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))


# allow the camera to warm up
time.sleep(0.1)
for frame in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):

    image = frame.array
    result = tfnet2.return_predict(image)
    if result != []:
        image,plate = boxing(image, result)
        print(plate)
        checkurl =  'https://api.thingspeak.com/channels/1051778/fields/'+field(plate)+'.json?api_key=KHRYRMUKJVIFZWX6&results=1&timezone=Asia%2FBangkok'
        # 1. เลขทะเบียนจาก str รวมกลับเข้าไปเปลี่ยนสถานะของ A1 from? to 1
        
        # show the frame
    cv2.imshow("Stream", image)
    key = cv2.waitKey(1) & 0xFF
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
   ```
   
