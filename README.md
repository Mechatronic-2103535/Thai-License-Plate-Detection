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
   
