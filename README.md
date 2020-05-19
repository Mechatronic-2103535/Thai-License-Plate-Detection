## Introduction
First of all this is my first github writing so i might explaine some thing quite weird i sorry from here LOL. For someone who begin to make something like this don't worry i will walk you though step by step

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
  1.Start by connecting gdrive into the google colab
     ```
     from google.colab import drive
     drive.mount('/content/drive')
     ```
  2.Place location of darkflow  in your Google drive
     ```
     %cd /content/ 
     !git clone https://github.com/thtrieu/darkflow.git
     %cd /content/darkflow
     !python setup.py build_ext --inplace
     ```
  3.change tensorflow version to tensorflow 1.0 
    ```
    pip install tensorflow==1.0
    ```
  4.
