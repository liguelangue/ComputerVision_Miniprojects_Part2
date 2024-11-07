# CS5330_F24_Group8_Assignment9-1

## Project Members

Anning Tian, Pingyi Xu, Qinhao Zhang, Xinmeng Wu

## Setup Instructions

Reuse the dataset from mini-project7

Download the LabelMe --> 

``` pip install labelme ```

## Usage Guide
`labelme` to open the LabelMe Desktop App

 - open the data file dir in the LabelMe
 - use **Create Rectangle** to annotate the target item in each image
 - set a new file dir and save the image with the annotation into it by different labels
 - use **json_to_csv.py** to convert saved JSON files into CSV files
 - **rcnn_multi.py** is the modified RCNN model for multi-class detection.

## Description 
1. Selected 3 different classes (TV, Phone, and Glasses) from our mini-projrct 7 dataset

2. The images from each class only contain one coorespomding item, such as a image from TV only contains TV, no other items from other classes.

3. After using LabelMe, we got 3 files that contain saved JSON files.

4. We used **json_to_csv.py** convert saved JSON files into CSV files and the final file structure is that a file with all CSV files and a cooresponding file with images for each class. 

5. Using collected dataset to train the RCNN model and extract the model as h5 file.


## The link to the trained model
https://drive.google.com/...
