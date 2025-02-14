# CS5330_F24_Group8_Mini_Project_9

## project members

Anning Tian, Pingyi Xu, Qinhao Zhang, Xinmeng Wu

## setup instructions

Download the zip file or use GitHub Desktop to clone the file folder

Open the folder and run the python file --> 

``` python WebCamSave.py ```

WebCamSave.py is the name of the Python file     

Run the file to initial the live cam detection     

## usage guide
For windows: python/python3 WebcamSave_own_windows.py      
For mac: python/python3 WebcamSave_own_mac.py       
## description of the project
For windows:
Model: yoloV5m.pt    
--img 640    
--epoch 150   
--batch-size 16  
       
For mac:    
Model: yoloV5n.pt   
--img 320   
--epoch 100   
--batch-size default   

## the link to the video demonstrating the application’s execution
https://northeastern.zoom.us/rec/share/UtCU7pPjiXOpv6H6UgafInP_EgRK2kC1De45IXN3yedXqdCs1PxJDY8vgKVnfK9D.AR9GchSa5coyEJjl?startTime=1730969196000    
     
## the link to the model and dataset
https://drive.google.com/drive/folders/1-hJrmhvnCkEGaa6dsMNjiiF5bGGs0BsP?usp=sharing

### model evaluation metrics
![1](./images/confusion_matrix.png)
![2](./images/PR_curve.png)

