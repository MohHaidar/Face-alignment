# Face-alignment

This script will take a folder of human images with transparent background, 
scale and center them around the face.
It uses the MTCNN module that is based on CNN to locate the face bounding box and the eyes position.


Python version >= 3.4

## Installation

1- For windows you need a python IDE (vscode, pycharm, etc..)
2- Make sure you are using python version 3.4 or higher
3- Download this repository to your local machine and extract it, or use git clone.
4- install the requirements using the following code (make sure you are connected to the internet):
```
pip install -r requirements.txt
```

## Running the script



1-Copy your images to the folder /remove bg
2-Run the face_align.py script with arguments: width of the resulted image, desired face ratio, images folder, output images folder
example:
```
python face_align.py 800 0.4 'remove bg' 'results'
```
The code will create a folder with the specified name, and save thhe images in a .png format.
The new images are square, with the faces centered exactly around the middle of the eyes horizontally.
The images where face is not detected will not be saved, and a message will show up in the console
(For detailed cropping and positioning the variables have to be changed manually inside the code)