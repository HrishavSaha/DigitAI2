#Import Statements
import cv2
import numpy as np
import pandas as pd

#Commenting out seaborn and matplotlib as they aren't used
#import seaborn as sns
#import matplotlib.pyplot as plt

#COmmenting fetch_openml as I've downloaded the data and stored it in csv files
#from sklearn.datasets import fetch_openml

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time

#Importing warnings and using the following command ignores any and all warnings raised by python
#This is important later
import warnings
warnings.filterwarnings("ignore")

#Setting an HTTPS Context to fetch data from OpenML
#if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
#    getattr(ssl, '_create_unverified_context', None)): 
#    ssl._create_default_https_context = ssl._create_unverified_context

#Fetching the data
#X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

#Saving the data to csv files
#X.to_csv('xData.csv', index=False)
#y.to_csv('yData.csv', index=False)

#Inverting the images to make it more suitable for real-life scenarios
#And saving it as a csv file
#inverted_X = 255.0 - X
#inverted_X.to_csv('xInvertedData.csv', index=False)

#Reading the data from csv files
#Note that I'm importing xInvertedData.csv, rather than xData.csv
#['class'] is added at the end of y to convert it from a DataFrame to a Series
#This is to preserve its objectType
X = pd.read_csv('xInvertedData.csv')
y = pd.read_csv('yData.csv')['class']

#Creating classes and nclasses
classes = ['0', '1', '2','3', '4','5', '6', '7', '8', '9']
nclasses = len(classes)

#Splitting the data and scaling it
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size=7500, test_size=2500)

#Scaling the features
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

#Fitting the training data into the model
clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(X_train_scaled, y_train)

#Calculating the accuracy of the model
y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
#print("The accuracy is :- ",accuracy)

#Initialising Camera Object
cap = cv2.VideoCapture(0)

#PAY
#ATTENTION
#HERE

# Infinite loop. pretty simple
while(True):
  #try-except block to handle errors
  try:
    #Capturing the image in FRAME
    ret, frame = cap.read()

    #Storing a grayscale, i.e, black-n-white, version of FRAME in GRAY
    #We will take the input from this version
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Getting the resolution of the FRAME/GRAY
    height, width = gray.shape

    #We will take input for the model from within a rectangle

    #To draw a rectangle, we need its upper-left corner co-ordinate
    #And bottom-right corner co-ordinate
    upper_left = (int(width / 2 - 56), int(height / 2 - 56))
    bottom_right = (int(width / 2 + 56), int(height / 2 + 56))

    #Drawing the rectangle on FRAME
    #And storing it in FRAME and not GRAY
    #This is important because, if we store it in GRAY
    #The border of the rectangle can show up in input and mess up the model's prediction
    frame = cv2.rectangle(frame, upper_left, bottom_right, (0, 255, 0), 2)

    #Here, we are extracting the data for input from GRAY
    #roi = Region Of Interest
    roi = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]

    #The world is still quite varied in colour in grayscale
    #We need to clamp all shades of black and white to their pure states
    #Simply put, we want all sorts of grays to disappear
    #How? By setting a threshold
    #And any shade above this threshold will be considered #000000, i.e, pure black
    #And any shade below or equal to this threshold will be considered #ffffff, i.e, pure white

    #Setting the threshold
    threshold = 150
    #Clamping the colors
    roi[roi>threshold] = 255
    roi[roi<=threshold] = 0

    #We still have some data prepping left
    #We still have to resize the image to match the target resolution
    #And anti-alias the resized image to make it smoother, as it gets jagged upon pixelation

    #What's anti-aliasing, you ask?
    #Ever played games?
    #Have you seen that on worse-than-average computers, the game has jagged images
    #Like a car has jagged borders, or a gun has jagged ends
    #That's aliasing
    #Most latest games have a setting called "Anti-Aliasing"
    #This essentially smooths out those jagged edges and makes the game look better
    #At the cost of computational power, of course

    #Back to the matter at hand
    #Remember back when we clamped the image?
    #Now, after we resize the image to a low resolution
    #It will look very jagged due to the pixelation
    #Hence, the need to anti-alias the image

    #First, we need to convert the image from cv2 format to PIL format
    #This is because anti-aliasing is a PIL feature
    im_pil = Image.fromarray(roi)

    #Now, we will convert the image to grayscale
    #Before you ask, this is because WE know that the image is in grayscale, but PIL doesn't and has additional data
    #This additional data is useless anyway, but it will interfere with the prediction
    image_bw = im_pil.convert('L')
    #Now, we first resize the image and then apply anti-aliasing
    image_bw_resized_inverted = image_bw.resize((28,28), Image.ANTIALIAS)

    #Now, we convert the PIL image into a NumPy Array and reshape it, as our model can predict only NumPy arrays
    test_sample = np.array(image_bw_resized_inverted).reshape(1,784)
    #Finally, we let the model predict
    test_pred = clf.predict(test_sample)
    #Prediction is printed
    print("Predicted class is: ", test_pred)

    #Display the resulting frame
    cv2.imshow('frame',frame)
    cv2.imshow('roi',roi)
    cv2.imshow('grayscale',gray)
    #This is to clear the terminal screen after every prediction
    os.system('cls')

    #Here, we are checking for the user pressing Q key, upon which, the loop will break and we will intiate clean-up
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  #except block. Nothing going on here, but it's necessary to type in this block, as a try block MUST be accompanied with an except block
  except Exception as e:
    pass

#This is the clean-up section.
#Here, we release the Camera Object, freeing the memory allocated to it, and destroy all cv2 windows open
cap.release()
cv2.destroyAllWindows()

#CONGRATS! YOU REACHED THE END!!!