#%matplotlib inline

import cv2               #for image read, flip, crop etc
import csv               #for csv file operation
import numpy as np       #for numpy array operation
import os                #for directory operation
import sklearn           #for yield, util etc
from sklearn.model_selection import train_test_split    # split train test data

#for model
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt


def _getLinesFromDrivingLogs(imagepath):
    lines = []
    with open(imagepath + '/driving_log.csv') as f:
        reader = csv.reader(f)
        for line in reader:
            lines.append(line)
    return lines

def _getRelativeImagePath(dataPath, line):
    """
    @description:      This module gets relative image path from absolution path
    @param line:       line read from driving log csv
    @return newLine:   new line with relative path
    @example input:    /Users/udacity/CarND-Behavioral-Cloning-P3/data/IMG/center_2017_12_02_17_49_38_331.jpg
    @example output:   ./data/IMG/center_2017_12_02_17_49_38_331.jpg
    """
    source_path = line
    filename = source_path.split('/')[-1]
    newLine = dataPath +'/IMG/'+filename.strip()
    return newLine

def getdata(imagepath):
    dirc = [x[0] for x in os.walk(imagepath)]
    data_dirc = list(filter(lambda x :os.path.isfile(x+'/driving_log.csv'), dirc))
    print(data_dirc)
    centerpic = []
    leftpic = []
    rightpic = []
    angel = []
    lines = _getLinesFromDrivingLogs(data_dirc[0])
    for line in lines:
        angel.append(float(line[3]))
        centerpic.append(_getRelativeImagePath(imagepath,line[0]))
        #centerpic.append(imagepath+'/'+line[0])
        leftpic.append(_getRelativeImagePath(imagepath,line[1]))
        rightpic.append(_getRelativeImagePath(imagepath,line[2]))
    
    return (centerpic, leftpic, rightpic, angel)


def Correct_data(centerPaths, leftPaths, rightPaths, angel, correction):
    imagepath = []
    imagepath.extend(centerPaths)
    imagepath.extend(leftPaths)
    imagepath.extend(rightPaths)
    angels = []
    angels.extend(angel)
    angels.extend(x + correction for x in angel)
    angels.extend(x - correction for x in angel)
    return (imagepath, angels)

def imgProc(batch_samples):
    images = []
    angles = []
    for imagePath, measurement in batch_samples:
        originalImage = cv2.imread(imagePath)
        image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
        images.append(image)
        angles.append(measurement)
        # Data augment: Flipping images
        images.append(cv2.flip(image,1))
        #print(len(images))
        angles.append(measurement*-1.0)
        #print(len(angles))
        plt.imshow(imgage)

def generator(samples, batch_size=32):
    """
    @description:      generates required images and measurement 
                       using sample(training/validation) in batches of batch size
    @param samples:    list of pairs containing imagePath and measuremnet
    @param batch_size: batch size to generate data, default is 32 
    """
    num_samples = len(samples)
    while 1: # Loops forever, generator never ends
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            imgProc(samples[offset:offset+batch_size])
            # Data augment: trim image to only see section with road
            inputs = np.array(images)
            outputs = np.array(angles)
            yield sklearn.utils.shuffle(inputs, outputs)

def getNvidiaModel():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100))
     
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model


# Reading images.
centerPaths, leftPaths, rightPaths, angel = getdata('D:/Student Data/Desktop/CarND-Behavioral-Cloning/data')
print(centerPaths[1])
print(leftPaths[0])
print(rightPaths[0])
print(len(angel))
imagePaths, measurements = Correct_data(centerPaths, leftPaths, rightPaths, angel, 0.2)
print(imagePaths[0])
print(measurements[0])
print('Total Images: {}'.format( len(imagePaths)))


# Splitting samples into training and validation samples
samples = list(zip(imagePaths, measurements))
#print(samples)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print('Train samples: {}'.format(len(train_samples)))
print('Validation samples: {}'.format(len(validation_samples)))


# Creating train and validation generators.
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


# Creating Model
model = getNvidiaModel()

# Compiling and training the model
model.compile(loss='mse', optimizer='adam')
 
model.summary()


history = model.fit_generator(train_generator, samples_per_epoch= len(train_samples) , validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=2, verbose=1)

    
model.save('model.h5')