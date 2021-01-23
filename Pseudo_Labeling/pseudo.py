import os
import tensorflow as tf
from tensorflow.keras import layers
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import PIL
from PIL import Image
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import math
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, RegressorMixin
import random

## Callback function to stop learning when converged
class Callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>0.99):
            self.model.stop_training = True
            
callback = Callback()

## Load model trained on labelled data using CNN
model=load_model('prediction_model.h5')

## Compile the model with RMSprop optimizer and categorical-cross entropy loss function
model.compile(optimizer = RMSprop(lr=0.001), loss = 'categorical_crossentropy', metrics = ['acc'])

## Preprocessing unlabeled data
face_cascade = cv2.CascadeClassifier('./haar_face.xml')

total = len(os.listdir('unlabeled'))
i = 0
for imgn in os.listdir('unlabeled'):  
  print(i*100/total)
  i = i + 1
  frame = cv2.imread('unlabeled/{}'.format(imgn))
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, 1.3, 5)
  roi=frame[100:300, 100:300]
  for (x,y,w,h) in faces:
      roi=gray[x:x+w, y:y+h]
  img=cv2.resize(roi, (48, 48))
  ## save the preprocessed unlabeled data
  cv2.imwrite('preprocess_unlabeled/{}'.format(imgn), img)


## Generate training data in batches
train = ImageDataGenerator(rescale=1.0/255,
                                  rotation_range=40,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True,
                                  fill_mode='nearest')

train_generator = train.flow_from_directory(
                    './archive/train',
                    batch_size=24176,
                    target_size=(48,48),
                    class_mode='categorical')

## training tuple
X_train, y_train = next(train_generator)

## Generate validation / test data
validation = ImageDataGenerator(rescale=1.0/255)

validation_generator = train.flow_from_directory(
                    './archive/test',
                    batch_size=6043,
                    target_size=(48,48),
                    class_mode='categorical')

## Validation / Test tuple
X_test, y_test = next(validation_generator)

## unlabeled data generator in a single batch
unlabeled_generator = train.flow_from_directory(
                    './Unlabeled_Total',
                    batch_size=13233,
                    target_size=(48,48),
                    class_mode=None)
unlabeled_X = next(unlabeled_generator)

## Defining the Pseudolabeler class with functions for processing and training the augmented data
class PseudoLabeler(BaseEstimator, RegressorMixin):

    
    def __init__(self, model, unlabled_data, sample_rate=0.2, seed=42):
        '''
        @sample_rate - percent of samples used as pseudo-labelled data
                       from the unlabled dataset
        '''
        assert sample_rate <= 1.0,

        self.sample_rate = sample_rate
        self.seed = seed
        self.model = model
        self.model.seed = seed
        
        self.unlabled_data = unlabled_data


    ## fit the model using pseuod labeling through augmented data
    def fit(self, X, y):
        '''
        Fit the data using pseudo labeling.
        '''
        
        ## Gnerating augmented data by concatenating the tuples
        augmenteddd = self.__create_augmented_train(X, y)
        res = [[ i for i, j in augmenteddd ], [ j for i, j in augmenteddd ]]
        
        ## features of the augmented data
        res[0] = np.array(res[0])

        ## pseudolabels of the augmented data
        res[1] = np.array(res[1])
        print(res[0].shape)
        print(res[1].shape)
        self.model.fit(
            res[0],
            res[1],
            epochs=20
        )
        
        return self


    def __create_augmented_train(self, X, y):
        '''
        Create and return the augmented_train set that consists
        of pseudo-labeled and labeled data.
        '''        
        num_of_samples = int(len(self.unlabled_data) * self.sample_rate)
        
        # Train the model and creat the pseudo-labels
        self.model.fit(X, y, epochs=20)
        pseudo_labels = np.argmax(self.model.predict(self.unlabled_data), axis=1)
        def f(x):
          return [(1 if (i == x) else 0) for i in range(5)]
        pseudo_labels = np.array([f(i) for i in pseudo_labels])
        # Add the pseudo-labels to the test set
        pseudo_data = self.unlabled_data.copy()
        print(pseudo_data.shape)
        print(pseudo_labels.shape)
        pseudo_data = list(zip(pseudo_data,pseudo_labels))
        
        # Take a subset of the test set with pseudo-labels and append in onto
        # the training set
        sampled_pseudo_data = random.sample(pseudo_data, num_of_samples)
        # temp_train = pd.concat([X, y], axis=1)
        temp_train = list(zip(X,y))
        temp_train.extend(sampled_pseudo_data)

        return shuffle(temp_train)

    ## Predict class of input  
    def predict(self, X):
        '''
        Returns the predicted values.
        '''
        return self.model.predict(X)
    
    ## save the trained model
    def save(self, X):
      self.model.save("pseudomodel.h5")

pseudo_model = PseudoLabeler(
    model,
    unlabeled_X,
    sample_rate = 0.3
)

pseudo_model.fit(X_train, y_train)
pseudo_model.save("pseudomodel.h5")