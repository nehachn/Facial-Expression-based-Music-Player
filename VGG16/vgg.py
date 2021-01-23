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
from keras.utils import plot_model
from keras.applications import VGG16

## Callback function to stop learning when converged
class Callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>0.99):
            self.model.stop_training = True
            
callback = Callback()

# Taking pre-trained base model as VGG16 
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
for layer in base_model.layers[:-4]:
    layer.trainable = False

## Add dense ReLu and Softmax layer, Dropout for Regularization to the pre-trained model 
def define_model(base_model, num_cat):
    inputs1 = Input(shape=(None, None, 3,))
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))
    print(model.summary())
    # plot_model(model, to_file='mode.png', show_shapes=True)
    return model
model = define_model(base_model, 8)

## Compiling the model using Categorical-Crossentropy loss function and Adam optimizer
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

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
                    batch_size=64,
                    target_size=(48,48),
                    class_mode='categorical')

## Generate validation / test data
validation = ImageDataGenerator(rescale=1.0/255)

validation_generator = validation.flow_from_directory(
                        './archive/test',
                        batch_size=64,
                        target_size=(48,48),
                        class_mode='categorical')

## Model fitting
history = model.fit_generator(train_generator,
                              epochs=20,
                              steps_per_epoch=24176//64,
                              verbose=1,
                              validation_data=validation_generator,
                             validation_steps=6043//64,
                             callbacks=[callback])

## Save the generated model
model.save("vgg_predictions.h5")

