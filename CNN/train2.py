import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import PIL
from PIL import Image
from tensorflow.keras.preprocessing import image
import numpy as np

class Callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>0.99):
            self.model.stop_training = True
            
callback = Callback()

model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(48,48,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
])

model.compile(optimizer = RMSprop(lr=0.001), loss = 'categorical_crossentropy', metrics = ['acc'])

tf.keras.utils.plot_model(
    model,
    to_file="model.png",
    show_shapes=False,
    show_dtype=False,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=96,
)

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

validation = ImageDataGenerator(rescale=1.0/255)

validation_generator = validation.flow_from_directory(
                        './archive/test',
                        batch_size=64,
                        target_size=(48,48),
                        class_mode='categorical')

history = model.fit_generator(train_generator,
                              epochs=50,
                              steps_per_epoch=24176//64,
                              verbose=1,
                              validation_data=validation_generator,
                              validation_steps=6043//64,
                              callbacks=[callback])

model.save('prediction2_new_model.h5')