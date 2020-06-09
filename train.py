#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from keras.layers import Convolution2D,MaxPooling2D,Flatten,Dense 
from keras.models import Sequential
from random import randint


# In[ ]:


model = Sequential()


# In[ ]:

r= randint(0,5)
if r == 1 :
    model.add(Convolution2D(filters=64, 
                        kernel_size=(3,3), 
                        activation='relu',
                   input_shape=(64, 64, 3)
                       ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                       ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
if r == 2 :
    model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                   input_shape=(64, 64, 3)
                       ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                       ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
if r == 3 :
    model.add(Convolution2D(filters=32, 
                        kernel_size=(5,5), 
                        activation='relu',
                   input_shape=(64, 64, 3)
                       ))
    model.add(Convolution2D(filters=64, 
                        kernel_size=(3,3), 
                        activation='relu',
                       ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
if r == 4 :
    model.add(Convolution2D(filters=32, 
                        kernel_size=(5,5), 
                        activation='relu',
                   input_shape=(64, 64, 3)
                       ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                       ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                       ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
else :
    model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                   input_shape=(64, 64, 3)
                       ))
    model.add(Convolution2D(filters=64, 
                        kernel_size=(3,3), 
                        activation='relu',
                       ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                   input_shape=(64, 64, 3)
                       ))
    model.add(MaxPooling2D(pool_size=(2, 2)))


# In[ ]:


model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


r=randint(1,3)
from keras_preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        'cnn_dataset/training_set/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        'cnn_dataset/test_set/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
model.fit(
        training_set,
        steps_per_epoch=8000,
        epochs=r,
        validation_data=test_set,
        validation_steps=800)


# In[ ]:

print((max(model.history.history['accuracy'])))
if (max(model.history.history['accuracy'])) > 0.90:
     model.save('my.h5')
file=open('/root/project/accuracy.txt',"w+")
file.write(str(max(model.history.history['accuracy'])))
file.close()
    
