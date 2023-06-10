#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install tensorflow


# In[2]:


pip install keras


# In[3]:


import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


# In[4]:


tf.__version__


# In[5]:


import matplotlib.pyplot as plt


# In[6]:


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('C:\\Users\\Hp\\Downloads\\Projects (Download me But will only work if you have python 3.7_3.8) (1)\\Projects\\datasets\\pokemon\\pika_pika\\training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')


# In[7]:


test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('C:\\Users\\Hp\\Downloads\\Projects (Download me But will only work if you have python 3.7_3.8) (1)\\Projects\\datasets\\pokemon\\pika_pika\\test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


# In[8]:


cnn = tf.keras.models.Sequential()


# In[9]:


cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))


# In[10]:


cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


# In[11]:


cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


# In[12]:


cnn.add(tf.keras.layers.Flatten())


# In[13]:


cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))


# In[14]:


cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# In[15]:


cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[16]:


trained_model = cnn.fit(x = training_set, validation_data = test_set, epochs = 25)


# In[17]:


import numpy as np
from keras.preprocessing import image
test_image = tf.keras.utils.load_img('C:\\Users\\Hp\\Downloads\\Projects (Download me But will only work if you have python 3.7_3.8) (1)\\Projects\\datasets\\pokemon\\pika_pika\\test_set\\pokemon2\\Pikachu_12.png', target_size = (64, 64))
test_image = tf.keras.utils.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if (result[0][0]).all == 1:
  prediction = 'pikachu'
else:
  prediction = 'raichu'


# In[18]:


print(prediction)


# In[19]:


print(result)


# In[ ]:




