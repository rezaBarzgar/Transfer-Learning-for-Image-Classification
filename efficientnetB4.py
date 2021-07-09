import os
import tensorflow as tf
import numpy as np
from PIL import Image
from matplotlib import cm
from numpy import save
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications.vgg16 import VGG16
import tensorflow.keras.applications.efficientnet as efn
from tensorflow.keras.applications import ResNet50
from tensorflow.python.keras import layers

base_dir = 'D:\projects\CNN\dataset'
train_dir = os.path.join(base_dir,'train/images').replace("\\","/")
validation_dir = os.path.join(base_dir,'test/images').replace("\\","/")
batch_size = 32

# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1.0,horizontal_flip=True)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator( rescale = 1.0)

# Flow training images in batches of 1 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir, class_mode = 'categorical', target_size = (380, 380),batch_size=batch_size,shuffle=False)

# Flow validation images in batches of 1 using test_datagen generator
validation_generator = test_datagen.flow_from_directory(validation_dir, class_mode = 'categorical', target_size = (380, 380),batch_size=batch_size,shuffle=False)

base_model = efn.EfficientNetB4(input_shape = (380, 380, 3), include_top = False, weights = 'imagenet')

for layer in base_model.layers:
    layer.trainable = False

dog_features = np.zeros(shape=(1,12,12,1792))
features = np.zeros(shape=(532,12,12,1792))
labels = np.zeros(shape=(532,19))
i = 0

#genetic make features
# dog_features = base_model.predict()

for input_batch , label_batch in train_generator:
    features_batch = base_model.predict(input_batch)
    features[i* batch_size:(i+1)*batch_size] = features_batch
    labels[i* batch_size:(i+1)*batch_size] = label_batch
    i += 1
    if i*batch_size > 532:
        break
save('train_features2.npy',features)
save('train_labels2.npy',labels)
features = np.zeros(shape=(380,12,12,1792))
labels = np.zeros(shape=(380,19))
i = 0
for input_batch , label_batch in validation_generator:
    features_batch = base_model.predict(input_batch)
    features[i* batch_size:(i+1)*batch_size] = features_batch
    labels[i* batch_size:(i+1)*batch_size] = label_batch
    i += 1
    if i*batch_size > 380:
        break
save('test_features2.npy',features)
save('test_labels2.npy',labels)
#-----------------------------------------------------------------
# #Flatten the output layer to 1 dimension
# x = layers.Flatten()(base_model.output)
#
# # Add a fully connected layer with 512 hidden units and ReLU activation
# x = layers.Dense(19, activation='softmax')(x)
#
# model = tf.keras.models.Model(inputs=base_model.input,outputs= x)
#
# model.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])
#
# vgghist = model.fit(train_generator, validation_data = validation_generator,epochs = 10)
