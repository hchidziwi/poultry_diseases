import numpy as np
import itertools
import os
import shutil
import csv
import random
import glob
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, Dropout, BatchNormalization, Conv2D, MaxPool2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from sklearn.metrics import confusion_matrix
from tensorflow.keras.applications import MobileNetV2, densenet


#dataset paths
train_path = '../data/train6/train'
valid_path = '../data/train6/valid'
test_path = '../data/train6/test'

#load data
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input, vertical_flip=True, rotation_range=45, rescale=1./255) \
    .flow_from_directory(directory=train_path, target_size=(224,224), classes=['cocci', 'healthy', 'salmo', 'ncd'], batch_size=64, shuffle=True)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input,  vertical_flip=True, rotation_range=45, rescale=1./255) \
    .flow_from_directory(directory=valid_path, target_size=(224,224), classes=['cocci', 'healthy', 'salmo', 'ncd'], batch_size=64, shuffle=False)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input, vertical_flip=True, rotation_range=45, rescale=1./255) \
    .flow_from_directory(directory=test_path, target_size=(224,224), classes=['cocci', 'healthy', 'salmo', 'ncd'], batch_size=64, shuffle=False)


imgs, labels = next(train_batches)

#load model without classifier layers
model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))

#make layers untrainable
for layer in model.layers:
    layer.trainable = False

#build a model on top od densenet
def build_model(model):
    x=model.output
    x=MaxPool2D(pool_size=(2, 2), strides=2)(x)
    x=Dropout(0.2, input_shape=(224,224,3))(x)
    x=Flatten()(x)
    preds=Dense(4,activation='softmax')(x) #final layer with softmax activation

    model=Model(inputs=model.input,outputs=preds)

    return model

model = build_model(model)

#setting early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

#compile the network
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

#train the model
history = model.fit(x=train_batches,
    steps_per_epoch=len(train_batches),
    validation_data=valid_batches,
    validation_steps=len(valid_batches),
    callbacks=early_stopping,
    epochs=100,
    verbose=2
)

# save model
model.save('MobileNetV2_multiclass_image_classification.h5')
print('Model Saved!')

# Save training history to a CSV file
csv_file_path = "../results/MobileNetV2_multiclass_training_history.csv"

with open(csv_file_path, mode='w', newline='') as csv_file:
    fieldnames = ['epoch', 'loss', 'accuracy', 'val_loss', 'val_accuracy']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    for epoch, metrics in enumerate(history.history['loss']):
        writer.writerow({
            'epoch': epoch + 1,
            'loss': metrics,
            'accuracy': history.history['accuracy'][epoch],
            'val_loss': history.history['val_loss'][epoch],
            'val_accuracy': history.history['val_accuracy'][epoch]
        })

print("Training history saved to", csv_file_path)


