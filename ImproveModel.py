import os
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Set your data directory
data_dir = './dataset2'

# Hyperparameters
batch_size = 32
epochs = 6  # Change this to match the epochs in ModetTrain.py
patience = 10

# Load the saved model
model = load_model('asl_cnn_model.h5')

# Prepare the data generators
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(data_dir, 'train'),
    target_size=(64, 64),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    os.path.join(data_dir, 'test'),
    target_size=(64, 64),
    batch_size=batch_size,
    class_mode='categorical')

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='min', restore_best_weights=True)

# Add a learning rate scheduler
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)  # Add this line

# Continue training the model
history = model.fit(train_generator, epochs=epochs, validation_data=test_generator, callbacks=[early_stopping, reduce_lr])  # Add reduce_lr to callbacks list

# Save the model
model.save('asl_cnn_model.h5')

# Evaluate the model on the test set
scores = model.evaluate(test_generator)

print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
