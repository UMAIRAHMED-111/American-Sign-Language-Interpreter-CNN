import numpy as np
import json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

def load_and_preprocess_image(image_path, img_width, img_height):
    img = image.load_img(image_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Load the class labels
with open('class_labels.json', 'r') as f:
    class_labels = json.load(f)

# Set the path to your ASL image
image_path = './C.jpg'

# Load the trained model
model = load_model('asl_cnn_model.h5')

# Load and preprocess the image
img_width, img_height = 64, 64  # Change the dimensions to match the training input shape
img_array = load_and_preprocess_image(image_path, img_width, img_height)

# Make a prediction
prediction = model.predict(img_array)
predicted_class_index = np.argmax(prediction)

# Get the class label from the index
predicted_class_label = list(class_labels.keys())[list(class_labels.values()).index(predicted_class_index)]

print(f"Predicted class index: {predicted_class_index}")
print(f"Predicted class label: {predicted_class_label}")




