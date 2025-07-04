import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Use the same parameters as in binary_training.py
batch_size = 32
img_size = (224, 224)
train_data_dir = "C:/Users/bonzi/Desktop/FYP_Project_TF/binary_splitdata/train"

# Load training dataset (only for representative data)
train_ds = image_dataset_from_directory(
    train_data_dir,
    image_size=img_size,
    batch_size=batch_size,
    color_mode="grayscale",  # Original images are grayscale.
    shuffle=True
)

# Preprocessing functions (same as in binary_training.py)
def convert_grayscale_to_rgb(image, label):
    image = tf.image.grayscale_to_rgb(image)
    return image, label

def normalize(image, label):
    image = preprocess_input(image)
    return image, label

# Apply the same transformations as in your training pipeline
train_ds = train_ds.map(convert_grayscale_to_rgb)
train_ds = train_ds.map(normalize)
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# Define a representative dataset generator that yields already preprocessed images.
def representative_data_gen():
    for images, _ in train_ds.take(100):
        # The images here are already in the same float32 format as required by your model
        yield [images]

# Load your trained Keras model
model = tf.keras.models.load_model("final_model_binary.keras")

# Set up the TFLite converter for full integer quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

# Quantize all operations to int8
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Specify that inputs and outputs should be int8
converter.inference_input_type  = tf.int8
converter.inference_output_type = tf.int8

# Convert the model
tflite_quant_model = converter.convert()

# Save the quantized model
with open("final_model_binary_quantized.tflite", "wb") as f:
    f.write(tflite_quant_model)

print("Full integer quantization complete. Model saved as final_model_binary_quantized.tflite")