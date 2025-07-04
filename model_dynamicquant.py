import tensorflow as tf

# Load your trained Keras model
model = tf.keras.models.load_model("final_model_binary.keras")

# Set up the TFLite converter for dynamic range quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# This optimization flag automatically applies dynamic range quantization for compatible ops.
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# No representative dataset is needed for dynamic quantization
tflite_dynamic_model = converter.convert()

# Save the dynamically quantized model
with open("final_model_dynamic_quantized.tflite", "wb") as f:
    f.write(tflite_dynamic_model)

print("Dynamic range quantization complete. Model saved as final_model_dynamic_quantized.tflite")