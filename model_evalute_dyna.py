import os
import time
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.metrics import confusion_matrix, classification_report

# Create a directory for saving evaluation result images
results_folder = "C:/Users/bonzi/Desktop/FYP_Project_TF/final_model_evaluate"
os.makedirs(results_folder, exist_ok=True)

# ---------------------------
# Parameters and Directories
# ---------------------------
batch_size = 32
img_size = (224, 224)
val_data_dir = "C:/Users/bonzi/Desktop/FYP_Project_TF/binary_splitdata/validation"

# Paths for models
original_model_path = "final_model_binary.keras"                # Saved original Keras model
dynamic_quant_model_path = "final_model_dynamic_quantized.tflite"  # Dynamic quantized TFLite model

# ---------------------------
# Load and Preprocess Validation Dataset
# ---------------------------
val_ds = image_dataset_from_directory(
    val_data_dir,
    image_size=img_size,
    batch_size=batch_size,
    color_mode="grayscale",  # Original images are grayscale.
    shuffle=False
)

def convert_grayscale_to_rgb(image, label):
    image = tf.image.grayscale_to_rgb(image)
    return image, label

def normalize(image, label):
    image = preprocess_input(image)
    return image, label

val_ds = val_ds.map(convert_grayscale_to_rgb)
val_ds = val_ds.map(normalize)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# ---------------------------
# Evaluate Original Model
# ---------------------------
print("Evaluating Original Model...")
model_orig = tf.keras.models.load_model(original_model_path)
loss_orig, acc_orig = model_orig.evaluate(val_ds, verbose=0)

start_time = time.time()
preds_orig_prob = model_orig.predict(val_ds)
end_time = time.time()
inference_time_orig = (end_time - start_time) / len(val_ds)  # average inference time per batch

# Get original model file size in MB
size_orig = os.path.getsize(original_model_path) / (1024 * 1024)

# Collect true labels and predictions for original model
all_labels_orig = []
for _, labels in val_ds:
    all_labels_orig.extend(labels.numpy())
all_labels_orig = np.array(all_labels_orig)
all_preds_orig = np.argmax(preds_orig_prob, axis=1)

# ---------------------------
# Evaluate Dynamic Quantized Model using TFLite Interpreter
# ---------------------------
print("Evaluating Dynamic Quantized Model...")
# Load the TFLite dynamic quantized model.
interpreter = tf.lite.Interpreter(model_path=dynamic_quant_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

loss_fn = SparseCategoricalCrossentropy()
all_losses_dyn = []  # losses for dynamic quant model
inference_times_dyn = []
correct_dyn = 0
total_dyn = 0

# For dynamic quantization, the inputs usually remain float32.
# We'll process entire batches directly without converting to int8.
all_preds_dyn = []
all_labels_dyn = []

for images, labels in val_ds:
    for i in range(images.shape[0]):
        t0 = time.time()
        # Process one image at a time to match the expected input shape [1, height, width, channels]
        img = images[i:i+1].numpy()
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        t1 = time.time()
        inference_times_dyn.append(t1 - t0)
        
        pred = np.argmax(output, axis=1)[0]
        all_preds_dyn.append(pred)
        all_labels_dyn.append(labels.numpy()[i])
        
        correct_dyn += (pred == labels.numpy()[i])
        total_dyn += 1
        
        # Compute loss for each image for dynamic quantized model.
        batch_loss = loss_fn(labels[i:i+1], output)
        all_losses_dyn.append(batch_loss.numpy())

acc_dyn = correct_dyn / total_dyn
loss_dyn = np.mean(all_losses_dyn)
inference_time_dyn = np.mean(inference_times_dyn)  # average inference time per batch

# Get dynamic quantized model file size in MB
size_dyn = os.path.getsize(dynamic_quant_model_path) / (1024 * 1024)

# ---------------------------
# Prepare Summary Table
# ---------------------------
summary_df = pd.DataFrame({
    "Model": ["Original", "Dynamic Quantized"],
    "Accuracy (%)": [round(acc_orig * 100, 2), round(acc_dyn * 100, 2)],
    "Loss": [round(loss_orig, 4), round(loss_dyn, 4)],
    "Avg Inference Time (s/batch)": [round(inference_time_orig, 4), round(inference_time_dyn, 4)],
    "Model Size (MB)": [round(size_orig, 2), round(size_dyn, 2)]
})

# Convert numeric columns to strings with the desired formatting
summary_df["Accuracy (%)"] = summary_df["Accuracy (%)"].apply(lambda x: f"{x:.2f}")
summary_df["Loss"] = summary_df["Loss"].apply(lambda x: f"{x:.4f}")
summary_df["Avg Inference Time (s/batch)"] = summary_df["Avg Inference Time (s/batch)"].apply(lambda x: f"{x:.4f}")
summary_df["Model Size (MB)"] = summary_df["Model Size (MB)"].apply(lambda x: f"{x:.2f}")

print(summary_df)

fig, ax = plt.subplots(figsize=(12, 2))
ax.axis('tight')
ax.axis('off')
tbl = ax.table(cellText=summary_df.values,
               colLabels=summary_df.columns,
               loc="center",
               cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.2)
plt.tight_layout()
summary_path = os.path.join(results_folder, "final_model_comparison_summary_dyna.png")
plt.savefig(summary_path, bbox_inches="tight")
plt.show()

# ---------------------------
# Confusion Matrix and Classification Report for Original Model
# ---------------------------
class_names = sorted(os.listdir(val_data_dir))

cm_orig = confusion_matrix(all_labels_orig, all_preds_orig)
report_orig = classification_report(all_labels_orig, all_preds_orig, target_names=class_names, output_dict=True)
report_df_orig = pd.DataFrame(report_orig).transpose()

# plt.figure(figsize=(8, 6))
# sns.heatmap(cm_orig, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.title("Confusion Matrix: Original Model")
# plt.tight_layout()
# cm_orig_path = os.path.join(results_folder, "confusion_matrix_original_dyna.png")
# plt.savefig(cm_orig_path, bbox_inches="tight")
# plt.show()

plt.figure(figsize=(14, 12))  # Increase figure size for 12 classes
ax = sns.heatmap(cm_orig, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 12})
plt.xlabel("Predicted Class", fontsize=14, fontweight="bold")
plt.ylabel("True Class", fontsize=14, fontweight="bold")
plt.title("Confusion Matrix: Original Model", fontsize=16, fontweight="bold")
plt.xticks(rotation=45, ha="right", fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.tight_layout()
cm_orig_path = os.path.join(results_folder, "confusion_matrix_original_dyna.png")
plt.savefig(cm_orig_path, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(10, len(report_df_orig) * 0.5 + 1))
ax.axis('tight')
ax.axis('off')
tbl = ax.table(cellText=report_df_orig.round(2).values,
               colLabels=report_df_orig.columns,
               rowLabels=report_df_orig.index,
               cellLoc='center',
               loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
plt.title("Classification Report: Original Model")
plt.tight_layout()
cr_orig_path = os.path.join(results_folder, "classification_report_originaldyna.png")
plt.savefig(cr_orig_path, bbox_inches="tight")
plt.show()

# ---------------------------
# Confusion Matrix and Classification Report for Dynamic Quantized Model
# ---------------------------
all_preds_dyn = np.array(all_preds_dyn)
all_labels_dyn = np.array(all_labels_dyn)

cm_dyn = confusion_matrix(all_labels_dyn, all_preds_dyn)
report_dyn = classification_report(all_labels_dyn, all_preds_dyn, target_names=class_names, output_dict=True)
report_df_dyn = pd.DataFrame(report_dyn).transpose()

# plt.figure(figsize=(8, 6))
# sns.heatmap(cm_dyn, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.title("Confusion Matrix: Dynamic Quantized Model")
# plt.tight_layout()
# cm_dyn_path = os.path.join(results_folder, "confusion_matrix_dynamic_quantized.png")
# plt.savefig(cm_dyn_path, bbox_inches="tight")
# plt.show()

plt.figure(figsize=(14, 12))  # Increase figure size for 12 classes
ax = sns.heatmap(cm_dyn, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 12})
plt.xlabel("Predicted Class", fontsize=14, fontweight="bold")
plt.ylabel("True Class", fontsize=14, fontweight="bold")
plt.title("Confusion Matrix: Dynamic Quantized Model", fontsize=16, fontweight="bold")
plt.xticks(rotation=45, ha="right", fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.tight_layout()
cm_dyn_path = os.path.join(results_folder, "confusion_matrix_original_elegant.png")
plt.savefig(cm_dyn_path, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(10, len(report_df_dyn) * 0.5 + 1))
ax.axis('tight')
ax.axis('off')
tbl = ax.table(cellText=report_df_dyn.round(2).values,
               colLabels=report_df_dyn.columns,
               rowLabels=report_df_dyn.index,
               cellLoc='center',
               loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
plt.title("Classification Report: Dynamic Quantized Model")
plt.tight_layout()
cr_dyn_path = os.path.join(results_folder, "classification_report_dynamic_quantized.png")
plt.savefig(cr_dyn_path, bbox_inches="tight")
plt.show()


