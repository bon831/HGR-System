import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm.keras import TqdmCallback  
import seaborn as sns  # Import seaborn for enhanced visualization

# Hyperparameters and config
# batch_size = 16
batch_size = 32
num_epochs = 200
# learning_rate = 0.01
learning_rate = 0.001
patience = 10
model_save_path = "final_model32_binary.keras"  # Save best model in Keras model format
result_dir = "C:/Users/bonzi/Desktop/FYP_Project_TF/binary_result_v2"  # Directory to save results
os.makedirs(result_dir, exist_ok=True)  # Create directory if it doesn't exist

# Check available device(s) - TensorFlow automatically uses GPU if available
print("Using devices:", tf.config.list_physical_devices('GPU'))

# New paths to the pre-split dataset directories
# Folder structure:
# binary_split/
#    train/        <-- contains subfolders for each class
#    validation/   <-- contains subfolders for each class
base_split_dir = "C:/Users/bonzi/Desktop/FYP_Project_TF/binary_splitdata"
train_data_dir = os.path.join(base_split_dir, "train")
val_data_dir   = os.path.join(base_split_dir, "validation")

# Image size expected by MobileNetV2
img_size = (224, 224)

# Load training dataset from the "train" subfolder
train_ds = image_dataset_from_directory(
    train_data_dir,
    image_size=img_size,
    batch_size=batch_size,
    color_mode="grayscale",  # Original images are grayscale.
    shuffle=True  # Shuffle training data
)

# Load validation dataset from the "validation" subfolder
val_ds = image_dataset_from_directory(
    val_data_dir,
    image_size=img_size,
    batch_size=batch_size,
    color_mode="grayscale",
    shuffle=False  # Disable shuffling for validation
)

# Get number of classes from the training directory structure (each subfolder is a class)
class_names = train_ds.class_names
num_classes = len(class_names)
print("Class names:", class_names, "Number of classes:", num_classes)

# Convert grayscale images to 3 channels for MobileNetV2 compatibility.
def convert_grayscale_to_rgb(image, label):
    image = tf.image.grayscale_to_rgb(image)
    return image, label

train_ds = train_ds.map(convert_grayscale_to_rgb)
val_ds = val_ds.map(convert_grayscale_to_rgb)

# Normalize images using preprocess_input from MobileNetV2
def normalize(image, label):
    image = preprocess_input(image)
    return image, label

train_ds = train_ds.map(normalize)
val_ds = val_ds.map(normalize)

# Prefetch datasets for performance improvement
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# Build the model using MobileNetV2 as the base model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))
base_model.trainable = False  # Freeze the base model for transfer learning

# Add global pooling and a final dense layer for classification
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Custom callback to log when a new best model is saved.
class BestModelLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super(BestModelLogger, self).__init__()
        self.best_val_loss = float('inf')
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get("val_loss")
        if current_val_loss is not None and current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.best_epoch = epoch + 1  # Epochs are zero-indexed.
            print(f"\nEpoch {self.best_epoch}: New best model saved with val_loss = {self.best_val_loss:.4f}")

# Define callbacks: early stopping, model checkpointing, training progress, and best model logging.
callback_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
    callbacks.ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True),
    TqdmCallback(verbose=1),  # Adds progress bar during training.
    BestModelLogger()
]

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=num_epochs,
    callbacks=callback_list
)

# Final evaluation on the validation set
val_loss, val_acc = model.evaluate(val_ds)
print(f"\nFinal Validation Loss: {val_loss:.4f}, Final Validation Accuracy: {val_acc * 100:.2f}%")

# Obtain predictions and true labels from the validation set for the classification report
all_preds = []
all_labels = []
for images, labels in val_ds:
    preds = model.predict(images)
    all_preds.extend(np.argmax(preds, axis=1))
    all_labels.extend(labels.numpy())

report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Plot the classification report as a table
fig, ax = plt.subplots(figsize=(12, len(report_df) * 0.4 + 1))
ax.axis('off')
tbl = ax.table(cellText=report_df.round(2).values,
               colLabels=report_df.columns,
               rowLabels=report_df.index,
               cellLoc='center',
               loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.2)
plt.suptitle("Binary Thresholding Dataset", fontsize=16, fontweight='bold')
plt.title("Classification Report")
plt.tight_layout(rect=[0, 0, 1, 0.95])
report_img_path = os.path.join(result_dir, "classification_report_table_tf.png")
plt.savefig(report_img_path, bbox_inches='tight')
plt.show()

# # Plot and save the confusion matrix
cm = confusion_matrix(all_labels, all_preds)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
# plt.figure(figsize=(10, 7))
# disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
# plt.suptitle("Binary Thresholding Dataset", fontsize=16, fontweight='bold')
# plt.title("Confusion Matrix")
# plt.tight_layout(rect=[0, 0, 1, 0.95])
# cm_path = os.path.join(result_dir, "confusion_matrix_tf.png")
# plt.savefig(cm_path)
# plt.show()

# ---------------------------
# Confusion Matrix for 12-Class Dataset with Enhanced Layout
# ---------------------------
plt.figure(figsize=(16, 14))  # Increase the figure size for better layout
ax = sns.heatmap(cm, 
                 annot=True, 
                 fmt='d', 
                 cmap='Blues',
                 xticklabels=class_names, 
                 yticklabels=class_names,
                 annot_kws={"size": 14})  # increase annotation size

plt.xlabel("Predicted Class", fontsize=16, fontweight="bold")
plt.ylabel("True Class", fontsize=16, fontweight="bold")
plt.title("Confusion Matrix", fontsize=18, fontweight="bold")
plt.xticks(rotation=45, ha="right", fontsize=14)  # Rotate x-axis labels for clarity
plt.yticks(rotation=0, fontsize=14)
plt.tight_layout()

cm_path = os.path.join(result_dir, "confusion_matrix_tf.png")
plt.savefig(cm_path, bbox_inches="tight")
plt.show()

# Plot Accuracy & Loss over epochs
epochs_range = range(1, len(history.history['loss']) + 1)
plt.figure(figsize=(12, 5))
plt.suptitle("Binary Thresholding Dataset", fontsize=16, fontweight='bold')

plt.subplot(1, 2, 1)
plt.plot(epochs_range, history.history['accuracy'], label='Training Accuracy')
plt.plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title("Accuracy Over Epochs")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, history.history['loss'], label='Training Loss')
plt.plot(epochs_range, history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title("Loss Over Epochs")

plt.tight_layout(rect=[0, 0, 1, 0.95])
training_plots_path = os.path.join(result_dir, "training_plots_tf.png")
plt.savefig(training_plots_path)
plt.show()

# Check unique labels in one batch of the validation dataset
for images, labels in val_ds.take(1):
    print("Unique validation labels:", np.unique(labels.numpy()))

print("Training complete.")