# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Set up paths to the dataset
import os

# Directory paths for each class
group_dirs = {
  "class1": "/content/drive/MyDrive/colab/COVID/images_COVID",
  "class2": "/content/drive/MyDrive/colab/Lung_Opacity/images_Lung_Opacity",
  "class3": "/content/drive/MyDrive/colab/Normal/images_Normal",
  "class4": "/content/drive/MyDrive/colab/Viral Pneumonia/images_Viral_Pneumonia"
}

# load data to split
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Directory paths for each class
group_dirs = {
    "class1": "/content/drive/MyDrive/colab/COVID/images_COVID",
    "class2": "/content/drive/MyDrive/colab/Lung_Opacity/images_Lung_Opacity",
    "class3": "/content/drive/MyDrive/colab/Normal/images_Normal",
    "class4": "/content/drive/MyDrive/colab/Viral_Pneumonia/images_Viral_Pneumonia"
}

# Helper function to get all file paths with labels
def get_all_filepaths_and_labels(group_dirs):
    filepaths = []
    labels = []
    for class_name, directory in group_dirs.items():
        for filename in os.listdir(directory):
            if filename.endswith(('png', 'jpg', 'jpeg')):
                filepaths.append(os.path.join(directory, filename))
                labels.append(class_name)
    return filepaths, labels

# Retrieve all file paths and labels
all_filepaths, all_labels = get_all_filepaths_and_labels(group_dirs)

# Split into train, validation, and test sets
train_files, temp_files, train_labels, temp_labels = train_test_split(
    all_filepaths, all_labels, test_size=0.3, stratify=all_labels, random_state=42
)

val_files, test_files, val_labels, test_labels = train_test_split(
    temp_files, temp_labels, test_size=0.33, stratify=temp_labels, random_state=42
)  # 33% of 30% = 10% of total dataset for test

# Print sizes of splits for verification
print(f"Train size: {len(train_files)}, Validation size: {len(val_files)}, Test size: {len(test_files)}")

# Import Libraries
# Step 1: Import Libraries
import tensorflow as tf
from tensorflow.keras import layers, models #building and training neural networks.

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt # Used to visualize training progress (e.g., loss and accuracy).

# Step 2: Build a Basic CNN
def create_cnn(input_shape, num_classes):
    model = models.Sequential()

    # 1st Convolutional Layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    # 2nd Convolutional Layer
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # 3rd Convolutional Layer
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flattening Layer
    model.add(layers.Flatten())

    # Fully Connected Layer
    model.add(layers.Dense(128, activation='relu'))

    # Output Layer
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

# Step 3: Compile the Model
# Define the input shape :dataset consists of 256x256 grayscale (black and white) images
input_shape = (256, 256, 1)
num_classes = 4  # COVID, Lung Opacity, Normal, Viral Pneumonia

# Create and compile the model
model = create_cnn(input_shape, num_classes)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create ImageDataGenerator for training and validation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,     # Normalize pixel values (0-255 to 0-1)
    rotation_range=20,       # Randomly rotate images
    width_shift_range=0.2,   # Randomly shift images horizontally
    height_shift_range=0.2,  # Randomly shift images vertically
    zoom_range=0.2,          # Randomly zoom into images
    horizontal_flip=True     # Flip images horizontally
)

# Training generator
train_generator = train_datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({"filename": train_files, "class": train_labels}),
    x_col="filename",
    y_col="class",
    target_size=(256, 256),  # Adjust to match your dataset's image size
    color_mode="grayscale",
    batch_size=16,
    class_mode="categorical"
)


# Validation generator
val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({"filename": val_files, "class": val_labels}),
    x_col="filename",
    y_col="class",
    target_size=(256, 256),
    color_mode="grayscale",
    batch_size=16,
    class_mode="categorical",
    shuffle=False  # No shuffle for validation data
)

# Step 4: Train the Model
from tensorflow.keras.callbacks import EarlyStopping

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,  # Stop if no improvement for 5 epochs
    restore_best_weights=True  # Revert to the best weights
)

history = model.fit(
    train_generator,  # Ensure the generator has a __len__ method
    validation_data=val_generator,
    epochs=10,
    callbacks=[early_stopping]
)

# Print history details to debug
print("Training complete.")
print("Final training loss:", history.history['loss'][-1])
print("Final validation loss:", history.history['val_loss'][-1])

# Create a test generator
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)  # Only rescale, no augmentation

# Define the test generator
test_generator = test_datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({"filename": test_files, "class": test_labels}),
    x_col="filename",
    y_col="class",
    target_size=(256, 256),  # Match the model's input size
    color_mode="grayscale",  # Match the model's input color mode
    batch_size=16,           # Same batch size as during training
    class_mode="categorical",  # Match the output labels format
    shuffle=False            # Do not shuffle for evaluation
)

# step 5.3 predict on each class
# predict on each class
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Get predictions on the test set
predictions = model.predict(test_generator, steps=test_generator.samples // test_generator.batch_size + 1)
predicted_classes = np.argmax(predictions, axis=1)  # Convert probabilities to class indices

# 2. Get true labels from the generator
true_classes = test_generator.classes  # True class indices
class_labels = list(test_generator.class_indices.keys())  # Get class labels (e.g., 'class1', 'class2', etc.)

# 3. Compute the confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)

# 4. Display the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# 5. Optional: Print a detailed classification report
print("Classification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np

# Step 1: Create the test generator
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)  # Rescale only, no augmentation

test_generator = test_datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({"filename": test_files, "class": test_labels}),
    x_col="filename",
    y_col="class",
    target_size=(256, 256),  # Match model input size
    color_mode="grayscale",  # Match model input
    batch_size=16,
    class_mode="categorical",  # Match the label format
    shuffle=False  # Do not shuffle for consistent evaluation
)

# Step 2: Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Step 3: Get predictions
predictions = model.predict(test_generator, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)  # Convert probabilities to class indices
true_classes = test_generator.classes  # True class indices
class_labels = list(test_generator.class_indices.keys())  # Class labels

# Step 4: Calculate additional metrics
accuracy = accuracy_score(true_classes, predicted_classes)
precision = precision_score(true_classes, predicted_classes, average="weighted")
recall = recall_score(true_classes, predicted_classes, average="weighted")
f1 = f1_score(true_classes, predicted_classes, average="weighted")

# Step 5: Print the results
print("\nTest Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Step 6: Detailed classification report
print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

# Step 6: Visualize Training Progress
# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Apply a cleaner style (seaborn)
sns.set(style="whitegrid")

# Define the number of epochs from history
epochs = range(1, len(history.history['accuracy']) + 1)

# Create a figure with two subplots (accuracy and loss)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot training and validation accuracy
ax1.plot(epochs, history.history['accuracy'], 'o-', label='Training Accuracy', color='firebrick')
ax1.plot(epochs, history.history['val_accuracy'], 'o-', label='Validation Accuracy', color='teal')
ax1.set_title('Training and Validation Accuracy')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.legend(loc='lower right')

# Plot training and validation loss
ax2.plot(epochs, history.history['loss'], 'o-', label='Training Loss', color='firebrick')
ax2.plot(epochs, history.history['val_loss'], 'o-', label='Validation Loss', color='teal')
ax2.set_title('Training and Validation Loss')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend(loc='upper right')

# Adjust spacing and display the plot
plt.tight_layout()
plt.show()


