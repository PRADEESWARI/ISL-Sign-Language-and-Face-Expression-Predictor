import os
import shutil
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -------------------------------
# STEP 1: CREATE VALIDATION SPLIT
# -------------------------------
base_dir = "fer_dataset"  # Change if your folder name is different
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

if not os.path.exists(val_dir):
    os.makedirs(val_dir, exist_ok=True)

    split_ratio = 0.15  # 15% of data for validation

    print("📂 Creating validation split...")
    for class_name in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        val_class_path = os.path.join(val_dir, class_name)
        os.makedirs(val_class_path, exist_ok=True)

        images = os.listdir(class_path)
        val_count = int(len(images) * split_ratio)
        val_images = random.sample(images, val_count)

        for img in val_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(val_class_path, img)
            shutil.move(src, dst)

    print("✅ Validation set created successfully!")
else:
    print("✅ Validation set already exists. Skipping split.")

# -------------------------------
# STEP 2: DATA AUGMENTATION
# -------------------------------
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=10, zoom_range=0.1, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    os.path.join(base_dir, "train"),
    target_size=(48, 48),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=64
)

val_gen = val_datagen.flow_from_directory(
    os.path.join(base_dir, "val"),
    target_size=(48, 48),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=64
)

test_gen = test_datagen.flow_from_directory(
    os.path.join(base_dir, "test"),
    target_size=(48, 48),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=64
)

# -------------------------------
# STEP 3: CNN MODEL ARCHITECTURE
# -------------------------------
model = Sequential([
    Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(48, 48, 1)),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Conv2D(256, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 emotions
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -------------------------------
# STEP 4: TRAIN THE MODEL
# -------------------------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30
)

# -------------------------------
# STEP 5: SAVE THE MODEL
# -------------------------------
model.save("face_cnn_model.h5")
print("✅ Facial Expression Model Saved as face_cnn_model.h5")
