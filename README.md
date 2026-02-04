# Groundnut Leaf Disease Detection Using Deep Learning 🌱

This project detects diseases in groundnut (peanut) leaves using Deep Learning (CNN).  
It helps farmers identify leaf diseases automatically from images.

## 📌 Features
- Classifies groundnut leaf images into healthy and diseased categories
- Uses Convolutional Neural Network (CNN)
- Easy to train and test
- Can be extended to mobile or web apps

---

## 📂 Dataset Structure

Your dataset should look like this:

dataset/
│
├── train/
│   ├── Healthy/
│   ├── Leaf_Spot/
│   ├── Rust/
│
├── test/
│   ├── Healthy/
│   ├── Leaf_Spot/
│   ├── Rust/

---

## 🧠 Model Used
- CNN (Convolutional Neural Network)
- Framework: TensorFlow & Keras

---

## 🛠️ Requirements

Install dependencies using:

```bash
pip install tensorflow matplotlib numpy

git clone https://github.com/yourusername/Groundnut_Leaf_Disease_Detection_Using_DL.git

cd Groundnut_Leaf_Disease_Detection_Using_DL

python train_model.py

📊 Output

Accuracy & loss graph

Saved model: groundnut_model.h5

🚀 Future Scope

Mobile app integration

More disease classes

Real-time detection using camera

👨‍💻 Author

Soubhagya Ranjan Mishra

---

# 🧪 2. Full Code File: `train_model.py`

Copy this and save as **train_model.py**

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# =========================
# Paths
# =========================
train_dir = "dataset/train"
test_dir = "dataset/test"

# =========================
# Image Generator
# =========================
train_gen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(128,128),
    batch_size=32,
    class_mode="categorical"
)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=(128,128),
    batch_size=32,
    class_mode="categorical"
)

# =========================
# CNN Model
# =========================
model = Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(train_data.num_classes,activation='softmax'))

# =========================
# Compile
# =========================
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# Train
# =========================
history = model.fit(
    train_data,
    epochs=10,
    validation_data=test_data
)

# =========================
# Save Model
# =========================
model.save("groundnut_model.h5")

# =========================
# Plot Accuracy & Loss
# =========================
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'],label='Train Accuracy')
plt.plot(history.history['val_accuracy'],label='Val Accuracy')
plt.legend()
plt.title("Accuracy")

plt.subplot(1,2,2)
plt.plot(history.history['loss'],label='Train Loss')
plt.plot(history.history['val_loss'],label='Val Loss')
plt.legend()
plt.title("Loss")

plt.show()

📁 Final Repo Structure

Groundnut_Leaf_Disease_Detection_Using_DL/
│
├── dataset/
│   ├── train/
│   └── test/
│
├── train_model.py
├── README.md
└── groundnut_model.h5 (after training)
