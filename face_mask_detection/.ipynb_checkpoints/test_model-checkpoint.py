import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Fungsi untuk membaca dan memproses dataset
def load_dataset(dataset_path):
    images = []
    labels = []

    for folder in os.listdir(dataset_path):
        if folder == 'with_mask':
            label = 1
        elif folder == 'without_mask':
            label = 0
        else:
            continue

        folder_path = os.path.join(dataset_path, folder)

        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (64, 64))  # Sesuaikan ukuran gambar sesuai kebutuhan
            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)

# Memuat dataset
dataset_path = '/dataset'
images, labels = load_dataset(dataset_path)

# Membagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Normalisasi data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Konversi label ke bentuk one-hot encoding
y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)

# Membangun model CNN sederhana
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Kompilasi model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Melatih model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Simpan model
model.save('face_mask_detection_model.h5')
