import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from preprocess import extract_features

dataset_folder = "dataset/speakers/"
speakers = os.listdir(dataset_folder)  # Auto-detect speakers

X, y = [], []
for speaker in speakers:
    for file in os.listdir(os.path.join(dataset_folder, speaker)):
        file_path = os.path.join(dataset_folder, speaker, file)
        features = extract_features(file_path)
        X.append(features)
        y.append(speakers.index(speaker))

X, y = np.array(X), np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(speakers), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))
model.save("models/speaker_model.h5")

print("✅ Speaker recognition model trained & saved!")
