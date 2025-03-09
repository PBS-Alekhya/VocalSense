import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from preprocess import extract_features

dataset_folder = "dataset/emotions/"
emotion_labels = os.listdir(dataset_folder)

X, y = [], []

for emotion in emotion_labels:
    emotion_folder = os.path.join(dataset_folder, emotion)

    if not os.path.isdir(emotion_folder):
        continue

    for file in os.listdir(emotion_folder):
        file_path = os.path.join(emotion_folder, file)

        if not file.endswith(".wav"):
            continue
        
        features = extract_features(file_path)

        if features is None:
            continue

        X.append(features)
        y.append(emotion_labels.index(emotion))

if len(X) == 0:
    print("\n Error: No valid audio files found! Check your dataset.")
    exit()

MAX_SAMPLES = 500  

X, y = np.array(X), np.array(y)

if len(X) > MAX_SAMPLES:
    X, y = X[:MAX_SAMPLES], y[:MAX_SAMPLES]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(emotion_labels), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

X_train = X_train[..., np.newaxis]  # Reshape for CNN
X_test = X_test[..., np.newaxis]

model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))  # Reduce epochs

model.save("models/emotion_model.h5")

print("\n✅ Emotion recognition model trained & saved!")
