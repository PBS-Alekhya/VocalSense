import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from preprocess import extract_features

dataset_folder = "dataset/age_gender/"
age_gender_labels = os.listdir(dataset_folder)

X, y = [], []
for category in age_gender_labels:
    category_folder = os.path.join(dataset_folder, category)

    if not os.path.isdir(category_folder):  # ✅ Skip non-folder items
        continue

    for file in os.listdir(category_folder):
        file_path = os.path.join(category_folder, file)
        
        if not file.endswith(".wav"):  # ✅ Ignore non-audio files
            continue
        
        features = extract_features(file_path)
        X.append(features)
        y.append(age_gender_labels.index(category))

X, y = np.array(X), np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and Train the Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(age_gender_labels), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))
model.save("models/age_gender_model.h5")

print("✅ Age & Gender recognition model trained & saved!")
