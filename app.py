from flask import Flask, render_template, jsonify, request
import numpy as np
import tensorflow as tf
import librosa
import pyaudio
import wave
import os
from preprocess import extract_features
from register_user import register_new_user

app = Flask(__name__)

# Load Models
speaker_model = tf.keras.models.load_model("models/speaker_model.h5")
emotion_model = tf.keras.models.load_model("models/emotion_model.h5")
age_gender_model = tf.keras.models.load_model("models/age_gender_model.h5")

# Speaker list
speakers = os.listdir("dataset/speakers/")
emotions = ["happy", "sad", "angry", "neutral"]
age_gender_labels = ["male_20", "female_20", "male_35", "female_35", "male_50", "female_50"]

# Mic recording settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050
CHUNK = 1024
RECORD_SECONDS = 4
WAVE_OUTPUT_FILENAME = "static/live_audio.wav"

def record_audio():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    frames = [stream.read(CHUNK) for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS))]

    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET"])
def predict():
    record_audio()
    features = extract_features(WAVE_OUTPUT_FILENAME).reshape(1, -1)

    speaker_prediction = speaker_model.predict(features)
    speaker_index = np.argmax(speaker_prediction)
    
    if np.max(speaker_prediction) > 0.8:
        speaker_name = speakers[speaker_index]
        greeting = f"Hello, {speaker_name}!"
    else:
        speaker_name = "Unknown User"
        greeting = "Unknown User. Want to register?"

    emotion_prediction = emotion_model.predict(features)
    emotion_detected = emotions[np.argmax(emotion_prediction)]

    age_gender_prediction = age_gender_model.predict(features)
    age_gender_detected = age_gender_labels[np.argmax(age_gender_prediction)]

    return jsonify({
        "greeting": greeting,
        "speaker": speaker_name,
        "emotion": emotion_detected,
        "age_gender": age_gender_detected
    })

@app.route("/register", methods=["POST"])
def register():
    user_name = request.form["name"]
    record_audio()
    register_new_user(user_name, WAVE_OUTPUT_FILENAME)
    return jsonify({"message": f"User {user_name} registered successfully!"})

if __name__ == "__main__":
    app.run(debug=True)
