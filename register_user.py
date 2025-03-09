import os
import shutil
import pyaudio
import wave

dataset_folder = "dataset/speakers/"

def record_audio(filename, duration=5):
    """Records audio from the mic and saves it."""
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 22050
    CHUNK = 1024

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print(f"🎤 Recording for {duration} seconds...")
    frames = [stream.read(CHUNK) for _ in range(0, int(RATE / CHUNK * duration))]

    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(filename, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()
    print(f"✅ Voice sample saved: {filename}")

def register_new_user(user_name):
    """Registers a new user by saving their voice sample."""
    user_folder = os.path.join(dataset_folder, user_name)
    os.makedirs(user_folder, exist_ok=True)

    audio_file = os.path.join(user_folder, "sample.wav")
    record_audio(audio_file, duration=5)

    print(f"✅ User {user_name} registered successfully!")

if __name__ == "__main__":
    user_name = input("Enter your name: ")
    register_new_user(user_name)
