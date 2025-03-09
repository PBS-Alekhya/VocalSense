from pydub import AudioSegment
import os

dataset_folder = "C:/Users/alekhya/OneDrive/Desktop/Alekhya/VocalSense/dataset/age_gender/"

for category in os.listdir(dataset_folder):
    category_folder = os.path.join(dataset_folder, category)
    
    for file in os.listdir(category_folder):
        file_path = os.path.join(category_folder, file)

        if file.endswith(".mp3"):
            audio = AudioSegment.from_mp3(file_path)
            new_file_path = file_path.replace(".mp3", ".wav")
            audio.export(new_file_path, format="wav")
            os.remove(file_path)
        elif file.endswith(".flac"):
            audio = AudioSegment.from_file(file_path, format="flac")
            new_file_path = file_path.replace(".flac", ".wav")
            audio.export(new_file_path, format="wav")
            os.remove(file_path)

print("✅ All files converted to WAV format!")
