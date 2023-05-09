import glob
import os

# Download ffmpeg (required for audio manipulations)
os.system("sudo apt update && sudo apt install ffmpeg")

# # Mount google drive and then unzip the model weights
# from google.colab import drive
# drive.mount('/content/drive')
try:
    os.system("unzip '/content/drive/MyDrive/NTI Project/output_file_name.zip'")
except:
    print('labeling model weights not found!')

# Download Whisper model weights
if not glob.glob('medium.en.pt'):
    os.system(
        "wget https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt -P /content/ ")

os.system('git clone https://github.com/ageitgey/face_recognition')
os.system('pip install face_recognition')

os.system("pip install -r /content/utils/requirements.txt")
