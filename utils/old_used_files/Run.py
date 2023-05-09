import pandas as pd
import csv
import numpy as np
from google.colab import files
import mic
from mic import get_audio
from transcribe import transcribe
from text_classify import classify
from transformers import pipeline
import pandas as pd


def run_project():
  audio = get_audio()
  audio = transcribe(audio)
  Mode = classify(audio)
  print(Mode , " The most repeated emotion is" ,Mode.label.mode()[0] ) 
  # return Mode.label.mode()[0]

