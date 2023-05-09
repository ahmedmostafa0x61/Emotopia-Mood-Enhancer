from transformers import pipeline
import pandas as pd

# load from previously saved models
classifier = pipeline("text-classification", model="/content/content/distilbert-base-uncased-finetuned-emotion")
sentiment_pipeline = pipeline("sentiment-analysis")

# ##### Labels should be related to System 2##############
labels = {'LABEL_0': 'sad', 'LABEL_1': 'joy', 'LABEL_2': 'love', 'LABEL_3': 'angry', 'LABEL_4': 'fear',
          'LABEL_5': 'surprise', 'LABEL_6': 'neutral','LABEL_7': 'worry'}


# def classify(text):
#   for orz in text:
#     pred = classifier(orz, top_k=1)
#     print(orz, lablels[pred[0]['label']])

def classify(text):
    ls = []
    for orz in text:
        if len(orz) > 2:
            pred = classifier(orz, top_k=1)
            ls.append(labels[pred[0]['label']])
    all_emotions = pd.DataFrame(ls, columns=['label'])
    final_emotion = all_emotions['label'].mode()[0]
    return final_emotion


def agree_or_no(text):
    result = sentiment_pipeline(text)
    return result[0]['label']
