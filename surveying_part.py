from utils.io_devices import get_audio, run_audio
from nlp.transcribe import transcribe
from nlp.text_classify import classify, agree_or_no
import random
from collections import Counter


def survey(text, name):
    
    if name != 'Unknown':
        prob = f'Welcome home {name}, you look {text} today. Is that right?'
    else:
        prob = f'Welcome home, you look {text} today. Is that right?'

    # Run audio to Welcome user, then ask if the CV prediction is right
    run_audio(prob)
    # listen for the user and get his audio
    user_input = get_audio()
    # Convert Audio to text
    trans_text = transcribe(user_input)
    # Find if there is any sign of agreement
    agree = agree_or_no(trans_text)
    if agree.lower() == 'positive':
        pass
    else:
        text = 'neutral'
    return text


def generate_questions(text, number=5):
    # text = survey(text)
    questions = open(f'/content/questions/{text}.txt').read().splitlines()

    all_user_emotions = []
    for i in range(1, number + 1):
        rand_question = random.choice(questions)
        run_audio(rand_question)
        if i < number:
            user_input = get_audio()
            trans_text = transcribe(user_input)
            text_emotion = classify(trans_text)
            all_user_emotions.append(text_emotion)
    occurrence_count = Counter(all_user_emotions)
    return occurrence_count.most_common(1)[0][0]
