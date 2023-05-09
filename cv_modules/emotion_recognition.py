from deepface import DeepFace


def find_emotion(frame):
    dominant_emo = 'neutral'
    try:
        result = DeepFace.analyze(frame, actions=['emotion'])
        dominant_emo = result[0]['dominant_emotion']
        # print('Test1:', result)
    except:
        dominant_emo = 'neutral'

    # print('Test2:',result)

    if (dominant_emo == 'angry') or (dominant_emo == 'disgust') or (dominant_emo == 'fear') or (dominant_emo == 'sad'):
        dominant_emo = 'sad'
    elif (dominant_emo == 'surprise') or (dominant_emo == 'neutral'):
        dominant_emo = 'neutral'
    else:
        dominant_emo = 'happy'
    return dominant_emo
