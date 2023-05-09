from cv_modules.camera_det_faces import det_faces
from cv_modules.emotion_recognition import find_emotion
from surveying_part import survey, generate_questions
from recommendation.emotopia import recommend_song
from cv_modules.face_reco import face_reco


def main():
    # Open Camera and Find Faces
    faces = det_faces()
    # Know the person name
    name, _ = face_reco(faces)
    # Detect The initial User Emotion from his face
    facial_emo = find_emotion(faces)
    print(f'Your Facial Emotion is: {facial_emo}')
    # Ask the user if the prediction is right
    user_quest = survey(facial_emo, name)
    # Ask the user some questions to get his final emotion
    user_final_emo = generate_questions(user_quest, number=3)
    recommend_song(user_final_emo)


if __name__ == '__main__':
    main()
