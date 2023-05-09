from IPython.display import Audio, display


def recommend_song(label):
    display(Audio('/content/music/' + label + '.mp3', autoplay=True))
