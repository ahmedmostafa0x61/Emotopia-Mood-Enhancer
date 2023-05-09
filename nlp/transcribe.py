import whisper

# load from previously saved models
model = whisper.load_model("/content/medium.en.pt")


def transcribe(audio):
    result = model.transcribe(audio)

    print(result["text"])

    x = []
    for count, i in enumerate(result["segments"]):
        z = result["segments"][count]['text'].split('.')
        for orz in z:
            x.append(orz)

    return x
