import speech_recognition as sr
from recorder import Recorder
r = Recorder()
r.record(15, output='output.wav')
r.play('output.wav')


AUDIO_FILE = ("output.wav")
# obtain audio from the microphone
r = sr.Recognizer()
with sr.AudioFile(AUDIO_FILE) as source:
    audio = r.record(source)

# recognize speech using Sphinx
try:
    print(r.recognize_google(audio))
except sr.UnknownValueError:
    print("Google could not understand audio")
except sr.RequestError as e:
    print("Google error; {0}".format(e))
