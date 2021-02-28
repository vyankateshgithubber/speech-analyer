import time
import pyaudio
import speech_recognition as sr
def callback(recognizer , audio):
    try:
        #this part of the code is actully where the real speech to text conversion occurs in accordance with the language specified in parameters
        print("you said : " + recognizer.recognize_google(audio,language="en-US"))
    except:
        #if the system was unable to record any kind of speech  , this wil be diplayed on screen 
        print("Opps didn't catch")
r=sr.Recognizer()#this is the recognizing person , wo will help us convert to text with the attributes it has
m=sr.Microphone(sample_rate=33000,chunk_size=4096)# the rate of breaking the CT signal into a DT , to store in memory in a buffer of size 4096 units
with m as source:
        r.energy_threshold=600#this is the threshold energy being considered as the noise and can also be set dynamically while calibrating
        r.adjust_for_ambient_noise(source,duration=5)# this is where your microphone is being callibrated for 5 sec , getting familiar wiht surrounding
        time.sleep(0.5)# the purpose of waiting here is to avoid the error while initial setup of input like glitches
        print("Speak now !!!! I am listening")
stop_listening=r.listen_in_background(m,callback)#listen in the background , like telling the microphone that you are in a noisy environment
for _ in range(80):time.sleep(0.1)# accept the input for 8 sec 
stop_listening()
for i in range(5):time.sleep(0.1)# terminate the code 
def main():
    return 
