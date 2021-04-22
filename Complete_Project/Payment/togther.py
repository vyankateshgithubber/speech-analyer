
import pandas as pd
import numpy as np
import nltk 
nltk.download('stopwords')

train_df = pd.read_csv("/content/drive/MyDrive/PROJECT/Audio_text/Datasets/train.txt",sep=';')
train_df.columns = ["Sentance","Emotion"]

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps = PorterStemmer()
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences



def tokenize(tweets):
    stop_words = stopwords.words("english")
    tokenized_tweets = []
    for tweet in tweets:
        # split all words in the tweet
        words = tweet.split(" ")
        tokenized_string = ""
        for word in words:
            # remove @handles -> useless -> no information
            if word[0] != '@' and word not in stop_words:
                # if a hashtag, remove # -> adds no new information
                if word[0] == "#":
                    word = word[1:]
                #word = ps.stem(word) # stemming line
                tokenized_string += word + " "
            
        tokenized_tweets.append(tokenized_string)
    return tokenized_tweets


def encod_tweets(tweets):
    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', split=" ", lower=True)
    tokenizer.fit_on_texts(tweets)
    return tokenizer, tokenizer.texts_to_sequences(tweets)

tokenized_tweets = tokenize(train_df['Sentance'])
tokenizer, encoded_tweets = encod_tweets(tokenized_tweets)

import random
def tok(data,emotion):
  index = data.index
  index = random.choice(index[data['Emotion']==emotion])
  s = data['Sentance'][index]
  to = tokenizer.texts_to_sequences([s])
  to = pad_sequences(to, maxlen= 20, padding='post')
  to = np.array(to,dtype="float32")
  data.drop(index=index)
  return to,data

import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")#32
        sample_rate=sound_file.samplerate
        #print("MFCC")
      
        #if chroma:
         #   stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=16000, n_mfcc=40).T, axis=0)
        #    print("MFCC")
            result=np.hstack((result, mfccs))
        #if chroma:
        #    chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=16000).T,axis=0)
        #    result=np.hstack((result, chroma))
        #if mel:
         #   mel=np.mean(librosa.feature.melspectrogram(X, sr=16000).T,axis=0)
         #   result=np.hstack((result, mel))
    return result

emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}        
# 'disgust':'anger','calm':'love','angry':'anger','neutral':'love'
#DataFlair - Emotions to observe
observed_emotions=['happy','disgust','calm','angry','neutral','surprised','fearful','sad']

#DataFlair - Load the data and extract features for each sound file
def load_data(train_df,test_size):
    x,y=[],[]    
    x = np.empty((1,60))
    count_sad=0
    count_happy=0
    map = {'happy':'joy','surprised':'surprise','disgust':' ','calm':'surprise','angry':'anger','neutral':' ','fearful':'fear','sad':'sadness'}
    for file in glob.glob("/content/drive/MyDrive/PROJECT/Audio_text/Datasets/ravdess/RAVDESS/Actor_*/*.wav"):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feeling = map[emotion]
        feature=extract_feature(file, mfcc=True, chroma=False, mel=False)
        feature = feature.astype('float32')
        if(feeling!=' '):
          tokenized, train_df = tok(train_df,feeling)
        else:
          tokenized = np.zeros((1,20))
        tokenized = np.reshape(tokenized,(1,-1))
        feature = np.reshape(feature,(1,-1))
       # print(tokenized,feature)
        out = np.hstack((feature,tokenized))
        if emotion=='happy':
          count_happy+=1
        if emotion=='sad':
          count_sad+=1
        if count_sad<=count_happy and emotion=='sad':
          x = np.vstack([x, out])
          y.append(observed_emotions.index(emotion))
        elif emotion !='sad':
          x = np.vstack([x, out])
          y.append(observed_emotions.index(emotion))
    count_sad=0
    count_happy=0
    for file in glob.glob("/content/drive/MyDrive/PROJECT/Audio_text/Datasets/TESS/*.wav"):
        file_name=os.path.basename(file)
        emotion=file_name.split("_")[2][:-4]
        if emotion not in observed_emotions:
            continue
        feeling = map[emotion]
        feature=extract_feature(file, mfcc=True, chroma=False, mel=False)
        feature = feature.astype('float32')
        if(feeling!=' '):
          tokenized, train_df = tok(train_df,feeling)
        else:
          tokenized = np.zeros((1,20))
        tokenized = np.reshape(tokenized,(1,-1))
        feature = np.reshape(feature,(1,-1))
       # print(tokenized,feature)
        out = np.hstack((feature,tokenized))
        if emotion=='happy':
          count_happy+=1
        if emotion=='sad':
          count_sad+=1
        if count_sad<=count_happy and emotion=='sad':
          x = np.vstack([x, out])
          y.append(observed_emotions.index(emotion))
        elif emotion !='sad':
          x = np.vstack([x, out])
          y.append(observed_emotions.index(emotion))
    x = np.delete(x,0,0)
    return train_test_split(x, np.array(y), test_size=test_size, random_state=9)

#DataFlair - Split the dataset
x_train,x_test,y_train,y_test=load_data(train_df,test_size=0.20)

x_train.shape,y_train.shape

x_traincnn=np.expand_dims(x_train,axis=2)
x_testcnn=np.expand_dims(x_test,axis=2)

x_traincnn.shape, x_testcnn.shape

import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
import keras
from matplotlib.pyplot import specgram
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense , Embedding
from keras.layers import LSTM
from keras.utils import to_categorical
from keras.layers import Input,Flatten,Dropout,Activation
from keras.layers import Conv1D,MaxPooling1D,AveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix,classification_report


model=Sequential()
model.add(Conv1D(256,5,padding='same',input_shape=(60,1)))
model.add(Activation('relu'))
model.add(Dropout(0.1))#0.1/0.2
model.add(Conv1D(128,5,padding='same'))
model.add(Activation('relu'))

model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128,5,padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(128,5,padding='same'))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(10))
model.add(Activation('softmax'))
opt=keras.optimizers.RMSprop(lr=0.00001,decay=1e-6)

#model.summary()

model.compile(loss='sparse_categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

cnnhistory=model.fit(x_traincnn,y_train,batch_size=16,epochs=250,validation_data=(x_testcnn,y_test))

#model = keras.models.load_model('/content/drive/MyDrive/multimodel')

#https://heartbeat.fritz.ai/working-with-audio-signals-in-python-6c2bd63b2daf
#score = model.evaluate(x_testcnn, y_test, verbose=0)
#print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
  
#y_pred = model.predict(x_testcnn)

# summarize history for accuracy

plt.plot(cnnhistory.history['accuracy'])
plt.plot(cnnhistory.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(cnnhistory.history['loss'])
plt.plot(cnnhistory.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#model.save('/content/drive/MyDrive/my_model')

def next_block(filepath,text):
  feature=extract_feature(filepath, mfcc=True, chroma=False, mel=False)
  feature=np.reshape(feature,(1,-1))
  #text = tokenize([text])
  to = tokenizer.texts_to_sequences([text])
  to = pad_sequences(to, maxlen= 20 , padding='post')
  to = np.array(to)  
  to = np.reshape(to,(1,-1))
  feature1 = np.hstack((feature,to))
  feature1 = np.reshape(feature1,(1,60,1))
  a = model.predict(feature1)
  classes = np.argmax(a, axis = 1)
  print("Emotion : " ,classes," ",observed_emotions[classes[0]])
  return observed_emotions[classes[0]]


import time
import pyaudio
import speech_recognition as sr
"""def callback(recognizer , audio):
    try:
        input_string=recognizer.recognize_google(audio,language="en-SG")
        #first run the next_block cell and then run this one 
        print(input_string)
        print("Calling")
        next_block(filepath=filepath,text=input_string) 
    except:
        print("Opps didn't catch")
r=sr.Recognizer()
m=sr.AudioFile(filepath)
with m as source:
        r.dynamic_energy_threshold=True
        r.adjust_for_ambient_noise(source,duration=1)
        time.sleep(0.5)
stop_listening=r.listen_in_background(m,callback)
for _ in range(8):time.sleep(0.1) 
stop_listening()
for i in range(5):time.sleep(0.1)
"""
