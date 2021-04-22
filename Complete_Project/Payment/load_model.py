import pandas as pd
import numpy as np
import nltk 
import keras
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
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=16000, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
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
observed_emotions=['happy','disgust','calm','angry','neutral','surprised','fearful','sad']

model = keras.models.load_model('/content/drive/MyDrive/PROJECT/Audio_text/Datasets/multimodel')

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