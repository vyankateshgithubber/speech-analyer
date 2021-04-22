import numpy as np
import pandas as pd
train_df = pd.read_csv("/content/drive/MyDrive/Payment/train.txt",sep=';')
train_df.columns = ["Sentance","Emotion"]
test_df = pd.read_csv("/content/drive/MyDrive/Payment/test.txt",sep=';')
test_df.columns = ["Sentance","Emotion"]
result_emotion=""
train_length = train_df.shape[0]
test_length = test_df.shape[0]
#train_length, test_length

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords

stop_words = stopwords.words("english")
#stop_words[:5]

# tokenize the sentences
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
                tokenized_string += word + " "
        tokenized_tweets.append(tokenized_string)
    return tokenized_tweets

def encod_tweets(tweets):
    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', split=" ", lower=True)
    tokenizer.fit_on_texts(tweets)
    return tokenizer, tokenizer.texts_to_sequences(tweets)

# example_str = tokenize(['This is a good day. @css #mlhlocalhost'])
# encod_str = encod_tweets(example_str)
# print(example_str)
# print(encod_str)

# apply padding to dataset and convert labels to bitmaps
def format_data(encoded_tweets, max_length, labels):
    x = pad_sequences(encoded_tweets, maxlen= max_length, padding='post')
    y = []
    for emoji in labels:
        bit_vec = np.zeros(20)
        bit_vec[emoji] = 1
        y.append(bit_vec)
    y = np.asarray(y)
    return x, y

# create weight matrix from pre trained embeddings
def create_weight_matrix(vocab, raw_embeddings):
    vocab_size = len(vocab) + 1
    weight_matrix = np.zeros((vocab_size, 300))
    for word, idx in vocab.items():
        if word in raw_embeddings:
            weight_matrix[idx] = raw_embeddings[word]
    return weight_matrix

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.wrappers import Bidirectional
from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# final model
def final_model(vocab_size, max_length, x, y, epochs = 5):
    embedding_layer = Embedding(vocab_size, 300, input_length=max_length, trainable=True, mask_zero=True)
    model = Sequential()
    model.add(embedding_layer)
    model.add(Bidirectional(LSTM(128, dropout=0.2, return_sequences=True)))
    model.add(Bidirectional(LSTM(128, dropout=0.2)))
    model.add(Dense(20, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x, y, epochs = epochs, validation_split = 0.25)
    score, acc = model.evaluate(x_test, y_test)
    return model, score, acc

import math

tokenized_tweets = tokenize(train_df['Sentance'])
tokenized_tweets += tokenize(test_df['Sentance'])
max_length = math.ceil(sum([len(s.split(" ")) for s in tokenized_tweets])/len(tokenized_tweets))
tokenizer, encoded_tweets = encod_tweets(tokenized_tweets)
#max_length, len(tokenized_tweets)

tokenizer_l = Tokenizer()
tokenizer_l.fit_on_texts(train_df['Emotion'])
train_label = tokenizer_l.texts_to_sequences(train_df['Emotion'])
test_label = tokenizer_l.texts_to_sequences(test_df['Emotion'])
#tokenizer_l.word_index

map = tokenizer_l.word_index
map_emotion = {3:'anger', 4:'fear', 1:'joy', 5:'love', 2:'sadness', 6:'surprise'}

x, y = format_data(encoded_tweets[:train_length], max_length, train_label)
len(x), len(y)
x_test, y_test = format_data(encoded_tweets[train_length:], max_length, test_label)

voc = tokenizer.word_index
#len(voc)

model , score, acc = final_model(len(voc)+1,max_length,x,y,epochs=5)
#model , score, acc

#model.summary()

y_pred = model.predict(x_test)
#y_pred

#for pred in y_pred:
 #   print(np.argmax(pred))

import math
from sklearn.metrics import classification_report, confusion_matrix

y_pred = np.array([np.argmax(pred) for pred in y_pred])
y_true = np.array(test_label)
#print(classification_report(y_true, y_pred))

emoji_pred = [map_emotion[pred] for pred in y_pred]
#emoji_pred

#first run this block so that it is recognized prior to the function call
def next_block(a):
  #print(a)
  input = [a]
  input_token = tokenizer.texts_to_sequences(input) # tokenize the input string
  x = pad_sequences(input_token, maxlen= max_length, padding='post')
  #print(x)
  output = model.predict(x) # predict the emotion of output
  #print("output - ",output)
  Emotion = map_emotion[np.argmax(output)]
  global result_emotion
  result_emotion=Emotion
  print(result_emotion)
  return result_emotion
