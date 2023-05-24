#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import pandas as pd
import re
import string
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
stopwords = nltk.corpus.stopwords.words('english')
from keras.initializers import Constant
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import LSTM, Embedding, Dense, SpatialDropout1D, Bidirectional
from sklearn.model_selection import train_test_split
import random
#%%
class disaster_tweet_detector():
    def __init__(self, df_train):
        self.df_train = df_train
        self.maxlen = 50
        
    def remove_url(self,text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'', text)
    
    def remove_html(self, text):
        html = re.compile(r'<.*?>')
        return html.sub(r'', text)
    
    def remove_emoji(self, text):
        emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)
    
    def remove_punct(self, text):
        table = str.maketrans('', '', string.punctuation)
        return text.translate(table)
    
    def preprocess_text(self, text, test = False):
        text = self.remove_url(text)
        text = self.remove_html(text)
        text = self.remove_emoji(text)
        text = self.remove_punct(text)
        return text
    
    def create_word_index(self):
        stop = set(stopwords)
        corpus = []
        for tweet in self.df_train['text']:
            words = [word.lower() for word in word_tokenize(tweet) if ((word.isalpha() ==1) & (word not in stop))]
            corpus.append(words)
        
        tokenizer_obj = Tokenizer()
        tokenizer_obj.fit_on_texts(corpus)
        sequences = tokenizer_obj.texts_to_sequences(corpus)
        
        # create word index for all training data, and pad the training data
        self.word_index = tokenizer_obj.word_index
        self.tokenizer_obj = tokenizer_obj
        self.tweet_pad = pad_sequences(sequences, maxlen = self.maxlen, truncating = 'post', padding = 'post')
    
    def create_embedding(self, embedding_dimension):
        # use this function to create embedding layer with given string: embedding dimension and tokenizers: word_index 
        # output is the embedding layer for the given embedding dimension
        embedding_dict = {}
        glove_filename = 'Data/glove-global-vectors-for-word-representation/glove.6B.'+ embedding_dimension +'d.txt'
        with open(glove_filename, 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vectors = np.asarray(values[1:], 'float32')
                embedding_dict[word] = vectors
        f.close()

        # create embedding matrix with pre trained glove file
        num_words = len(self.word_index)+1
        embedding_matrix = np.zeros((num_words, int(embedding_dimension)))
        for word, i in tqdm(self.word_index.items()):
            if i > num_words:
                continue
            emb_vec = embedding_dict.get(word)
            if emb_vec is not None:
                embedding_matrix[i] = emb_vec
            embedding_layer = Embedding(num_words, int(embedding_dimension), embeddings_initializer = Constant(embedding_matrix), input_length = self.maxlen, trainable = False)
        print(f'Embedding layer of {embedding_dimension} dimension created!')
        return embedding_layer
    
    def create_lstm(self, num_neurons, bidir):
        if bidir == True:
            return Bidirectional(LSTM(num_neurons, dropout = 0.2, recurrent_dropout = 0.2))
        else:
            return LSTM(num_neurons, dropout = 0.2, recurrent_dropout = 0.2)
        
    def build_classfier(self):
        # clean training data
        self.df_train['text'] = self.df_train['text'].apply(lambda x: self.preprocess_text(x))
        self.create_word_index()
        
        ## try embedding dimension with 200D, and bidirectional lstm layer
        model = tf.keras.Sequential()
        model.add(self.create_embedding('200'))
        model.add(SpatialDropout1D(0.2))
        model.add(self.create_lstm(64, True))
        model.add(Dense(1, activation = 'sigmoid'))
        model.compile(loss = 'binary_crossentropy', optimizer = RMSprop(learning_rate = 1e-5), metrics = ['accuracy'])
        self.model = model
        
        print('Created a deep learning classification model with a embedding layer and LSTM layer')
        print(model.summary())
        
    def train_model(self, num_epochs):
        # split 10% of training data and use as validation data
        train_X, val_X, train_y, val_y = train_test_split(self.tweet_pad, self.df_train['target'].values, test_size = 0.1)
        self.history = self.model.fit(train_X, train_y, epochs = num_epochs, validation_data = (val_X, val_y))
        self.model.save('users_model')
        
    def predict(self, test_text, model):
        # perform the same preprocess on the given test text
        test_str = self.preprocess_text(test_text, True)
        stop = set(stopwords)
        corpus = []
        # remove any stop words
        for word in word_tokenize(test_str):
            if (word.isalpha() ==1) & (word not in stop):
                corpus.append(word)    
        test = ' '.join(corpus)
        # tokenize the given text and pad the array 
        sequences = self.tokenizer_obj.texts_to_sequences([test])
        test_arr = pad_sequences(sequences, maxlen = self.maxlen, truncating = 'post', padding = 'post')

        confidence = model.predict(test_arr)[0][0]
        return confidence


#%%
if __name__ == '__main__':
    # let user decide if they want to use pretrained model or retrain the model
    start = input('Use pretrained model (note: retraining the model will take a while)? [Y/N]: ').upper()
    while (start != 'Y') & (start != 'N'):
        start = input('Use pretrained model? [Y/N]: ').upper()
    
    # read training data
    train_df = pd.read_csv("Data/nlp-getting-started/train.csv")
    # create disaster detector object
    disaster_detector = disaster_tweet_detector(train_df)
    # build the deep learning model with LSTM layer
    disaster_detector.build_classfier()
    
    if start == 'N':
        # train the model with preprocessed training data and set the epochs to 40
        disaster_detector.train_model(num_epochs = 40)
        trained_model = disaster_detector.model
    else:
        trained_model = tf.keras.models.load_model('my_model')
    
    # now, we can test how the model is doing
    test_df = pd.read_csv("Data/nlp-getting-started/test.csv")
    
    start = input('try it out [Y/N]: ').upper()
    while (start != 'Y') & (start != 'N'):
        start = input('try it out [Y/N]: ').upper()
    while start == 'Y':
        user_input = input('Try it out [1: use random example; 2: user give example]: ')
        while (user_input != '1') & (user_input != '2'):
            user_input = input('Please give a choice 1 or 2: ')
        if user_input == '1':
            test_text = random.sample(test_df['text'].tolist(), 1)[0]
        else:
            test_text = input('Please provide an tweet you want to test: ')

        confidence = disaster_detector.predict(test_text, trained_model)
        if confidence >= 0.5:
            print(f'given tweet example: {test_text}')
            print(f'This tweet is a disaster tweet with confidence {round(confidence*100, 2)}%')
        else:
            print(f'given tweet example: {test_text}')
            print(f'This tweet is not a disaster tweet with confidence {round(100 - confidence*100, 2)}%')
            
        start = input('try it out [Y/N]: ').upper()
        while (start != 'Y') & (start != 'N'):
            start = input('try it out [Y/N]: ').upper()
        print('Thank you for using!')
