import tensorflow as tf
import numpy as np
import os
import math
from tensorflow import keras
import sys
import pickle
from six.moves import cPickle
import gzip
import urllib.request
from tensorflow.contrib.learn.python.learn.datasets import base
from dataset import DataSet
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from genericNeuralNet2 import GenericNeuralNet, variable, variable_with_weight_decay

def load_imdb(maxlen=256,num_words=10000):
    # print(tf.__version__)
    imdb=keras.datasets.imdb
    (train_data, train_labels),(test_data,test_labels)=imdb.load_data(num_words=num_words)  
    
    print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
    # print(len(train_data[0]),len(train_data[1]))
    # A dictionary mapping words to an integer index
    word_index = imdb.get_word_index()

    # The first indices are reserved
    word_index = {k:(v+3) for k,v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    def decode_review(text):
        return ' '.join([reverse_word_index.get(i, '?') for i in text])
    decode_review(train_data[0])


    train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                    value=word_index["<PAD>"],padding='post',maxlen=maxlen)

    test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                    value=word_index["<PAD>"],padding='post',maxlen=maxlen)

    validation_size = 5000

    validation_data = train_data[:validation_size]
    train_data = train_data[validation_size:]
    validation_labels = train_labels[:validation_size]
    train_labels = train_labels[validation_size:]
    print(validation_data.shape)
    print(train_data.shape)
    print(validation_labels.shape)
    print(train_labels.shape)

    train = DataSet(train_data, train_labels)
    validation = DataSet(validation_data, validation_labels)
    test = DataSet(test_data, test_labels)

    return base.Datasets(train=train, validation=validation, test=test)


class textCL_model(GenericNeuralNet):
    def __init__(self,weight_decay,vocab_size,maxlen,dense1_unit,**kwargs):
        self.vocab_size=vocab_size
        self.weight_decay=weight_decay
        self.dense1_unit=dense1_unit
        self.input_dim=maxlen
        self.num_classes = 2        
        super(textCL_model, self).__init__(**kwargs)

    def inference(self,input_x):
        input_x_reshaped=tf.reshape(input_x, [-1, self.input_dim])
        
        with tf.variable_scope('embedding'):
            weights = variable_with_weight_decay('weights', [self.vocab_size* self.dense1_unit],
                                                stddev=1.0 / math.sqrt(float(self.dense1_unit)),wd=self.weight_decay)                    
            embedding_layer=tf.nn.embedding_lookup(tf.reshape(weights,(self.vocab_size,self.dense1_unit)),input_x_reshaped)    

        flat=tf.reduce_mean(embedding_layer, axis=[1])

        with tf.variable_scope('dense1'):
            weights = variable_with_weight_decay('weights', (self.dense1_unit* self.dense1_unit),
                                                stddev=1.0 / math.sqrt(float(self.dense1_unit)),wd=self.weight_decay)            

            dense1 = tf.nn.relu(tf.matmul(flat, tf.reshape(weights, (self.dense1_unit, self.dense1_unit))))

        with tf.variable_scope('dense2'):
            weights = variable_with_weight_decay('weights', (self.dense1_unit* self.num_classes),
                                                stddev=1.0 / math.sqrt(float(self.dense1_unit)),wd=self.weight_decay)            
            logits=tf.matmul(dense1, tf.reshape(weights, (self.dense1_unit, self.num_classes)))
        return logits
        

    def load_trained_model(self, input_x):
            model = Sequential()                       
            model=self.inference(input_x)
            model.load_weights( self.file_name)
            return model            

    def predictions(self, logits):
        preds = tf.nn.softmax(logits, name='preds')
        return preds 


    def retrain(self, num_steps, feed_dict):        

        retrain_dataset = DataSet(feed_dict[self.input_placeholder], feed_dict[self.labels_placeholder])
        
        for step in range(num_steps):   
            iter_feed_dict = self.fill_feed_dict_with_batch(retrain_dataset)
            self.sess.run(self.train_op, feed_dict=iter_feed_dict)


    def get_all_params(self):
        # names=[n.name for n in tf.get_default_graph().as_graph_def().node]
        all_params = []
        for layer in ['embedding','dense1', 'dense2']:        
            for var_name in ['weights']:
                temp_tensor = tf.get_default_graph().get_tensor_by_name("%s/%s:0" % (layer, var_name))            
                all_params.append(temp_tensor)
        return all_params 

    def placeholder_inputs(self):
        input_placeholder = tf.placeholder(
            tf.int32, 
            shape=(None, self.input_dim),
            name='input_placeholder')
        labels_placeholder = tf.placeholder(
            tf.int32,             
            shape=(None),
            name='labels_placeholder')
        return input_placeholder, labels_placeholder






