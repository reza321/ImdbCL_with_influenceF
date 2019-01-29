import tensorflow as tf
from tensorflow import keras
import numpy as np
# print(tf.__version__)
imdb=keras.datasets.imdb
(train_data, train_labels),(test_data,test_labels)=imdb.load_data(num_words=10000)

print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
# print(train_data[0])
# print(np.shape(train_data[0]))
# print(train_labels)
# print(test_data.shape)
# print(test_labels.shape)

len(train_data[0]), len(train_data[1])

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
                value=word_index["<PAD>"],padding='post',maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                value=word_index["<PAD>"],padding='post',maxlen=256)

vocab_size = 10000

# print(train_data.shape)
# print(train_labels.shape)
# print(test_data.shape)
# print(test_labels.shape)

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

tf.set_random_seed(0)

print(partial_y_train.shape)
# c = np.random.random([10,1])
# b = tf.nn.embedding_lookup(c, [1, 3])
# print(c)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print (sess.run(b))


def model(vocab_size,x):
    # with tf.name_scope("embedding"):
    #     # W = tf.Variable(tf.constant(0.0, shape=[doc_vocab_size, embedding_dim]), trainable=True, name="W")
    #     W = tf.get_variable("embedding", [vocab_size, 16]) 
    #     embedded = tf.nn.embedding_lookup(W,x)
    # with tf.variable_scope("embedding"):
    with tf.variable_scope("embedding"):
        w2=tf.get_variable(name="what",shape=[256,16],initializer=tf.truncated_normal_initializer(
            stddev=0.2, dtype=tf.float32),dtype=tf.float32)       

    # inputs=tf.nn.embedding_lookup(w2,x)
    gh=tf.matmul(x,w2)
    # flat=tf.reduce_mean(inputs, axis=[1,2])
    # dense_w=tf.get_variable("dense1",shape=(16,16))
    # dense1=tf.nn.relu(tf.matmul(flat,dense_w))

    # dense_w2=tf.get_variable("dense2",shape=(16,1))
    # dense2=tf.nn.sigmoid(tf.matmul(dense1,dense_w2))
    return gh
   
with tf.variable_scope("embedding"):
    w2=tf.get_variable(name="what",shape=[10000,16],initializer=tf.truncated_normal_initializer(
        stddev=0.2, dtype=tf.float32),dtype=tf.float32)    
x=tf.placeholder(tf.int32,[15000,256])  

# gh=tf.matmul(x,w2)
gh=tf.nn.embedding_lookup(w2,x)

init = tf.global_variables_initializer()        

# x=tf.placeholder(dtype=tf.float32,shape=[15000])

with tf.Session() as sess:
    sess.run(init)    
    # b=model(10000,x)


    # print(partial_x_train.shape) #(15000,256)        
    feed_dict={x:partial_x_train}
    bb=sess.run(gh,feed_dict=feed_dict)
    print(bb)
    print(bb.shape)    

