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
# print(train_labels)
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


def model(vocab_size,x,labels):
    with tf.variable_scope("embedding"):
        w2=tf.get_variable(name="what",shape=[vocab_size,16],initializer=tf.truncated_normal_initializer(
            stddev=0.2, dtype=tf.float32),dtype=tf.float32)    
    gh=tf.nn.embedding_lookup(w2,x)
    flat=tf.reduce_mean(gh, axis=[1])
    dense_w=tf.get_variable("dense1",shape=(16,16))
    dense1=tf.nn.relu(tf.matmul(flat,dense_w))

    dense_w2=tf.get_variable("dense2",shape=(16,2))
    logits=tf.matmul(dense1,dense_w2)
    
    labels = tf.one_hot(labels, depth=2) 
    cross_entropy = - tf.reduce_sum(tf.multiply(labels, tf.nn.log_softmax(logits)), reduction_indices=1)        
    indiv_loss_no_reg = cross_entropy

    loss_no_reg = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    tf.add_to_collection('losses', loss_no_reg)
    total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    optimizer = tf.train.AdamOptimizer(0.01)
    train_op = optimizer.minimize(total_loss) 
    
    return loss_no_reg,train_op
   
# with tf.variable_scope("embedding"):
#     w2=tf.get_variable(name="what",shape=[10000,16],initializer=tf.truncated_normal_initializer(
#         stddev=0.2, dtype=tf.float32),dtype=tf.float32)    
# x=tf.placeholder(tf.int32,[15000,256])	

# # gh=tf.matmul(x,w2)
# gh=tf.nn.embedding_lookup(w2,x)
# x=tf.placeholder(dtype=tf.float32,shape=[15000])


x=tf.placeholder(tf.int32,shape=(None,256))	
labels=tf.placeholder(tf.int32,shape=(None))	
b=model(10000,x,labels)
init = tf.global_variables_initializer()        

epochs=100
batch_size=500
with tf.Session() as sess:
    sess.run(init)    
    # print(partial_x_train.shape) #(15000,256)        
    for epoch in range(epochs):    	
        for i in range(int(partial_x_train.shape[0]/batch_size)):
            input_x=partial_x_train[i*batch_size:(i+1)*batch_size]
            input_y=partial_y_train[i*batch_size:(i+1)*batch_size]

            feed_dict={x:input_x,labels:input_y}
            loss_val,_=sess.run(b,feed_dict=feed_dict)            
            print(loss_val)










