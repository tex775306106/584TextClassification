import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_20newsgroups
import time
import datetime
from f_model import forward_model
from tensorflow.contrib import learn
import math


#function of generating batches
def batch_iter(data, batch_size, epochs):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield data[start_index:end_index]


# Choose which categories of news you want to import
#set cates to None to load all categories
cates=['comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space',
 'soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc',
 'talk.religion.misc']

#Load datasets for both training and testing
print("Loading dataset")
trainingset=fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'),categories=cates, shuffle=True, random_state=42)
testingset=fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'),categories=cates, shuffle=True, random_state=42)

#Load labels
x_text = trainingset['data']
x_testtext=testingset['data']
labels = []
test_labels=[]
for i in range(len(x_text)):
    label = [0 for j in trainingset['target_names']]
    label[trainingset['target'][i]] = 1
    labels.append(label)
for i in range(len(x_testtext)):
    label2 = [0 for j in testingset['target_names']]
    label2[testingset['target'][i]] = 1
    test_labels.append(label2)
y = np.array(labels)

y_test=np.array(test_labels)
y_n=np.argmax(y, axis=1)
y_test2 = np.argmax(y_test, axis=1)

# Build vocabulary for the input data and set the maximum length to be 1000
max_document_length = 1000
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))
x_test = np.array(list(vocab_processor.fit_transform(x_testtext)))


with tf.Session() as sess:
    #load the model and set 2 filters with shapes [5,dimension of word embedding] and [7,dimension of word embedding]
    #There are 128 filters for each shape
    cnn = forward_model(
        sequence_length=x.shape[1],
        num_classes=y.shape[1],
        vocab_size=len(vocab_processor.vocabulary_),
        filter_sizes=list(map(int, "5,7".split(","))),
        num_filters=128)

    # Define Training step
    #AdamOptimizer can control the learning rate decay
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(cnn.learning_rate).minimize(cnn.loss, global_step=global_step)

    #Initalize all variables
    sess.run(tf.global_variables_initializer())
    vocabulary = vocab_processor.vocabulary_

    #Apply pretrained word embedding
    print("Loading pretrained GloVe word embeddings\n")
    embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocab_processor.vocabulary_), 50))
    f = open("WordVectors/glove.6B.50d.txt")
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        idx = vocabulary.get(word)
        if idx != 0:
            embedding_vectors[idx] = vector
    f.close()
    sess.run(cnn.W.assign(embedding_vectors))

    def train_step(x_batch, y_batch, learning_rate):
        feed_dict = {
          cnn.input_x: x_batch,
          cnn.input_y: y_batch,
          cnn.learning_rate: learning_rate,
          cnn.dropout_keep_prob: 0.85
        }
        _, step,  loss, accuracy= sess.run([optimizer, global_step, cnn.loss, cnn.accuracy], feed_dict)
        print("step {}, loss {:g}, accuracy {:g}, learning rate {:g}".format(step, loss, accuracy, learning_rate))
        return accuracy

    def eval(x_batch, y_batch):
        feed_dict = {
            cnn.input_x: x_batch,
            cnn.input_y: y_batch,
            cnn.dropout_keep_prob: 1.0
        }
        loss,acc = sess.run([cnn.loss,cnn.accuracy],feed_dict)
        return acc




    # Generate batches
    batches = batch_iter(
        list(zip(x, y)), 64, 200)
    # It uses dynamic learning rate with a high value at the beginning to speed up the training
    max_learning_rate = 0.01
    min_learning_rate = 0.0005
    decay_rate = len(y)/32

    counter = 0
    acc_train=0.0
    beginTime = time.time()
    #Training
    for batch in batches:
        learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-counter/decay_rate)
        counter += 1
        x_batch, y_batch = zip(*batch)
        acc_train+=train_step(x_batch, y_batch, learning_rate)
        current_step = tf.train.global_step(sess, global_step)
    acc_train=acc_train/counter
    endTime=time.time()
    print("Total number of traing examples: {}".format(len(y_n)))
    print("Total training time: {}".format(endTime-beginTime))
    print("Average accuracy on training set: {:g}".format(acc_train))


    print("Total number of testing examples: {}".format(len(y_test2)))
    batches2 = batch_iter(list(zip(x_test, y_test)), 64, 1)
    acc_eval=0.0
    counter2=0.0
    #Testing
    for batch_ in batches2:
        counter2+=1.0
        x_batch2, y_batch2 = zip(*batch_)
        acc_eval+=eval(x_batch2,y_batch2)
    acc_eval=acc_eval/counter2
    print("Average accuracy on testing set: {:g}".format(acc_eval))
