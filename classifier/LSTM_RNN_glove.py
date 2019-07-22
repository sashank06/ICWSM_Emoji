#!/usr/bin/env python
# coding: utf-8

# In[1]:
#Some parts of the code adopted from Udacity's Tensorflow deep learning course
__author__ = ["Sashank Santhanam"]
__credits__ = ["Sashank Santhanam"]
__maintainer__ = "Sashank Santhanam"
__email__ = "ssantha1@uncc.edu"

import pandas as pd
import numpy as np
import tensorflow as tf
import nltk, re, time
from nltk.corpus import stopwords
from string import punctuation
from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from collections import namedtuple
import dask
from collections import Counter



tf.__version__


# In[3]:


df = pd.read_excel('data_english.xlsx')

text = df['tweets'].tolist()


def remove_http(tweets):
    removed = []
    for tweet in tweets:
        text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',"",str(tweet))
        #text = text.encode('ISO-8859-1').decode('utf-8','ignore')
        text = text.encode('utf-8').decode('utf-8','ignore')
        removed.append(text)
    return removed
tweets = remove_http(text)


from nltk.tokenize import TweetTokenizer

tknzr = TweetTokenizer()
token_tweet = []
for tweet in tweets:
    token_tweet.append(tknzr.tokenize(tweet))




words = []for tweet in token_tweet:
    for word in tweet:
        word = word.lower()
        word = word.replace("@", "")
        words.append(word.lower())


label = df['labels'].tolist()


labels_two = np.array([1 if each == 1 else 0 for each in label])


# # Create Vocab, word2int and int2word

# In[15]:


vocab = set(words)
vocab_to_int = dict(zip(vocab,range(1,len(vocab)+1)))
vocab_to_int['UNK'] = 0
print("Vocabulary Size including PAD and UNK: ",len(vocab_to_int))


# In[16]:


def loadWordVectors(filePath,vocab):
    txt = open(filePath)
    wordVecs = np.zeros((len(vocab),300),dtype=float)
    for line in txt:
        splitData = line.split(" ")
        word = splitData[0]
        #word = unicode(word,'utf8')
        if(word not in vocab):
            continue
        vector = splitData[1:len(splitData)]
        wordVecs[vocab[word]] = np.array(vector,dtype=float)
    return wordVecs
wordVecSize = 300
wordVecs = loadWordVectors('/Users/ssantha1/Desktop/real_fake/glove.6B/glove.6B.300d.txt',vocab_to_int)


np.save('word_Vecs.npy',wordVecs)


tweet_ints = []
for each in token_tweet:
    tweet_ints.append([vocab_to_int[word.replace("@","").lower()] if word in vocab_to_int else vocab_to_int['UNK'] for word in each])


# In[19]:

tweet_lens = Counter([len(x) for x in tweet_ints])
non_zero_idx = [ii for ii, tweet in enumerate(tweet_ints) if len(tweet) != 0]
tweet_ints = [tweet_ints[ii] for ii in non_zero_idx]
labels_two_way = np.array([labels_two[ii] for ii in non_zero_idx])

seq_len = 100
features = np.zeros((len(tweet_ints), seq_len), dtype=int)
for i, row in enumerate(tweet_ints):
    features[i, -len(row):] = np.array(row)[:seq_len]




#split_frac = 0.8
split_idx = int(len(features)*0.7)
train_x, val_x = features[:split_idx], features[split_idx:]
train_y, val_y = labels_two_way[:split_idx], labels_two_way[split_idx:]

test_idx = int(len(val_x)*0.5)
val_x, test_x = val_x[:test_idx], val_x[test_idx:]
val_y, test_y = val_y[:test_idx], val_y[test_idx:]

print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape), 
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))


# In[27]:


def get_batches(x, y, batch_size=100):
    '''Create batches for training data'''
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]
def get_test_batches(x, batch_size):
    '''Create the batches for the testing data'''
    n_batches = len(x)//batch_size
    x = x[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size]


# In[28]:


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


# In[51]:

#To Do: Updates these values
wordVecs = np.load('word_Vecs.npy')
embed_size = 300
batch_size = 25
lstm_size = 512
num_layers = 2
dropout = 0.5
learning_rate = 0.001
epochs = 20
multiple_fc = True
fc_units = 64


# In[52]:


def build_rnn(n_words, embed_size, batch_size, lstm_size, num_layers, dropout, learning_rate, multiple_fc, fc_units):
    '''Build the Recurrent Neural Network'''

    tf.reset_default_graph()

    # Declare placeholders we'll feed into the graph
    with tf.name_scope('inputs'):
        inputs = tf.placeholder(tf.int32, [None, None], name='inputs')

    with tf.name_scope('labels'):
        labels = tf.placeholder(tf.int32, [None, None], name='labels')

    keep_prob = tf.placeholder(tf.float64, name='keep_prob')

    # Create the embeddings
    with tf.name_scope("embeddings"):
        W = tf.Variable(wordVecs,name="W")
        embed = tf.nn.embedding_lookup(W, inputs)
        #embed = tf.reduce_sum(embed, 1)

    # Build the RNN layers
    with tf.name_scope("RNN_layers"):
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, 
                                         output_keep_prob=keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)
    
    # Set the initial state
    with tf.name_scope("RNN_init_state"):
        initial_state = cell.zero_state(batch_size, tf.float64)

    # Run the data through the RNN layers
    with tf.name_scope("RNN_forward"):
        outputs,final_state = tf.nn.dynamic_rnn(cell,embed,dtype=tf.float64,swap_memory=True)  
    
    # Create the fully connected layers
    with tf.name_scope("fully_connected"):
        
        # Initialize the weights and biases
        weights = tf.truncated_normal_initializer(stddev=0.1)
        biases = tf.zeros_initializer()
        
        dense = tf.contrib.layers.fully_connected(outputs[:, -1],
                    num_outputs = fc_units,
                    activation_fn = tf.sigmoid,
                    weights_initializer = weights,
                    biases_initializer = biases)
        
        dense = tf.contrib.layers.dropout(dense, keep_prob)
        
        # Depending on the iteration, use a second fully connected  layer
        if multiple_fc == True:
            dense = tf.contrib.layers.fully_connected(dense,
                        num_outputs = fc_units,
                        activation_fn = tf.sigmoid,
                        weights_initializer = weights,
                        biases_initializer = biases)
            
            dense = tf.contrib.layers.dropout(dense, keep_prob)
    
    # Make the predictions
    with tf.name_scope('predictions'):
        predictions = tf.contrib.layers.fully_connected(dense, 
                          num_outputs = 1, 
                          activation_fn=tf.sigmoid,
                          weights_initializer = weights,
                          biases_initializer = biases)
        
        tf.summary.histogram('predictions', predictions)
    
    # Calculate the cost
    with tf.name_scope('cost'):
        cost = tf.losses.mean_squared_error(labels, predictions)
        tf.summary.scalar('cost', cost)
    
    # Train the model
    with tf.name_scope('train'):    
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Determine the accuracy
    with tf.name_scope("accuracy"):
        correct_pred = tf.equal(tf.cast(tf.round(predictions), 
                                        tf.int32), 
                                        labels)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
    
    # Merge all of the summaries
    merged = tf.summary.merge_all()    

    # Export the nodes 
    export_nodes = ['inputs', 'labels', 'keep_prob','initial_state',        
                    'final_state','accuracy', 'predictions', 'cost', 
                    'optimizer', 'merged']
    Graph = namedtuple('Graph', export_nodes)
    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])
    
    return graph


# In[53]:


def train(model, epochs, log_string):
    '''Train the RNN'''

    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Used to determine when to stop the training early
        valid_loss_summary = []
        train_loss_summary = []
        
        # Keep track of which batch iteration is being trained
        iteration = 0

        print()
        print("Training Model: {}".format(log_string))

        train_writer = tf.summary.FileWriter('./logs/glove/train/{}'.format(log_string), sess.graph)
        valid_writer = tf.summary.FileWriter('./logs/glove/valid/{}'.format(log_string))

        for e in range(epochs):
            state = sess.run(model.initial_state)
            
            # Record progress with each epoch
            train_loss = []
            train_acc = []
            val_acc = []
            val_loss = []

            with tqdm(total=len(train_x)) as pbar:
                for _, (x, y) in enumerate(get_batches(train_x,       
                                               train_y, 
                                               batch_size), 1):
                    feed = {model.inputs: x,
                            model.labels: y[:, None],
                            model.keep_prob: dropout,
                            model.initial_state: state}
                    summary, loss, acc, state, _ = sess.run([model.merged, 
                                                  model.cost, 
                                                  model.accuracy, 
                                                  model.final_state, 
                                                  model.optimizer], 
                                                  feed_dict=feed)       
              
                    
                    # Record the loss and accuracy of each training  batch
                    
                    train_loss.append(loss)
                    train_acc.append(acc)
                    
                    # Record the progress of training
                    train_writer.add_summary(summary, iteration)
                    
                    iteration += 1
                    pbar.update(batch_size)
            
            # Average the training loss and accuracy of each epoch
            avg_train_loss = np.mean(train_loss)
            avg_train_acc = np.mean(train_acc) 
            train_loss_summary.append(avg_train_loss)

            val_state = sess.run(model.initial_state)
            with tqdm(total=len(val_x)) as pbar:
                for x, y in get_batches(val_x,val_y,batch_size):
                    feed = {model.inputs: x,
                            model.labels: y[:, None],
                            model.keep_prob: 1,
                            model.initial_state: val_state}
                    summary, batch_loss, batch_acc, val_state = sess.run([model.merged, model.cost,model.accuracy,model.final_state],feed_dict = feed)

                    
                    # Record the validation loss and accuracy of each epoch
                                          
                    val_loss.append(batch_loss)
                    val_acc.append(batch_acc)
                    pbar.update(batch_size)
            
            # Average the validation loss and accuracy of each epoch
            avg_valid_loss = np.mean(val_loss)    
            avg_valid_acc = np.mean(val_acc)
            valid_loss_summary.append(avg_valid_loss)
            
            # Record the validation data's progress
            valid_writer.add_summary(summary, iteration)

            # Print the progress of each epoch
            print("Epoch: {}/{}".format(e, epochs),
                  "Train Loss: {:.3f}".format(avg_train_loss),
                  "Train Acc: {:.3f}".format(avg_train_acc),
                  "Valid Loss: {:.3f}".format(avg_valid_loss),
                  "Valid Acc: {:.3f}".format(avg_valid_acc))

            # Stop training if the validation loss does not decrease after 3 epochs
            
            '''if avg_train_loss > min(train_loss_summary):
                print("No Improvement.")
                stop_early += 1
                if stop_early == 3:
                    break   
            
            # Reset stop_early if the validation loss finds a new low
            # Save a checkpoint of the model
            else:'''
            print("New Record!")
            stop_early = 0
            checkpoint ="./Glove/tweets_{}.ckpt".format(log_string)
            saver.save(sess, checkpoint)


# In[54]:


# start from 128, false , 128
#add more values to the list to run multiple experiments. Not the ideal way but with a smaller dataset, you run multiple experiment quickly

for l in [lstm_size]:
    for multiple_fc in [True,False]:
        for f in [fc_units]:
            log_string = 'ru={},fcl={},fcu={}_latest_new'.format(l,
                                                      multiple_fc,
                                                      f)
            model = build_rnn(n_words =wordVecs, 
                              embed_size = embed_size,
                              batch_size = batch_size,
                              lstm_size = l,
                              num_layers = num_layers,
                              dropout = dropout,
                              learning_rate = learning_rate,
                              multiple_fc = multiple_fc,
                              fc_units = f)            
            train(model, epochs, log_string)


# In[55]:


checkpoint_new = "./tweets_ru=32,fcl=False,fcu=64_latest.ckpt"
# checkpoint_new = "./Glove/tweets_ru=32,fcl=False,fcu=64_latest_new.ckpt"

def make_predictions(lstm_size, multiple_fc, fc_units, checkpoint):
    '''Predict the sentiment of the testing data'''
    
    # Record all of the predictions
    all_preds = []

    model = build_rnn(n_words = wordVecs, 
                      embed_size = embed_size,
                      batch_size = batch_size,
                      lstm_size = lstm_size,
                      num_layers = num_layers,
                      dropout = dropout,
                      learning_rate = learning_rate,
                      multiple_fc = multiple_fc,
                      fc_units = fc_units) 
    
    with tf.Session() as sess:
        saver = tf.train.Saver()
        # Load the model
        saver.restore(sess, checkpoint)
        test_state = sess.run(model.initial_state)
        for _, x in enumerate(get_test_batches(test_x, 
                                               batch_size), 1):
            feed = {model.inputs: x,
                    model.keep_prob: 1,
                    model.initial_state: test_state}
            predictions = sess.run(model.predictions,feed_dict=feed)
            for pred in predictions:
                all_preds.append(float(pred))
                
    return all_preds

predictions_try = make_predictions(32, False, 64, checkpoint_new)

print("Test accuracy with dropout 0.5: {:.3f}".format(np.mean(predictions_try)))


