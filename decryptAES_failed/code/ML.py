from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import os


def toArray(s):
    ret = ''.join(format(ord(x), 'b').zfill(8) for x in s)
    # print (ret)
    return ret


def reformat():
    with open('output') as f:
        save = pickle.load(f)
        count = 0
        trainnum = len(save)
        train_data = np.zeros(trainnum * 128, dtype=np.bool).reshape((-1, 128))
        train_label = np.zeros(trainnum * 128, dtype=np.bool).reshape((-1, 128))
        test_data = np.zeros((len(save) - trainnum) * 128, dtype=np.bool).reshape((-1, 128))
        test_label = np.zeros((len(save) - trainnum) * 128, dtype=np.bool).reshape((-1, 128))
        print(train_data.shape)
        for inst in save:
            if count < trainnum:
                train_label[count] = np.array([False if i == '0' else True for i in toArray(inst)])
                train_data[count] = np.array([False if i == '0' else True for i in toArray(save[inst])])
            else:
                test_data[count - trainnum] = np.array([True if i == '1' else False for i in toArray(save[inst])])
                test_label[count - trainnum] = np.array([True if i == '1' else False for i in toArray(inst)])
            count += 1
        pickle.dump(train_data, open('all_data_large', 'w'))
        pickle.dump(train_label, open('all_label_large', 'w'))
        del save


def loaddatasets(num_train):
    all_data = pickle.load(open('all_data'))
    all_label = pickle.load(open('all_label'))
    train_data = all_data[:num_train].astype(np.float32)
    train_label =np.array( [[0, 1] if x else [1, 0] for x in all_label[:num_train, 5]]).astype(np.float32)
    test_data = all_data[num_train:].astype(np.float32)
    test_label =np.array( [[0, 1] if x else [1, 0] for x in all_label[num_train:, 5]]).astype(np.float32)
    return train_data, train_label, test_data, test_label


def acc(predictions,label):
    return  (100.0 * np.sum(np.argmax(predictions,1)==np.argmax(label,1)) / predictions.shape[0])


if __name__ == '__main__':
    reformat()
    raise
    train_data, train_label, test_data, test_label = loaddatasets(70000)
    #train_data, train_label, test_data, test_label = pickle.load(open('smalldata'))
    print (train_data[0].shape)

    #print(train_data)
    #print(test_label)
    print ('DATA LOADED')
    batch_size = 128
    beta = 0.001

    h1_nodes = 128
    h2_nodes = 512

    graph = tf.Graph()

    with graph.as_default():
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, 128))
        tf_train_label = tf.placeholder(tf.float32, shape=(batch_size, 2))
        tf_test_dataset = tf.constant(test_data)

        # H1
        h1_w = tf.Variable(tf.truncated_normal([128, h1_nodes]))
        h1_b = tf.Variable(tf.zeros([h1_nodes]))
        h1 = tf.nn.relu(tf.matmul(tf_train_dataset, h1_w) + h1_b)

        # H2
        h2_w = tf.Variable(tf.truncated_normal([h1_nodes, h2_nodes]))
        h2_b = tf.Variable(tf.zeros([h2_nodes]))
        h2 = tf.nn.relu(tf.matmul(h1, h2_w) + h2_b)


        # last

        w = tf.Variable(tf.truncated_normal([h2_nodes, 2]))
        b = tf.Variable(tf.zeros([2]))

        logits = tf.matmul(h2, w)+b

        loss= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,tf_train_label) + beta * tf.nn.l2_loss(w))

        optimizer= tf.train.GradientDescentOptimizer(0.1).minimize(loss)

        prediction = tf.nn.softmax(logits)
        test_relu1= tf.nn.relu(tf.matmul(tf_test_dataset,h1_w)+h1_b)
        test_relu2= tf.nn.relu(tf.matmul(test_relu1,h2_w)+h2_b)
        test_prediction= tf.nn.softmax(tf.matmul(test_relu2,w)+b)

    numstep = 30000
    with tf.Session(graph=graph) as ses:
        tf.initialize_all_variables().run()
        print ('TF INIT')
        for step in range(numstep):
            offset = (step*batch_size) % (train_label.shape[0]-batch_size)
            batch_data=train_data[offset:offset+batch_size,:]
            batch_label=train_label[offset:offset+batch_size,:]

            feed_dict= {tf_train_dataset:batch_data, tf_train_label:batch_label,}
            _,l,pp = ses.run([optimizer,loss,prediction],feed_dict=feed_dict)
            if step % 500==0:
                print ("batch ACC %d :%f"%(step,acc(pp,batch_label)))
        print ("Test ACC %.1f"%acc(test_prediction.eval(),test_label))
