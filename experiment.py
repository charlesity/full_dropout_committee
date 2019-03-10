#Code written by Charles I. Saidu https://github.com/charlesity
#All rights reserved
import math
import numpy as np
import argparse
import sys

from network import net
from config import Config
from keras.datasets import mnist
from keras import backend as K
K.clear_session()
from sklearn.model_selection import StratifiedShuffleSplit
from keras.utils import to_categorical
from matplotlib import pyplot as plt
import time

from utilities import *

import argparse

start_time = time.time()
parser = argparse.ArgumentParser()

parser.add_argument("q_type", type=int,
                    help="1 =fqbdc KL acquition, 2=fqbdc jensen acquition, 3 = qbdc acquition, 4 = bald acquition, else = random acquition")

parser.add_argument("dropout_type", type=int,
                    help="1 =standard, 2 = batchwise, query type")

parser.add_argument("re_initialize_weights", type=int,
                    help="determine whether or not weights should be re-initialized")

arg = parser.parse_args()


(X_train, y_train), (X_test, y_test) = mnist.load_data()

C = Config()
#set the shape configuration
C.img_rows, C.img_col, C.channels = X_train[0].shape[0], X_train[0].shape[0], 1

#use StratifiedShuffleSplit to partition the training set into labeled and unlabeled
sss = StratifiedShuffleSplit(n_splits = C.num_experiments, train_size = C.initial_training_ratio)

all_training_loss = np.zeros((C.T + 1, ))
all_val_loss = np.zeros((C.T + 1, ))
all_training_acc = np.zeros((C.T + 1, ))
all_val_acc = np.zeros((C.T + 1, ))


y_test = to_categorical(y_test)
for train_index, un_labeled_index in sss.split(X_train,y_train):
    active_train_X = X_train[train_index]
    active_train_y = y_train[train_index]

    unlabeled_X = X_train[un_labeled_index]
    unlabeled_y = y_train[un_labeled_index]

    if K.image_data_format() == 'channels_first':
        active_train_X = active_train_X.reshape(active_train_X.shape[0], C.channels, C.img_rows, C.img_col)
        X_test = X_test.reshape(X_test.shape[0], C.channels, C.img_rows, C.img_col)
        unlabeled_X = unlabeled_X.reshape(unlabeled_X.shape[0], C.channels, C.img_rows, C.img_col)
        C.input_shape = (C.channels, C.img_rows, C.img_col)
    else:
        active_train_X = active_train_X.reshape(active_train_X.shape[0], C.img_rows, C.img_col, C.channels)
        X_test = X_test.reshape(X_test.shape[0], C.img_rows, C.img_col, C.channels)
        unlabeled_X = unlabeled_X.reshape(unlabeled_X.shape[0], C.img_rows, C.img_col, C.channels)
        C.input_shape = (C.img_rows, C.img_col, C.channels)

    unlabeled_y = to_categorical(unlabeled_y)
    active_train_y = to_categorical(active_train_y)

    C.output_shape = (active_train_y.shape[1],)

    model = model_instance(arg, C)


    history = model.getModel().fit(active_train_X, active_train_y, epochs=C.nb_epoch, batch_size=C.epoch_batch_size)
    current_weights = model.getModel().get_weights()  #save just in case weights are initialized at every active learning query

    scores = model.getModel().evaluate(X_test, y_test, batch_size=C.epoch_batch_size)
    training_loss = [np.mean(history.history['loss'])]

    training_acc = [np.mean(history.history['acc'])]
    val_loss = [scores[0]]
    val_acc = [scores[1]]

    for query_count in range(C.T):
        print (query_count, " active acquition ")
        #randomly select from the unlabeled set
        subsampled_indices = np.random.choice(unlabeled_X.shape[0], C.subsample_size)
        if arg.q_type == 1:  # fully partial posterior query by dropout committee with KL
            print ('fp_qbdc_KL')
            index_maximum = committee_KL(model, unlabeled_X[subsampled_indices], C)

        elif arg.q_type == 2:  # fully partial posterior query by dropout committee with jensen
            print ('fp_qbdc_jensen')
            index_maximum = committee_Jensen_divergence(model, unlabeled_X[subsampled_indices], C)

        elif arg.q_type == 3:  # query by dropout committee
            print ('qbdc')
            index_maximum = qbdc(model, unlabeled_X[subsampled_indices], C, active_train_X, active_train_y)
        elif arg.q_type == 4: #bald
            print('bald acquition')
            index_maximum=bald(model, unlabeled_X[subsampled_indices], C)

        else: #random
            print ('Random acquisition')
            index_maximum = np.random.uniform(0, subsampled_indices.shape[0], C.active_batch).astype(np.int)


        addition_samples_indices = subsampled_indices[index_maximum] # subset of the indices with the maximum information

        # print ("active x and y ", active_train_X.shape, active_train_y.shape)
        additional_samples_X = unlabeled_X[addition_samples_indices]
        additional_samples_y = unlabeled_y[addition_samples_indices]
        # print (additional_samples_X.shape, additional_samples_y.shape)

        # print(unlabeled_X.shape, unlabeled_y.shape)
        unlabeled_X = np.delete(unlabeled_X, addition_samples_indices, axis=0)
        unlabeled_y = np.delete(unlabeled_y, addition_samples_indices, axis=0)
        # print(unlabeled_X.shape, unlabeled_y.shape)


        # print (active_train_X.shape, active_train_y.shape)
        active_train_X = np.append(active_train_X, additional_samples_X, axis=0)

        active_train_y = np.append(active_train_y, additional_samples_y, axis=0)

        # run experiments for both re-initialized weights and non-initialized weights
        if arg.re_initialize_weights == 0:
            model.getModel().set_weights(current_weights)
        else:
            model = model_instance(arg, C)

        # model = net(input_shape=input_shape, output_shape=output_shape)

        history = model.getModel().fit(active_train_X, active_train_y, epochs=C.nb_epoch, batch_size=C.epoch_batch_size)
        scores = model.getModel().evaluate(X_test, y_test, batch_size=C.epoch_batch_size)

        val_loss.append(scores[0])
        val_acc.append(scores[1])
        training_loss.append(np.mean(history.history['loss']))
        training_acc.append(np.mean(history.history['acc']))

    all_training_loss += np.array(training_loss)
    all_training_acc += np.array(training_acc)
    all_val_loss += np.array(val_loss)
    all_val_acc += np.array(val_acc)



all_training_loss /=C.num_experiments
all_training_acc /=C.num_experiments
all_val_loss /=C.num_experiments
all_val_acc /=C.num_experiments


duration = time.time() - start_time
print (duration)
plt.plot(all_val_loss, label ='val_loss')
plt.plot(all_training_loss, label ='training_loss')
plt.legend()
plt.figure()
plt.plot(all_val_acc, label ='val_acc')
plt.plot(all_training_acc, label='training_accuracy')
plt.legend()
plt.show()
