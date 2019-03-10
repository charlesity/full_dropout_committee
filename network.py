
import warnings
warnings.filterwarnings("ignore")

import math
from scipy.misc import logsumexp
import numpy as np

from keras.regularizers import l2
from keras import Input, Model
from keras.layers import Dropout, Conv2D, Dense, MaxPooling2D, Flatten
from keras import backend as K

import time


class net:

    def __init__(self,input_shape, output_shape, batch_size, hidden_sizes =[200, 200, 50, 10], filters=[20,20]
                 , filter_sizes=[3, 3], max_pooling_sizes =[2,2], n_epochs = 40, dropout = 0.5,
                 dropout_type ='standard'):
        # We construct the network
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.filters= filters
        self.filter_sizes = filter_sizes
        self.max_pooling_sizes = max_pooling_sizes
        self.hidden_sizes = hidden_sizes
        self.n_epochs = n_epochs
        self.dropout= dropout
        self.batch_size = batch_size


        #input part of the network
        inputs = Input(shape=input_shape)
        first_conv = Conv2D(filters[0], kernel_size=filter_sizes[0], activation='relu')(inputs)
        first_max_pool = MaxPooling2D(pool_size=(max_pooling_sizes[0], max_pooling_sizes[0]))(first_conv)
        #convolutional part of the network
        for i in  np.arange(1, len(filters)):
            inter = Conv2D(filters[i], kernel_size=filter_sizes[i], activation='relu')(first_max_pool)
            # inter = Dropout(.2)(inter)
            inter = MaxPooling2D(pool_size=(max_pooling_sizes[i], max_pooling_sizes[i]))(inter)

        # fully connected part of the network
        inter = Flatten()(inter)
        for h_size in hidden_sizes:
            inter = Dense(h_size, activation='relu')(inter)
            if dropout_type == 'standard':
                inter = Dropout(dropout)(inter) #standard dropout
            elif dropout_type == 'batchwise':
                inter = Dropout(dropout, noise_shape=(1, h_size) )(inter) #batchwise dropout
        outputs = Dense(output_shape[0], activation='softmax')(inter)
        model = Model(inputs, outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model
        # model.summary()
        # exit()

        #function for sampling committee member via forward passes
        self.f = K.function([model.layers[0].input, K.learning_phase()],[model.layers[-1].output])

    def committee_posterior(self, n_committee, x_unlabled,  type='ours'
                            , X_train=None, y_train=None):
        sampled_committee_posterior = np.zeros((n_committee, x_unlabled.shape[0], self.output_shape[0]))
        if type=='ours':
            # sampled_committee_posterior = np.zeros((n_committee, x_unlabled.shape[0], self.output_shape[0]))
            for i in range(n_committee-1):
                sampled_committee_posterior[i] = self.f((x_unlabled, 1))[0]

            #consensus committee member, standard dropout average
            sampled_committee_posterior[n_committee -1] = self.model.predict(x_unlabled)
            return sampled_committee_posterior
        else:

            #get the last weight matrix
            L = self.model.layers[-1].get_weights()
            # sample each committee and recompute loss
            for i in range(n_committee):
                B = np.random.binomial(1, 1-self.dropout, size=L[0].shape)
                L[0] = np.multiply(L[0], B) #sample bernoulli random matrix and perform element wise multiplication to dropout
                self.model.layers[-1].set_weights(L) #set dropout matrix
                self.model.fit(X_train, y_train, epochs=self.n_epochs//2)
                sampled_committee_posterior[i] = self.model.predict(x_unlabled)
            return sampled_committee_posterior

    def stochastic_foward_pass(self, x_unlabled):
        return self.f((x_unlabled, 1))[0]


    def getModel(self):
        return self.model;
