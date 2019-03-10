from keras import backend as K


class Config:

    def __init__(self):
        self.initial_training_ratio = .01
        self.subsample_size = 100 #number of samples from which ranking will be done
        self.active_batch = 50 #bumber of samples to query from the oracle
        self.number_of_committee = 5
        self.nb_epoch = 100
        self.epoch_batch_size = 32
        self.T = 20
        self.dropout = 0.5
        self.dropout_iterations = 50
        self.num_experiments  = 1
        #both shapes set to initialy none until data is seen
        self.input_shape = None #
        self.output_shape =None #
        self.img_rows =None
        self.img_col =None
        self.channels = None
