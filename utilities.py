from scipy.stats import entropy
from network import net
import numpy as np


def committee_KL(model, unlabled_x, C):
    committee_posterior =  model.committee_posterior(C.number_of_committee, unlabled_x) #pass the reference to the method
    rank_unlabeled = np.zeros((unlabled_x.shape[0], ))
    theta = 1/committee_posterior.shape[0]
    for x_index in range(committee_posterior.shape[1]):
        e = 0
        for cmtt in range(committee_posterior.shape[0]-1):
            e +=  entropy(committee_posterior[-1][x_index][:], committee_posterior[cmtt][x_index][:])
        e/=theta
        rank_unlabeled[x_index] = e

    return rank_unlabeled.argsort()[-C.active_batch:][::-1]

def committee_Jensen_divergence(model, unlabled_x, C):
    committee_posterior =  model.committee_posterior(C.number_of_committee, unlabled_x) #pass the reference to the method
    rank_unlabeled = np.zeros((unlabled_x.shape[0], ))
    theta = 1/committee_posterior.shape[0]
    for x_index in range(committee_posterior.shape[1]):
        e = 0
        for cmtt in range(committee_posterior.shape[0]-1):
            m = (committee_posterior[-1][x_index][:] + committee_posterior[cmtt][x_index][:])/2
            e +=  entropy(committee_posterior[-1][x_index][:], m)*.5 + entropy(committee_posterior[cmtt][x_index][:], m)*.5
        e/=theta
        rank_unlabeled[x_index] = e

    return rank_unlabeled.argsort()[-C.active_batch:][::-1]

def qbdc(model, unlabled_x, C, active_train_X, active_train_y):
    committee_posterior =  model.committee_posterior(C.number_of_committee, unlabled_x, X_train=active_train_X, y_train=active_train_y, type='qbdc') #pass the reference to the method
    rank_unlabeled = np.zeros((committee_posterior.shape[1], ))

    for x_index in range(rank_unlabeled.shape[0]):
        rank_unlabeled[x_index] = rg(committee_posterior, x_index)
    return rank_unlabeled.argsort()[-C.active_batch:][::-1]

def LABEL(committee_posterior, x_index):
    outputv=np.zeros((committee_posterior.shape[2],))
    for i in range(committee_posterior.shape[0]):
        outputv[committee_posterior[i][0][:].argmax()] += 1
    return outputv.argmax()

def rg(committee_posterior, x_index):
    sum = 0
    for cmmt in range(committee_posterior.shape[0]):
        max_prob = committee_posterior[cmmt][x_index][:].max()
        consensus_label= (LABEL(committee_posterior, x_index))
        prob_label = committee_posterior[cmmt][x_index][consensus_label]
        sum += (max_prob- prob_label)
    return sum

def bald(model, unlabled_x, C):
    score_All = np.zeros(shape=(unlabled_x.shape[0], C.output_shape[0]))
    All_Entropy_Dropout = np.zeros(shape=unlabled_x.shape[0])

    for d in range(C.dropout_iterations):
        dropout_score = model.stochastic_foward_pass(unlabled_x)

        score_All = score_All + dropout_score

        #computing F_X
        dropout_score_log = np.log2(dropout_score)
        Entropy_Compute = - np.multiply(dropout_score, dropout_score_log)
        Entropy_Per_Dropout = np.sum(Entropy_Compute, axis=1)
        All_Entropy_Dropout = All_Entropy_Dropout + Entropy_Per_Dropout

    # average out the probabilities .... ensembled based on dropout paper
    Avg_Pi = np.divide(score_All, C.dropout_iterations)
    # take log of the average
    Log_Avg_Pi = np.log2(Avg_Pi)
    #multply the average with their repective log probabilities -- to calculate entropy
    Entropy_Avg_Pi = -np.multiply(Avg_Pi, Log_Avg_Pi)
    Entropy_Average_Pi = np.sum(Entropy_Avg_Pi, axis=1)

    G_X = Entropy_Average_Pi

    Average_Entropy = np.divide(All_Entropy_Dropout, C.dropout_iterations)
    F_X = Average_Entropy

    U_X = G_X - F_X

    return U_X.argsort()[-C.active_batch:][::-1]


def model_instance(arg_param, C):
    if arg_param.dropout_type == 1:
        model = net(input_shape=C.input_shape, output_shape=C.output_shape, batch_size=C.epoch_batch_size, n_epochs=C.nb_epoch, dropout = C.dropout
                    , dropout_type='standard')
    elif arg_param.dropout_type == 2:
        model = net(input_shape=C.input_shape, output_shape=C.output_shape, batch_size=C.epoch_batch_size, n_epochs=C.nb_epoch, dropout = C.dropout
                    , dropout_type='batchwise')
    else:
        model = net(input_shape=C.input_shape, output_shape=C.output_shape, batch_size=C.epoch_batch_size, n_epochs=C.nb_epoch, dropout = C.dropout
                    , dropout_type=None)
    return model
