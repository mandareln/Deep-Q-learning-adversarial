
import numpy as np
from keras.layers import *
from keras.models import *
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from itertools import combinations, product
from tensorflow.examples.tutorials.mnist import input_data
from keras.datasets import cifar10
import keras


def get_cnn_model(input_shape = None):
    cnn_model = Sequential()
    cnn_model.add(Conv2D(64,5, padding = 'same', activation = 'relu', input_shape = input_shape))
    cnn_model.add(Dropout(0.5))
    
    cnn_model.add(Conv2D(64,5, padding = 'same', activation = 'relu',))
    cnn_model.add(Dropout(0.5))
    
    cnn_model.add(Flatten())
    cnn_model.add(Dense(128, activation = 'relu'))
    #cnn_model.add(Dropout(0.5))
    
    return cnn_model


def get_dnn_model(input_shape = 10):
    dnn_model = Sequential()
    dnn_model.add(Dense(128, activation = 'relu', input_shape = [input_shape]))
    #dnn_model.add(Dropout(0.5))
    return dnn_model


def update_q_model(exp_replay = None, q_model = None, batch_size = None, dis_factor = None):
   
   train_input_1 = []
   train_input_2 = []
   
   train_label = []
   for ind in range(len(exp_replay)):
    
        init_img_state = exp_replay[ind][0]
        init_prob_state = exp_replay[ind][1]
        
        next_img_state = exp_replay[ind][4]
        next_prob_state = exp_replay[ind][5]
        
        action_taken = exp_replay[ind][2]
        reward_obtained = exp_replay[ind][3]
               
        target = q_model.predict([init_img_state,init_prob_state])[0]
                        
        Q_sa = np.max(q_model.predict([next_img_state,next_prob_state])[0])
#                
        if reward_obtained == 10 or reward_obtained == -1:
            target[action_taken] = reward_obtained
        else:
            target[action_taken] = reward_obtained + dis_factor * Q_sa
        
        
        train_input_1.append(init_img_state)
        train_input_2.append(init_prob_state)
        
        train_label.append(target)
       
   train_input_1 = np.squeeze(np.array(train_input_1))
   train_input_2 = np.squeeze(np.array(train_input_2))
   
   train_label = np.array(train_label)  
     
   if len(train_input_1.shape) == 3:
       train_input_1 = np.expand_dims(train_input_1, axis = -1)
     
   train_input_1,train_input_2,train_label = shuffle(train_input_1,train_input_2,train_label, random_state = 0)
     
   #print('model update')
   q_model.train_on_batch([train_input_1[0:batch_size],train_input_2[0:batch_size]], train_label[0:batch_size])
   
   return q_model
