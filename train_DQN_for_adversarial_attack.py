
import numpy as np
from keras.layers import *
from keras.models import *
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from itertools import combinations, product
from tensorflow.examples.tutorials.mnist import input_data
from keras.datasets import cifar10
import keras
from helper_functions import *

dataset = 'MNIST'
#dataset = 'CIFAR'

if dataset == 'MNIST':
    input_model = load_model('./mnist_classification_model')
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    mnist_images = mnist.train.images
    mnist_train_images = np.reshape(mnist_images,[mnist_images.shape[0],28,28,1])
    X_train = mnist_train_images
    y_train = mnist.train.labels
    mnist_images = mnist.test.images
    X_test = np.reshape(mnist_images,[mnist_images.shape[0],28,28,1])
    y_test = mnist.test.labels
    input_shape = [28,28,1]
    
    num_classes = 10
    lamda = .4


if dataset == 'CIFAR':
    input_model = load_model('./CIFAR_classification_model')
    cifar_class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    num_classes = 10
    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    input_shape = [32,32,3]
    lamda = .1


### defining the action space of agent
block_size = 2
x_span = list(range(0,X_train.shape[1],block_size))
blocks = list(product(x_span,x_span)) 


### defining DQN model
cnn_model = get_cnn_model(input_shape = input_shape)
dnn_model = get_dnn_model(input_shape = num_classes)

prob_input = Input(shape = [num_classes])
img_input = Input(shape = input_shape)

prob_rep = dnn_model(prob_input)
img_rep = cnn_model(img_input)

x = Concatenate(axis = -1)([prob_rep,img_rep])
x = Dense(len(blocks), activation = 'linear')(x)

q_model = Model(inputs = [img_input,prob_input], outputs = [x])

q_model.summary()
q_model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])



#### Q learning hyperparameters
eps = .9
dis_factor = 0.9
max_memory = 1000
max_blocks_attack = 15


sucess = []
sucess_rate = []
exp_replay = []

### generating espisodes from 5k test images
for games in range(5000):
    
    sample_img = X_test[games]
    sample_img = np.expand_dims(sample_img,axis = 0)
    
    if games > 0 and games % 300 == 0:
        eps = eps - 0.1
    
    if eps <= 0.1:
        eps = 0.1
    
    orig_label = np.argmax(input_model.predict(sample_img),axis = 1)
    orig_img = np.array(sample_img)
    
   
    for ite in range(0,max_blocks_attack):
        #print(ite)       
        sample_img_prob = input_model.predict(sample_img)[0]
        sample_img_prob = np.expand_dims(sample_img_prob,axis = 0)
        
        if np.random.rand() < eps:
            action = np.random.randint(0,len(blocks))
        else:
            action = np.argmax(q_model.predict([sample_img,sample_img_prob]))
               
        attack_region = np.zeros((sample_img.shape))       
        attack_cord = blocks[action]
        attack_region[0,attack_cord[0]:attack_cord[0]+block_size, attack_cord[1]:attack_cord[1]+block_size,:] = 1
               
        sample_img_noise = sample_img + lamda * attack_region
        sample_img_noise_prob = input_model.predict(sample_img_noise)
        
        mod_label = np.argmax(input_model.predict(sample_img_noise),axis = 1)
        
        if mod_label != orig_label:
            
            reward = 10.
            sucess.append(1)
            exp_replay.append([sample_img,sample_img_prob,action,reward,sample_img_noise,sample_img_noise_prob])
            break
        
        else:
            reward = -.1
            exp_replay.append([sample_img,sample_img_prob,action,reward,sample_img_noise,sample_img_noise_prob])
            
        sample_img = np.array(sample_img_noise)
               
    
    if ite == (max_blocks_attack-1):
        reward = -1.
        exp_replay.append([sample_img,sample_img_prob,action,reward,sample_img_noise,sample_img_noise_prob])
        sucess.append(0)
                                           
        
    if len(exp_replay) >= max_memory:
       
       q_model = update_q_model(exp_replay = exp_replay, q_model = q_model, batch_size = 32, dis_factor = dis_factor)
       exp_replay = []
              
       print('Q model updated,', 'sucess rate', np.mean(np.array(sucess)))
       sucess_rate.append(np.mean(np.array(sucess)))
       sucess = []
             


### saving the q model and sucess rate
q_model.save(dataset + '_Q_adversarial_model')


       

