
import numpy as np
from keras.layers import *
from keras.models import *
from matplotlib import pyplot as plt
from itertools import combinations, product
from tensorflow.examples.tutorials.mnist import input_data
from keras.datasets import cifar10
import keras
from PIL import Image 
from scipy.misc import imresize

#dataset = 'CIFAR'
dataset = 'MNIST'


if dataset == 'MNIST':
    input_model = load_model('./mnist_classification_model')
    loaded_Q_model = load_model(dataset + '_Q_adversarial_model') 
    save_dir = './MNIST_adversarial_examples/'
    
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
    loaded_Q_model = load_model(dataset + '_Q_adversarial_model')  
    save_dir = './CIFAR_adversarial_examples/'
    
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
max_blocks_attack = 15
block_size = 2
x_span = list(range(0,X_train.shape[1],block_size))
blocks = list(product(x_span,x_span)) 


sucess = 0
for games in range(5000,len(X_test)):
   
    sample_img = X_test[games]
    sample_img = np.expand_dims(sample_img,axis = 0)
    
    orig_label = np.argmax(input_model.predict(sample_img),axis = 1)[0]
    orig_img = np.array(sample_img)
                                                                                                                                                                         
    
    for ite in range(max_blocks_attack):
              
        sample_img_prob = input_model.predict(sample_img)[0]
        sample_img_prob = np.expand_dims(sample_img_prob,axis = 0)
        
        action = np.argmax(loaded_Q_model.predict([sample_img,sample_img_prob]))
                              
        attack_region = np.zeros((sample_img.shape))       
        attack_cord = blocks[action]
               
        attack_region[0,attack_cord[0]:attack_cord[0]+block_size, attack_cord[1]:attack_cord[1]+block_size,:] = 1
        
             
        sample_img_noise = sample_img + lamda * attack_region
                 
        mod_label = np.argmax(input_model.predict(sample_img_noise),axis = 1)[0]
        
        
        if mod_label != orig_label:
            sucess += 1
            print(sucess)
            
            sample_img_noise = np.squeeze(sample_img_noise * 255.)
            sample_img_noise = Image.fromarray(sample_img_noise.astype('uint8'))
            sample_img_noise = sample_img_noise.resize((128, 128)) 
            
            if dataset == 'CIFAR':
                sample_img_noise.save(save_dir + str(games) + '_actual_' + cifar_class_names[orig_label] + '_mod_' + cifar_class_names[mod_label] + '.png')
            
            if dataset == 'MNIST':
                sample_img_noise.save(save_dir + str(games) + '_actual_' + str(orig_label) + '_mod_' + str(mod_label) + '.png')
                               
                    
            break
        
        sample_img = np.array(sample_img_noise)
   
     