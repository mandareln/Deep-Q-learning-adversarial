# Deep Q learning for adversarial attacks

To train the DQN model for MNIST/CIFAR-10 dataset, run the train_DQN_for_adversarial_attack.py.

After the end of the training, it saves the train Q model in the directory.

Run test_DQN_adversarial_attack.py to generate and save adversarial images. 

The keras classification model files for MNIST and CIFAR are generated with keras version 2.1.4

Run 'pip install keras==2.1.4' if the loading of the models fails.
