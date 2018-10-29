# Deep Q learning for adversarial attacks

To train the DQN model for MNIST/CIFAR-10 dataset, run the train_DQN_for_adversarial_attack.py

The dataset can be selected on line 13/14 from the train_DQN_for_adversarial_attack.py.

After the end of the training, it saves the trained Q model in the same directory.

Run test_DQN_adversarial_attack.py to generate and save the adversarial images. 

The keras classification model files for MNIST and CIFAR are generated with keras version 2.1.4
Run 'pip install keras==2.1.4' if the loading of the models fails.
