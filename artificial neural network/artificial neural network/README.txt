To run this file one need run this following command from command line

python ann_classify.py optdigits-3.tra optdigits-3.tes epochs hidden_units learning_rate

example

python ann_classify.py optdigits-3.tra optdigits-3.tes 1000 5 0.1
Here
optdigits-3.tra is the training dataset
optdigits-3.tes is the test dataset
epochs  is the number of iterations to train the data
hidden_units is the number of nodes in the hidden layer. we used 5 and 50.
learning_rate is the number how our neural network will learn.

the training and test dataset has to be in the same folder of the python file.