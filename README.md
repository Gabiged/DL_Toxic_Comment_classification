# DL_Toxic_Comment_classification
Natural Language Processing
Toxic comment challenge
data source: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
The aim of this work is to create a tool to help to sort the forum comments made, with particular attention to the negative and the toxic ones.
The aim of this work is to build a multi-label classifier, to assign forum posts to one or more of the 6 classes:
toxic, severe_toxic, obscene, threat, insult, identity_hate.

For this assignment I will use the recommended Distilbert model and DistilBertTokenizer as tokenizer.
The work plan is:
* preprocess given dataset;
* tokenize text using DistilBertTokenize;
* the network will have the DistilBERT model. Follwed by a Droput and Linear Layer. They are for Regulariaztion and Classification.
* The number of targets for Linear Layer is 6 because that is the total number of our target labels.
* Final model layer outputs is what will be used to calcuate the loss and to get the accuracy of prediction.
* The trained model will be saved at its best validation dataset loss value.
* The loss function used will be a combination of Binary Cross Entropy which is implemented as BCELogits Loss in PyTorch;
* Optimizer AdamW is used to update the weights of the neural network to improve its performance.
* Schedule with a constant learning rate preceded by a warmup period during which the learning rate increases linearly between 0 and the initial lr set in the optimizer.
* Since it is multilabel classifikation I will use a Sigmoid activation to the output layer for each output to have its own and independent probability.
