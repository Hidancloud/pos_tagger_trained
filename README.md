**pos tagger by lstm implemented on pytorch**


pretrained model weights are at file 'model-file'   
to train the model execute 'python tagger_train.py'   
to test it with corpus.test use 'python tagger_predict.py corpus.test model-file corpus.out'   
to check accuracy of the predicted pos use 'python tagger_eval.py corpus.out corpus.answer'   
   
*Pretrained weights* are trained on 3 epochs and gets 91% accuracy on the test set (1 hour of training on the colab gpu)
model uses only char embedding, their convolutions, lstm and dense linear layer to get probabilities with log_softmax
