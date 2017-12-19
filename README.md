# text-sentiment-analysis

This analyzer is based on SVM. Implementation of SVM is taken from [Tensorflow: CookBook](https://github.com/nfmcclure/tensorflow_cookbook)

### Prerequisites
1. download [dataset](http://ai.stanford.edu/~amaas/data/sentiment/)
2. copy unpacked dataset to root folder of this repo
3. use preparator.py script to transform dataset to more usable view

### Train
For training use script main.py, parameters which you can try to variate:
* number_epoch
* learning_rate
* alpha_val
* batch_size

### Test
For check accuracy of training use test.py script. 
It will print accuracies for test part of the dataset.

### Portability
use freeze.py script to freeze your model to use it in your purpose in another projects.
This script will produce file model.pb in trained_model folder.
* Input tensor: "x:0"
* Output tensor: "y_pred:0"
<br/>You can check how to import this frozen model in java by example in [this repository](https://github.com/PROteinBY/text-polarity)

### Try your example
If you want to check model by your example use scripts:
* predict_from_frozen_model.py
* predict_from_std_model.py
<br/>text which you want recognise put in text variable for both of this scripts

