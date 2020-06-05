"""
CSCI 630 : Foundations of Artificial Intelligence
Lab 2
Author: Sarthak Thakkar (st4070)

main_class.py

This Program takes input from user to either train or predict
from a trainned model. the trainning is done with two approaches
decision tree and ada Boosting for prediction of sentence and
the trainned model is then stored in serialized object to be
used for prediction.
"""

import sys
import pickle
from decision_tree import DecisionTree
from ada_boost import AdaBoost

def train(example_file,out_file,learning_type):
    """
    Trains a model basesd on example_data and learning
    approach and writes model to serialized object.

    :param example_file:    training file
    :param out_file:        hypothesis file
    :param learning_type:   learning approach

    :return:
    """
    if learning_type =='dt':
        # print('Trainning Model with decision trees')
        decisionModel = DecisionTree(example_file,out_file)
        decisionModel.train()

    if learning_type == 'ada':
        # print('Trainning model with adaboost')
        adaModel = AdaBoost(example_file,out_file)
        adaModel.train()

def predict(hypothesis,test_file):
    """
    It predicts the language for statements in given file
    using the model specified in hypothesis file.

    :param hypothesis:  model to be used
    :param test_file:   file containing untagged sentences

    :return:
    """
    hypothesis_file = open(hypothesis,'rb')
    model = pickle.load(hypothesis_file)
    hypothesis_file.close()

    model.predict(test_file)

if __name__ == "__main__":
    """
    It takes input from user and starts program execution.
    """
    if sys.argv[1] == 'train':
        example_file = sys.argv[2]
        out_file = sys.argv[3]
        learning_type = sys.argv[4]
        train(example_file,out_file,learning_type)

    if sys.argv[1] == 'predict':
        out_file = sys.argv[2]
        learning_type = sys.argv[3]
        predict(out_file, learning_type)
