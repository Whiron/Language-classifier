"""
CSCI 630 : Foundations of Artificial Intelligence
Lab 2
Author: Sarthak Thakkar (st4070)

weighted_data.py

This Program contains WeightedData class. It has all the operational
functions required for ada boosting such as updating weights,
normalizing weights and maintaining weights of every sentence for
splits.
"""

class WeightedData:
    def __init__(self,sentences):
        """
        Initialisation of weightedData Object.

        :param sentences:   List of sentences to be processed for sample weights.
        """
        self.data_set = sentences
        self.sum1=0
        for sentence in self.data_set:
            sentence.weight = 1/len(self.data_set)
            self.sum1 += sentence.weight

        self.sum2=1

    def normalize(self):
        """
        Normalize the wieghts of available sentence Data so that they add upto 1
        and can be used for processing further.

        :return:
        """
        norm_val = self.sum2/self.sum1
        self.sum1=0

        for sentence in self.data_set:
            sentence.weight *= norm_val
            self.sum1 += sentence.weight

    def update_weight(self,ctr,new_weight):
        """
        Update the wieght of a specific sentence object.

        :param ctr:         index of object to be updated
        :param new_weight:  new weight to be updated.
        :return:
        """
        self.sum1 -= self.data_set[ctr].weight
        self.data_set[ctr].weight = new_weight
        self.sum1 += new_weight



