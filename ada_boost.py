"""
CSCI 630 : Foundations of Artificial Intelligence
Lab 2
Author: Sarthak Thakkar (st4070)

ada_boost.py

This Program contains AdaBoost class. It creates a model by
ada boosting the stumps(trees of height 1) and combinning them
based on the best information gain available and individual weights
assigned to them.
"""

import pickle
import math
from sentence import Sentence
from node import Node
from weighted_data import WeightedData

class AdaBoost:

    def __init__(self,example_file,out_file):
        """
        Initialising the Ada boost object

        :param example_file:    file with tagged sentences
        :param out_file:        hypothesis file to save object
        """
        self.sentence_data = self.process_file(example_file)
        self.out_file = out_file
        self.tree=''
        self.ensemble = []

        out_file_refresh = open(out_file,'w')
        out_file_refresh.close()

    def process_file(self,file):
        """
        To Get Sentence objects for all the lines in the file.
        :param file: path of the input_file
        :return: list of Sentence object attributes.
        """
        in_file =open(file,'r')
        line_list=[]

        for line in in_file:
            line_list.append(Sentence(line))

        return line_list

    def train(self,k=10):
        """
        Generate an ADA booseted ensemble and store it in a hypothesis.

        :param k    maximum number of stumps to be generated:
        :return:
        """
        train_data = self.sentence_data
        attributes = set(train_data[0].attributes.keys())
        weighted_data = WeightedData(train_data)
        self.ensemble =[]

        for ctr in range(k):
            stump = makeTree(train_data,attributes,[],1)
            error_sum=0

            for data in train_data:
                predicted_tag = stump.decide(data)

                if predicted_tag != data.tag:
                    error_sum += data.weight

            for i in range(len(train_data)):
                data = train_data[i]
                predicted_tag = stump.decide(data)

                if predicted_tag == data.tag:
                    new_weight = data.weight * (error_sum/(1-error_sum))
                    weighted_data.update_weight(i,new_weight)

            weighted_data.normalize()
            stump.weight = math.log((1-error_sum)/error_sum)
            self.ensemble.append(stump)

        out_file_obj = open(self.out_file,'wb')
        pickle.dump(self,out_file_obj)
        out_file_obj.close()

    def predict(self,test_file):
        """
        Predicts The language of every line in given file based on the decision tree Hypothesis.

        :param test_file: path of file containing untagged sentences.
        :return:
        """
        test_file = open(test_file, 'r')
        for line in test_file:
            line_data = Sentence(line, False)
            line_data.tag = self.majority_result(line_data)
            print(line_data.tag)

    def majority_result(self,sentence_data):
        """
        It gives the majority tag value by evaluating the weights of
        stumps and calculating the resultant stump with majority.

        :param sentence_data:   Data to be processed to find majority
        :return: tag with majority weight
        """
        count={}
        max_count =0
        result=None

        for stump in self.ensemble:
            predicted_tag = stump.decide(sentence_data)

            if predicted_tag in count.keys():
                count[str(predicted_tag)] += stump.weight
            else:
                count[str(predicted_tag)] = stump.weight

            if count[str(predicted_tag)] > max_count:
                max_count = count[str(predicted_tag)]
                result = predicted_tag

        return result

def get_best_gain(data,attribute_list):
    """
    Get the best attribute to split on based on information gain.

    :param data:            Sentence Data to be processed for split
    :param attribute_list:  List of attributes to select an attribute

    :return: best split attribute, remaining attributes for further splits.
    """
    final_entropy_data = {}
    child_lists={}
    for attribute in attribute_list:
        child_lists[str(attribute)]={}
        count1={}
        count2 ={}
        true_count = 0
        false_count = 0
        major_count_1=0
        minor_count_1=0
        major_count_2=0
        minor_count_2=0
        true_list = []
        false_list = []
        key_value = True
        for sentence in data:
            if sentence.attributes[attribute] == key_value:
                true_count += 1
                true_list.append(sentence)
            else:
                false_count+=1
                false_list.append(sentence)
        if len(true_list) > 0:
            for line in true_list:
                if line.tag not in count1.keys():
                    count1[str(line.tag)]=1
                else:
                    count1[str(line.tag)]+=1
            if len(count1.keys()) > 1:
                major_count_1 = count1['en']
                minor_count_1 = count1['nl']
            else:
                for key in count1.keys():
                    major_count_1 = count1[str(key)]
        if len(false_list) > 0:
            for line in false_list:
                if line.tag not in count2.keys():
                    count2[str(line.tag)] = 1
                else:
                    count2[str(line.tag)] += 1
            if len(count2.keys()) > 1:
                major_count_2 = count2['en']
                minor_count_2 = count2['nl']
            else:
                for key in count2.keys():
                    major_count_2 = count2[str(key)]

        total_count = true_count+false_count

        # true_entropy
        if true_count > 0:
            if major_count_1 > 0:
                if minor_count_1 > 0:

                    true_entropy = (true_count / total_count) * (
                            ((major_count_1 / true_count) * math.log2(1 / (major_count_1 / true_count))) + (
                            minor_count_1 / true_count) * math.log2(1 / (minor_count_1 / true_count)))
                else:
                    true_entropy = (true_count / total_count) * (
                        ((major_count_1 / true_count) * math.log2(1 / (major_count_1 / true_count))))
            else:
                true_entropy = (true_count / total_count) * (
                    ((minor_count_1 / true_count) * math.log2(1 / (minor_count_1 / true_count))))
        else:
            true_entropy = 0

        # false_entropy
        if false_count > 0:
            if major_count_2 > 0:
                if minor_count_2 > 0:
                    false_entropy = (false_count / total_count) * (
                            ((major_count_2 / false_count) * math.log2(1 / (major_count_2 / false_count))) + (
                            minor_count_2 / false_count) * math.log2(1 / (minor_count_2 / false_count)))
                else:
                    false_entropy = (false_count / total_count) * (
                        ((major_count_2 / false_count) * math.log2(1 / (major_count_2 / false_count))))
            else:
                false_entropy = (false_count / total_count) * (
                    ((minor_count_2 / false_count) * math.log2(1 / (minor_count_2 / false_count))))
        else:
            false_entropy = 0
        final_entropy = true_entropy + false_entropy
        # print(key,final_entropy)
        final_entropy_data[str(attribute)] = final_entropy
        # print('attribute ',attribute,'entropy',final_entropy,'true_count',true_count,'major_1',major_count_1,'minor_1',minor_count_1,'false_count',false_count,'major_2',major_count_2,'minor_2',minor_count_2,count1.keys(),count2.keys())
        child_lists[str(attribute)][True]=true_list
        child_lists[str(attribute)][False] = false_list
    final_entropy_data = sorted(final_entropy_data.items(), key=
    lambda kv: (kv[1], kv[0]))
    final_value = list(final_entropy_data)[0]
    # print('Best entropy is for ', final_value)
    return final_value[0],child_lists[str(final_value[0])]


def makeTree(train_data,attributes,parent_data,depth_ctr=15):
    """
    Recursive Function to create stumps until terminating condition
    are reached based on given weights.

    :param train_data:  Sentence object data to be trainned
    :param attributes:  List of attributes to find best split attribute
    :param parent_data: Return parent maximum data if an empty node is generated.
    :param depth_ctr:   Depth counter to control depth.

    :return: Ada Boosted Tree
    """
    if len(train_data) ==0:
        return Node(get_majority(parent_data),False)
    if len(attributes) ==0:
        return Node(get_majority(parent_data), False)

    best_label , child_dict = get_best_gain(train_data,attributes)
    root = Node(best_label)
    for value in child_dict:
        new_train_data = child_dict[value]
        subTree = Node(get_majority(new_train_data),False)
        root.insert(value, subTree)

    return root

def get_majority(data_collection):
    """
    Get the label tagged by majority sentence objects frmm
    the given list of objects.

    :param examples: Given list of Sentence Object to find majority label

    :return: Label tagged by majority objects of list.
    """
    count = {}
    for data in data_collection:
        if data.weight:
            weight = data.weight
        else:
            weight = 1
        if str(data.tag) in count.keys():
            count[str(data.tag)] += weight
        else:
            count[str(data.tag)] = weight

    count = sorted(count.items(), key=
    lambda kv: (kv[1], kv[0]), reverse=True)

    max_count_tuple = list(count)[0]
    max_count_value = max_count_tuple[0]
    max_count = max_count_tuple[1]

    return max_count_value