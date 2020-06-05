"""
CSCI 630 : Foundations of Artificial Intelligence
Lab 2
Author: Sarthak Thakkar (st4070)

node.py

This Program contains Node class. Which is used for generating
trees in decision model and ada boosting.It stores weights of nodes,
attributes to splited on and the children branches[True,False]
pointing to following attributes and further splits.
"""

class Node:
    def __init__(self,tag,has_children=True):
        """
        Initialisation of Node Object.

        :param tag:             attribute value of the node that it has been splited on
        :param has_children:    Boolen for has further branches or not
        """
        self.tag=tag
        self.child_attributes ={}
        self.has_children = has_children
        self.weight = None

    def insert(self,tag,new_node):
        """
        Add branch to existing node.
        :param tag:         the attribute of new branch.
        :param new_node:    the new node to be attached.
        :return:
        """
        self.child_attributes[tag] = new_node

    def decide(self,line_data):
        """
        Predict the class of the given set of data based on the current tree.

        :param line_data: the line data to be processed.
        :return: Predicted tag of given Sentence data object.
        """
        node = self

        while node:
            if not node.has_children:
                return node.tag

            next_side = line_data.attributes[node.tag]

            if next_side in node.child_attributes:
                node = node.child_attributes[next_side]
            else:
                return majority_result(node)

        return None

def get_majority(examples):
    """
    Get the label tagged by majority sentence objects frmm
    the given list of objects.

    :param examples: Given list of Sentence Object to find majority label

    :return: Label tagged by majority objects of list.
    """
    count = {}
    for ex in examples:
        if str(ex.tag) in count.keys():
            count[str(ex.tag)] += 1
        else:
            count[str(ex.tag)] = 1

    count = sorted(count.items(), key=
    lambda kv: (kv[1], kv[0]), reverse=True)

    max_count_tuple = list(count)[0]
    max_count_value = max_count_tuple[0]
    max_count = max_count_tuple[1]

    return max_count_value

def majority_result(node):
    """
    It gives the majority tag value by evaluating the weights of
    stumps and calculating the resultant stump with majority.

    :param sentence_data:   Data to be processed to find majority
    :return: tag with majority weight
    """
    if not node.has_children:
        return node.tag

    if not node.child_attributes:
        return None

    count={}
    max_count=-1
    max_val = None

    for next_side in node.child_attributes:
        attr = majority_result(node.child_attributes[next_side])

        if not attr:
            continue

        if attr in count.keys():
            count[attr]+= 1
        else:
            count[attr]=1

        if count[attr] > max_val:
            max_count=count[attr]
            max_val=attr

    return max_val
