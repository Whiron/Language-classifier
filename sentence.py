"""
CSCI 630 : Foundations of Artificial Intelligence
Lab 2
Author: Sarthak Thakkar (st4070)

sentence.py

This Program contains Sentence class. Which extracts attributes
from the given statements which can be further used for training
or prediction of sentence.
"""

import string
import re

class Sentence:

    def __init__(self,line,tagged=True):
        """
        Initialisation of Sentence object
        :param line:    line to be processed
        :param tagged:  is it tagged already or not
        """
        table = str.maketrans(dict.fromkeys(string.punctuation))
        self.weight=None
        if tagged:
            contents = line.split('|')
            self.tag = contents[0]
            self.line = contents[1].translate(table)
        else:
            self.tag=None
            self.line = line.translate(table)

        self.attributes=self.get_attributes(self.line)
        # print(self.attributes)

    def get_attributes(self,data_line):
        """
        extract attributes as key-value pairs for given line.

        :param data_line: line to be processed
        :return: key-values pairs of attributes.
        """
        consonent_count = 0
        ee_val = False
        aa_val = False
        ij_val = False
        en_count = 0
        oo_val = False
        ing_val = False
        nan_val = False
        ae_val = False
        cons_count = 0
        vowel_count = 0
        de_count = 0
        vow_isto_cons = 0.0
        data_line_value = data_line
        value_list = {}
        double_consonant = re.compile("([bcdfghjklmnpqrstvwxz])(\\1)", re.I)
        double_ee = re.compile("([e])(\\1)", re.I)
        double_aa = re.compile("([a])(\\1)", re.I)
        occ_ij = re.compile("(ij)", re.I)
        ends_en = re.compile("(en)(\Z)", re.I)
        double_oo = re.compile("([o])(\\1)", re.I)
        ends_ing = re.compile("(ing)(\Z)", re.I)
        vowel_list = ['a', 'e', 'i', 'o', 'u']
        ends_ae = re.compile("(ae)(\Z)", re.I)
        starts_de = re.compile("(\A)(de)", re.I)

        # Check repeated consonenets:
        for word in data_line_value.split(' '):
            ans = double_consonant.search(word)
            if ans != None:
                consonent_count += 1
        if consonent_count > 1:
            value_list['rep-cons'] = True
        else:
            value_list['rep-cons'] = False

        # check for ee
        for word in data_line_value.split(' '):
            ans = double_ee.search(word)
            if ans != None:
                ee_val = True
        value_list['ee_val'] = ee_val

        # check for aa
        for word in data_line_value.split(' '):
            ans = double_aa.search(word)
            if ans != None:
                aa_val = True
        value_list['aa_val'] = aa_val

        # check for ij
        for word in data_line_value.split(' '):
            ans = occ_ij.search(word)
            # print(ans)
            if ans != None:
                ij_val = True
        value_list['ij_val'] = ij_val

        # ends in en
        for word in data_line_value.split(' '):
            ans = ends_en.search(word)
            if ans != None:
                en_count += 1
        if en_count > 1:
            value_list['en_count'] = True
        else:
            value_list['en_count'] = False

        # check for oo
        for word in data_line_value.split(' '):
            ans = double_oo.search(word)
            if ans != None:
                oo_val = True
        value_list['oo_val'] = oo_val

        # ends in ing
        for word in data_line_value.split(' '):
            ans = ends_ing.search(word)
            if ans != None:
                ing_val = True
        value_list['ing_val'] = ing_val

        # is Nan
        for word in data_line_value.split(' '):
            ans = word.isalnum()
            if ans == False and word != '\n':
                nan_val = True
        if nan_val > 0:
            value_list['nan_val'] = True
        else:
            value_list['nan_val'] = False

        # count consonant is to vowel ratio
        for Word in data_line_value.split(' '):
            if Word.isalnum():
                for word in Word:
                    if word in vowel_list:
                        vowel_count += 1
                    if word not in vowel_list and word.isalnum():
                        cons_count += 1
        vow_isto_cons = vowel_count / cons_count
        if vow_isto_cons > 0.7:
            value_list['vow_isto_cons'] = True
        else:
            value_list['vow_isto_cons'] = False

        # ends in ae
        for word in data_line_value.split(' '):
            ans = ends_ae.search(word)
            if ans != None:
                ae_val = True
        value_list['ae_val'] = ae_val

        # check for words starting with de
        for word in data_line_value.split(' '):
            ans = starts_de.search(word)
            if ans != None:
                de_count += 1
        if de_count >1 :
            value_list['de_count'] = True
        else:
            value_list['de_count'] = False

        return value_list