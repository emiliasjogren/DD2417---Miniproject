#  -*- coding: utf-8 -*-
from __future__ import unicode_literals
import faulthandler
faulthandler.enable()
import argparse
from collections import defaultdict
import codecs
import re

"""
Train a Trigram language model given a text.
This program is based on the Bigram trainer file for assignment 2 of of the course DD2417 Language Engineering at KTH.

"""


class TrigramTrainer(object):
    """
    This class constructs a bigram language model from a corpus.
    """
    def __init__(self):
        """
        Constructor. Processes the file f and builds a language model
        from it.

        :param f: The training file.
        """

        # The mapping from words to identifiers.
        self.index = {}
       

        # The mapping from identifiers to words.
        self.word = {}

        # An array holding the unigram counts.
        self.unigram_count = defaultdict(int)


        """
        The trigram, bigram, and unigram counts. Since most of these are zero, we store these
        in a hashmap rather than an array to save space (and since it is impossible
        to create such a big array anyway).
        If the bigram is "I like", get the count with: self.bigram_count['i']['like']
        If the trigram is "I like to", get the count with: self.trigram_count['i like']['to']
        """
        self.bigram_count = defaultdict(lambda: defaultdict(int)) #nested dict
        self.trigram_count = defaultdict(lambda: defaultdict(int)) #nested dict
        
        # The identifier of the previous word processed.
        self.last_index = -1

        # The identifier of the pre-previous word processed. (two words previous)

        self.sec_last_index = -2
        
        # Number of unique words (word forms) in the training corpus.
        self.unique_words = 0

        # The total number of words in the training corpus.
        self.total_words = 0


    def process_files(self, f):
        """
        Processes the file f.
        """
        with open(f, 'r', encoding='utf-8', errors='ignore') as file:
            text = file.read().lower()
        clean_text = re.sub(r"[^a-zA-Z',.!?]", ' ', text)
        self.tokens = re.findall(r"\b\w+(?:'\w+)?\b|[?.!,]", clean_text) #Gör om ord med whitespace till höger och till vänster 
        self.total_words=len(self.tokens) #added this
        
        
        
        for token in self.tokens:
            self.process_token(token)

            


    def process_token(self, token):
        """
        Processes one word in the training corpus, and adjusts the unigram,
        bigram, and trigram counts.

        :param token: The current word to be processed.
        """
        
        #if we have not encountered this token before, update bigram, trigram and unigram counts
        
        if not self.last_index==-1: #if we're looking at the second word in the corpus and onwards
            self.bigram_count[self.word[self.last_index]][token]+=1
        if self.sec_last_index >=0: #if we have at least two words in the dict
            self.trigram_count[(self.word[self.sec_last_index],self.word[self.last_index])][token]+=1 
        
        if token not in self.unigram_count: 
            
            self.index[token]=self.unique_words #key: token, value: index
            self.word[self.unique_words]=token #key: index, value: word
            self.unique_words+=1
            self.unigram_count[token]+=1     
        else:
            self.unigram_count[token]+=1
        
        self.sec_last_index=self.last_index
        self.last_index=self.index[token]
      
       

    def stats(self):
        """
        Creates a list of rows to print of the language model.
        """
        rows_to_print = []
        rows_to_print.append(str(self.unique_words) + " " + str(self.total_words)) #first row with V N
        for token in self.word: #V lines with identifier, token, and # of times token appears
            rows_to_print.append(str(token) + " " + str(self.word[token]) + " " + str(self.unigram_count[self.word[token]]))
        
        for token_1 in self.bigram_count: #lines with all non-zero bigram probabilities
            for token_2 in self.bigram_count[token_1]:
                proba = self.bigram_count[token_1][token_2]/self.unigram_count[token_1]
                rounded = f'{proba:.15f}' #round to exactly 15 decimals (even if 15 zeros)
                rows_to_print.append(str(self.index[token_1]) + " " + str(self.index[token_2]) + " " + str(rounded))

        rows_to_print.append("-2") #to show we're switching to trigrams

        for token_1_2 in self.trigram_count:
            for token_3 in self.trigram_count[token_1_2]:
                token_1, token_2 = token_1_2
                #print(token_1_2)
                proba = self.trigram_count[token_1_2][token_3]/self.bigram_count[token_1][token_2]
                rounded = f'{proba:.15f}' #round to exactly 15 decimals (even if 15 zeros)
                rows_to_print.append(str(self.index[token_1]) + " " + str(self.index[token_2]) + " " + str(self.index[token_3]) + " " +str(rounded))

        rows_to_print.append("-1") #last row
        
        return rows_to_print



def main():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='TrigramTrainer')
    parser.add_argument('--file', '-f', type=str,  required=True, help='file from which to build the language model')
    parser.add_argument('--destination', '-d', type=str, help='file in which to store the language model')

    arguments = parser.parse_args()

    trigram_trainer = TrigramTrainer()

    trigram_trainer.process_files(arguments.file)
    print("file processed")

    stats = trigram_trainer.stats()
    if arguments.destination:
        with codecs.open(arguments.destination, 'w', 'utf-8' ) as f:
            for row in stats: f.write(row + '\n')
    else:
        for row in stats: print(row)
    

if __name__ == "__main__":
    main()
