
import argparse
import codecs
from collections import defaultdict
import random

"""
Trigram-based word predictor.
This file is based on the computer assignment 2 for the course DD2417 Language engineering at KTH. It has been augmented to predict trigrams.
"""

class TrigramPredictor(object) :
    """
    This class generates predicted words from a language model.
    """
    def __init__(self):
    
        # The mapping from words to identifiers.
        self.index = {}

        # The mapping from identifiers to words.
        self.word = {}

        # An array holding the unigram counts.
        self.unigram_count = {}

        # The bigram probabilities.
        self.bigram_prob = defaultdict(dict)

        # The trigram probabilities
        self.trigram_prob = defaultdict(dict)

        # Number of unique words (word forms) in the training corpus.
        self.unique_words = 0

        # The total number of words in the training corpus.
        self.total_words = 0

        # The identifier of the previous word processed in the test corpus. Is -1 if the last word was unknown.
        self.last_index = -1

        # The identifier of the second previous word processed in the test corpus. Is -1 if the last word was unknown.
        self.sec_last_index =-1

        # The number of words processed in the test corpus.
        self.test_words_processed = 0

        #if we find ourselves stuck in a bigram probability loop (this happens when the bigram probabilites only predict words that were already predicted with the trigram probabilites)
        self.stuck_in_loop = 0 


    def read_model(self,filename):
        """
        Reads the contents of the language model file into the appropriate data structures.

        :param filename: The name of the language model file.
        :return: <code>true</code> if the entire file could be processed, false otherwise.
        """

        try:
            with codecs.open(filename, 'r', 'utf-8') as f:
                self.unique_words, self.total_words = map(int, f.readline().strip().split(' '))
                
                for i in range(self.unique_words): #the lines with words and indexes and counts
                    index, word, count= map(str, f.readline().strip().split(' '))
                    self.index[word]=int(index)
                    self.word[int(index)]=word
                    self.unigram_count[word]=int(count)
                while True: #the non-zero bigram probs
                    line=f.readline()                    
                    if line =="-2\n": #if we've reached the end of the bigram probs
                        break
                    else:
                        token_1, token_2, prob = map(float, line.strip().split(' '))
                        self.bigram_prob[int(token_1)][int(token_2)]=prob 
                while True: #the non-zero trigram probs
                    line=f.readline()                    
                    if line =="-1\n": #if we've reached the end of the doc
                        break
                    else:
                        token_1, token_2, token_3, prob = map(float, line.strip().split(' '))
                        self.trigram_prob[(int(token_1), int(token_2))][int(token_3)]=prob 
                
                return True
        
        except IOError:
            print("Couldn't find trigram probabilities file {}".format(filename))
            return False

    def predict_trigram(self, w, n=5):
        """
        Predicts and prints n different words, based on the input list of words w, and sampling from the distribution
        of the language model. Since this is a trigram model we only look at the last two words (Markov assumption). 
        This method works best with large texts. 
        """
        
        previous_words = []
        choices =[] #our final generated text
        for word in w:
            previous_words.append(word)
        
        i=0 #count the number of words in the choices list
        switched_to_bigrams=False
        j=0 #count the number of words in the choices list that come from bigrams
        self.stuck_in_loop=0 
        while i<n:
                        
            if len(previous_words)>1: #if we have at least two words in our input sentence
                #if both words are known and there is a trigram prob and we have not yet gone through all possible trigram options
                if (previous_words[-2] in self.index and previous_words[-1] in self.index) and ((self.index[previous_words[-2]], self.index[previous_words[-1]]) in self.trigram_prob) and i<len(list(self.trigram_prob[(self.index[previous_words[-2]], self.index[previous_words[-1]])].values())):
    
                    probas = list(self.trigram_prob[(self.index[previous_words[-2]], self.index[previous_words[-1]])].values())
                    #choose a word based on weighted random choice (probas are weights)
                    next_word = self.word[random.choices(list(self.trigram_prob[self.index[previous_words[-2]],self.index[previous_words[-1]]].keys()), weights=probas, k=1)[0]]
                else: #if there isn't a trigram prob
                    switched_to_bigrams=True
                    next_word=self.predict_bigram(previous_words,j)
                    
            else: #if we've only got one word in our input sentence
                switched_to_bigrams=True
                next_word=self.predict_bigram(previous_words,j)
                
            if self.stuck_in_loop>5000: #if we've spent too much time in bigrams, switch to unigrams and choose a random word weighted with unigram frequencies
            #we force quit the bigrams after 5000 loops to gain time. We therefore assume that it takes less than 5000 tries to get all available bigrams.
                probas=list(self.unigram_count.values())
            
                next_word = random.choices(list(self.unigram_count.keys()), weights=probas, k=1)[0]

            if next_word not in choices:
                
                choices.append(next_word)
                i+=1
                if switched_to_bigrams==True:
                    j+=1
           
       
        return(choices)

    def predict_bigram(self, previous_words,j):
        #if the word is known and there is a bigram prob and we haven't yet gone through all bigram options
        if previous_words[-1] in self.index and self.index[previous_words[-1]] in self.bigram_prob and j<len(list(self.bigram_prob[self.index[previous_words[-1]]].values())):  
            self.stuck_in_loop +=1 #add this failsafe so that we don't endlessly loop around in bigram probas    
            probas=list(self.bigram_prob[self.index[previous_words[-1]]].values())
            #choose a word based on weighted random choice (probas are weights)
            next_word = self.word[random.choices(list(self.bigram_prob[self.index[previous_words[-1]]].keys()), weights=probas, k=1)[0]]
            #print(next_word)
                
                
        else: #if the word doesn't have a bigram prob and/or is unknown, choose a random word weighted with unigram frequencies.
                        
            probas=list(self.unigram_count.values())
            
            next_word = random.choices(list(self.unigram_count.keys()), weights=probas, k=1)[0]
            #print(next_word)
            
            
        return(next_word)      


 
def main():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Trigram Predictor')
    parser.add_argument('--file', '-f', type=str,  required=True, help='file with language model')
    parser.add_argument('--input', '-i', type=list, required=True, help='user input')
    parser.add_argument('--number_of_predictions', '-n', type=int, default=5)

    arguments = parser.parse_args()
    
    predictor = TrigramPredictor()
    predictor.read_model(arguments.file)
    predictor.predict_trigram(arguments.input,arguments.number_of_predictions)

if __name__ == "__main__":
    main()

# example command prompt line: python TrigramPredictor.py -f train_file_model.txt -i [thank] -n 5