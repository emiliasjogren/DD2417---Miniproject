
from collections import deque
import TrigramPredictor as tripred
import re

"""
Evaluating the accuracy of a trigram predictor.

"""

def process_files( f):
        """
        Processes the file f.
        """
        with open(f, 'r', encoding='utf-8', errors='ignore') as file:
            text = file.read().lower()
        clean_text = re.sub(r"[^a-zA-Z',.!?]", ' ', text)
        tokens = re.findall(r"\b\w+(?:'\w+)?\b|[?.!,]", clean_text) 
        return tokens 

eval_text=process_files('test_file.txt')

trigram_predictor = tripred.TrigramPredictor()
trigram_predictor.read_model("train_file_model.txt")

def evaluate_text(eval_text, number_of_options=5):
        """ evaluate the number of correctly predicted words and keystrokes given a test text"""
        start = eval_text[0]
        saved_keystrokes = 0
        total_keystrokes = 0

        found_words = 0
        total_words = 0

        window=deque([start], maxlen=2) #only looking at the two preceding words since this is a trigram model

        for word in eval_text[1:]: #the word is the word we seek to predict
            options = trigram_predictor.predict_trigram(window, number_of_options)
                       
            (found_word, saved_strokes) = evaluate_options(word, options, 5)
            
            found_words += found_word
            saved_keystrokes += saved_strokes
            total_words += 1
            total_keystrokes += len(word)
            
            window.append(word)
        
        
        print(f'Saved keystroke percentage: {100*saved_keystrokes/total_keystrokes:.2f}%')
        print(f'Found words percentage: {100*found_words/total_words:.2f}%')
        
def evaluate_options(target_word, options, nr_of_suggestions = 5):
        """Target word is the next word, options is a list of the k most probable words according to the model
        Outputs found_word which is 1 if the target_word is in options else 0
        Outputs saved_keystrokes which is the amount of saved keystrokes, 0 if target_word not in options"""
        if target_word in options:
                if target_word in options[:nr_of_suggestions]:
                    found_word = 1
                    saved_keystrokes = len(target_word)
                    return found_word, saved_keystrokes
                for len_of_word in range(len(target_word)+1):
                    options = [word for word in options if word[:len_of_word] == target_word[:len_of_word]]
                    if target_word in options[:nr_of_suggestions]:
                        found_word = 1
                        saved_keystrokes = len(target_word) - len_of_word
                        return found_word, saved_keystrokes
                return(0,0) #if we went through the entire word before finding it
        else:
                found_word = 0
                saved_keystrokes = 0
                return found_word, saved_keystrokes

for i in range(10):
    evaluate_text(eval_text,100)


