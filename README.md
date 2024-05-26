# DD2417 - Miniproject: Word Predictor

## Project Introduction
Most modern mobile phones have some kind of word prediction software. As you are typing a word in a text message or an e-mail, the word predictor displays a shortlist of the most probable completions of the word. The list of suggestions is updated for each keystroke the user makes. If one of the suggestions indeed is the word you intended to type, you can type the word just by clicking on it, thereby saving many keystrokes.

Such a system must have a language model to be able to suggest the next word. The task in
this project is to:
1. Implement word prediction using three different language models:
    - one based on n-gram models, 
    - one based on recurrent neural networks
    - one based on the transformer architecture.
2. Implement some simple GUI to show how your word prediction works
3. Evaluate by computing the proportion of saved keystrokes using different models and inputs

## Repository Content
The repository contains the following components:
- **Data**: The data files the models were trained on and how they were created
- **Training**: Code for training the different models:
    - _Trigram_
    - _RNN_: Code to train and evaluate, as well as best model and evaluation metrics plots
    - _Transformer_
- **GUI**: Source code for the simple graphical user interface (GUI) to showcase the word prediction functionality.


## Usage
To run the code:
- **Data**: Download data to your own repo
- **Training**: 
    - _Trigram_
    - _RNN_: Train.ipynb file, should be able to just run to achieve results presented 
    - _Transformer_: TrainTransformer.ipynb file, should be able to just run to achieve results presented 
- **GUI**: HOW TO RUN GUI

## Results
*Keystroke = Percentage of saved keystrokes

*Word = Percentage of found words

Here generating a list of 100 most probable words, displaying 5 and updating according to what user writes.


|             | **Trigram model** |                | **RNN**        |                | **Transformer** |                |
|-------------|:-----------------:|:--------------:|:--------------:|:--------------:|:---------------:|:--------------:|
|             |    *Keystroke*    |    *Word*      |    *Keystroke* |    *Word*      |    *Keystroke* |    *Word*      |
| 5 predictions |      22.45%       |     31.99%     |      28.40%    |     41.95%     |      28.97%     |     42.87%     |
| 20 predictions |     34.15%        |     52.26%     |      40.51%    |     59.23%     |      41.09%     |     60.94%     |
| 50 predictions |     41.09%        |     62.47%     |      48.91%    |     69.67%     |      49.71%     |     71.47%     |
| 100 predictions|     47.84%        |     71.67%     |      54.61%    |     76.44%     |      55.65%     |     78.09%     |



## Contributors
- Agaton Domberg: domberg@kth.se
- Lucia Karens: lkarens@kth.se
- Emilia Sjögren: emsjog@kth.se
