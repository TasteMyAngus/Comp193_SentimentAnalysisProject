# From nltk library, we will use word_tokenize and sent_tokenize functions

import nltk

class SentimentAnalyzer:
    def __init__(self):
        self.self = self
        nltk.download('punkt')


    def text_tokenize(self, text):
        sentences = nltk.sent_tokenize(text)
        return sentences
    
    def clean_sentence(self, sentence):
        cleaned_sentence = sentence.replace
    
    def sentence_tokenize(self, sentence):
        words = nltk.word_tokenize(sentence)
        return words
    

def main():
    