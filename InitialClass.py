from nltk.tokenize import sent_tokenize
import nltk

class SentimentAnalyzer:
    def __init__(self):
        self.self = self
        nltk.download('punkt')


    def text_tokenize(self, text):
        sentences = sent_tokenize(text)
        return sentences
    
    def clean_sentence(self, sentence):
        cleaned_sentence = sentence.replace
    
    def sentence_tokenize(self, sentence):
        words = nltk.word_tokenize(sentence)
        return words
    

def main():
    