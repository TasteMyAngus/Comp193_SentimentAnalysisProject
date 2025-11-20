# From nltk library, we will use word_tokenize and sent_tokenize functions

import nltk
import pandas as pd

nltk.download("punkt_tab")

reviews = pd.read_csv("Restaurant_Reviews.tsv", sep="\t")


def text_tokenize(text):
    # sentences = nltk.sent_tokenize(text)
    return nltk.sent_tokenize(text)

def clean_sentence(sentence):
    cleaned_sentence = sentence.replace

def sentence_tokenize(sentence):
    words = nltk.word_tokenize(sentence)
    return words
    

print("Before Anything")
print(reviews.head())
print("\n\n")

reviews["Sentences"] = reviews["Review"].apply(text_tokenize)
print("After Sentence split")
print(reviews.head())
print("\n\n")

reviews["Words"] = reviews["Sentences"].apply(
    lambda sent_list: [sentence_tokenize(s) for s in sent_list]
)

print("After Word split")
print(reviews[0:1])
print("\n\n")
