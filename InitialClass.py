# From nltk library, we will use word_tokenize and sent_tokenize functions

import nltk
import pandas as pd

nltk.download("punkt_tab")

reviews = pd.read_csv("Restaurant_Reviews.tsv", sep="\t")

def split_into_sentences(text):
    return nltk.sent_tokenize(text)

def split_sentences_into_words(sentence):
    return nltk.word_tokenize(sentence)

def lowercase_words(sentence):
    cleaned_sentence = [word.lower() for word in sentence]
    return cleaned_sentence
    
# Print first two lines of data before any changes
print("\nDefault Data:")
print(reviews[0:2])

# Split text into sentences and print first two lines of data
reviews["Sentences"] = reviews["Review"].apply(split_into_sentences)
print("\nAfter Splitting Into Sentences:")
print(reviews[0:2])

# Split sentences into words and print first two lines of data
reviews["Words"] = reviews["Sentences"].apply(
    lambda sent_list: [split_sentences_into_words(s) for s in sent_list]
)
print("\nAfter Splitting Into Words:")
print(reviews[0:2])

# Turn words to lowercase and print first few lines
reviews["Words"] = reviews["Words"].apply(
    lambda group: [lowercase_words(sentence) for sentence in group]
)
print("\nAfter Making All Words Lowercase:")
print(reviews[0:2])
