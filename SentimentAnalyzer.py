# From nltk library, we will use word_tokenize and sent_tokenize functions

import nltk
import sklearn
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

##GLOBALS##

# Define stop words, but keep some important negation and intensity words
stop_words = set(stopwords.words("english")) - set(["not", "no", "nor", "never", "but", "very", "too", "n't"])
stop_words = list(stop_words)

# Load the dataset
reviews = pd.read_csv("Restaurant_Reviews.tsv", sep="\t")

# Y source: labels
y = reviews["Liked"].tolist()

##FUNCTIONS##

def split_into_sentences(text):
    return nltk.sent_tokenize(text)

def split_sentences_into_words(sentence):
    return nltk.word_tokenize(sentence)

def lowercase_words(sentence):
    cleaned_sentence = [word.lower() for word in sentence]
    return cleaned_sentence

def remove_punctuation_from_word_list(word_list):
    cleaned_word_list = [
        word for word in word_list if word.isalnum()
    ]
    return cleaned_word_list

# Turn reviews into bag-of-words
def vectorize_reviews(reviews):
    vectorizer = CountVectorizer(
        stop_words=stop_words,
        token_pattern=r'\b\w+\b'  # Token pattern to include single-character words
    )
    X = vectorizer.fit_transform(reviews)
    return X

# Split into training and testing sets
def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def main():

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

    # Remove punctuation elementsfrom word lists and print first few lines
    reviews["Words"] = reviews["Words"].apply(
        lambda group: [remove_punctuation_from_word_list(sentence) for sentence in group]
    )
    print("\nAfter Removing Punctuation:")
    print(reviews[0:2])

    # Turn words to lowercase and print first few lines
    reviews["Words"] = reviews["Words"].apply(
        lambda group: [lowercase_words(sentence) for sentence in group]
    )
    print("\nAfter Making All Words Lowercase:")
    print(reviews[0:2])

    # Vectorize sentences and print the shape of the resulting matrix
    X = vectorize_reviews(reviews["Words"].apply(lambda r: " ".join(word for sentence in r for word in sentence)))
    print("\nAfter Vectorization:")
    print(X.shape)
    print("\nNumber of labels: ", len(y))

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    
    clf = MultinomialNB()
    # learn from train set
    clf.fit(X_train, y_train)
    # predict on test set
    y_pred = clf.predict(X_test)

if __name__ == "__main__":
    main()