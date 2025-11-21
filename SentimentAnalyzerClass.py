import nltk
import sklearn
import pandas as pd
import re
from spellchecker import SpellChecker
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class SentimentAnalyzer:
    def __init__(self, data_path, train=True):
        # Download necessary NLTK resources
        nltk.download("punkt")
        nltk.download("punkt_tab")
        nltk.download("stopwords")
        # wordnet, omw-1.4, averaged_perceptron_tagger for lemmatization and POS tagging
        nltk.download("wordnet")
        nltk.download("omw-1.4")
        nltk.download("averaged_perceptron_tagger")
        # Define stop words, but keep some important negation and intensity words
        self.stop_words = set(stopwords.words("english")) - set(["not", "no", "nor", "never", "but", "very", "too"])
        self.stop_words = list(self.stop_words)
        # Load the dataset
        self.reviews = pd.read_csv(data_path, sep="\t")
        # Y source: labels
        self.y = self.reviews["Liked"].tolist()
        # Initialize MultinomialNB classifier
        self.clf = MultinomialNB()
        # Training flag
        self.train = train

        ###Testing things
        # Initialize spell checker
        self.spell_checker = SpellChecker()
        # Lemmatize
        self.lemmatizer = WordNetLemmatizer()


    def split_into_sentences(self, text):
        return nltk.sent_tokenize(text)

    def split_sentences_into_words(self, sentence):
        # split_sentence = nltk.word_tokenize(sentence)

        # Merge clitics using merge_clitics method
        split_sentence_merged = self.merge_clitics(nltk.word_tokenize(sentence))
        return split_sentence_merged

    # Merge clitics with preceding workd
    # [..., did, n't, ...] -> [..., didn't, ...]
    def merge_clitics(self, word_list):
        merged = []
        for word in word_list:
            if word in {"n't", "'re", "'ve", "'ll", "'d", "'s"} and merged:
                merged[-1] = merged[-1] + word
            else:
                merged.append(word)
        return merged

    def lowercase_words(self, sentence):
        cleaned_sentence = [word.lower() for word in sentence]
        # spell checker: if works, create method
        # misspelled = self.spell_checker.unknown(cleaned_sentence)
        # corrected_sentence = [
        #     self.spell_checker.correction(word) if word in misspelled else word
        #     for word in cleaned_sentence
        #     ]
        return cleaned_sentence

    def remove_punctuation_from_word_list(self, word_list):
        cleaned_word_list = [word for word in word_list if re.match(r"^[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?$", word)]
        return cleaned_word_list

    def vectorize_reviews(self, reviews):
        """
        vectorizer = CountVectorizer(
            stop_words=self.stop_words,
            token_pattern=r'\b\w+\b',  # Token pattern to include single-character words
            #ngram_range=(1, 2)
        )
        """

        # Scales weight of term down based on frequency
        vectorizer_tfid = TfidfVectorizer(
            stop_words=self.stop_words,
            token_pattern=r'\b\w+\b',  # Token pattern to include single-character words
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        X = vectorizer_tfid.fit_transform(reviews)
        return X

    def split_data(self, X, y, test_size, random_state):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test

    def preprocess(self):
        print("\nDefault Data:")
        print(self.reviews[6:7])

        # Split text into sentences
        self.reviews["Sentences"] = self.reviews["Review"].apply(self.split_into_sentences)
        print("\nAfter Splitting Into Sentences:")
        print(self.reviews[6:7])

        # Split sentences into words
        self.reviews["Words"] = self.reviews["Sentences"].apply(
            lambda sent_list: [self.split_sentences_into_words(s) for s in sent_list]
        )
        print("\nAfter Splitting Into Words:")
        print(self.reviews[6:7])
        
        # Remove punctuation elements from word lists
        self.reviews["Words"] = self.reviews["Words"].apply(
            lambda group: [self.remove_punctuation_from_word_list(sentence) for sentence in group]
        )
        
        print("\nAfter Removing Punctuation:")
        print(self.reviews[6:7])

        # Turn words to lowercase
        self.reviews["Words"] = self.reviews["Words"].apply(
            lambda group: [self.lowercase_words(sentence) for sentence in group]
        )
        print("\nAfter Making All Words Lowercase:")
        print(self.reviews[6:7])

        # Vectorize sentences
        X = self.vectorize_reviews(self.reviews["Words"].apply(lambda r: " ".join(word for sentence in r for word in sentence)))
        return X
    
    def train_model(self, X_train, y_train, X_test, y_test):
        self.clf.fit(X_train, y_train)
        y_pred = self.clf.predict(X_test)
        print("\nModel Evaluation:")
        print("\nAccuracy:", accuracy_score(y_test, y_pred))

    def run(self):
        X = self.preprocess()

        print("\nAfter Vectorization:")
        print(X.shape)
        print("\nNumber of labels: ", len(self.y))

        X_train, X_test, y_train, y_test = self.split_data(X, self.y, test_size=0.2, random_state=42)
        if self.train:
            self.train_model(X_train, y_train, X_test, y_test)
            

if __name__ == "__main__":
    analyzer = SentimentAnalyzer("Restaurant_Reviews.tsv", True)
    analyzer.run()
