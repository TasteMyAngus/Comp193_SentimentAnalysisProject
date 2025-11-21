import nltk
import sklearn
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class SentimentAnalyzer:
    def __init__(self, data_path, train=True):
        # Download necessary NLTK resources
        nltk.download("punkt")
        nltk.download("punkt_tab")
        nltk.download("stopwords")
        # Define stop words, but keep some important negation and intensity words
        self.stop_words = set(stopwords.words("english")) - set(["not", "no", "nor", "never", "but", "very", "too", "n't"])
        self.stop_words = list(self.stop_words)
        # Load the dataset
        self.reviews = pd.read_csv(data_path, sep="\t")
        # Y source: labels
        self.y = self.reviews["Liked"].tolist()
        # Initialize MultinomialNB classifier
        self.clf = MultinomialNB()
        # Training flag
        self.train = train

    def split_into_sentences(self, text):
        return nltk.sent_tokenize(text)

    def split_sentences_into_words(self, sentence):
        return nltk.word_tokenize(sentence)

    def lowercase_words(self, sentence):
        cleaned_sentence = [word.lower() for word in sentence]
        return cleaned_sentence

    def remove_punctuation_from_word_list(self, word_list):
        cleaned_word_list = [word for word in word_list if word.isalnum()]
        return cleaned_word_list

    def vectorize_reviews(self, reviews):
        vectorizer = CountVectorizer(
            stop_words=self.stop_words,
            token_pattern=r'\b\w+\b'  # Token pattern to include single-character words
        )
        X = vectorizer.fit_transform(reviews)
        return X

    def split_data(self, X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test

    def preprocess(self):
        print("\nDefault Data:")
        print(self.reviews[0:2])

        # Split text into sentences
        self.reviews["Sentences"] = self.reviews["Review"].apply(self.split_into_sentences)
        print("\nAfter Splitting Into Sentences:")
        print(self.reviews[0:2])

        # Split sentences into words
        self.reviews["Words"] = self.reviews["Sentences"].apply(
            lambda sent_list: [self.split_sentences_into_words(s) for s in sent_list]
        )
        print("\nAfter Splitting Into Words:")
        print(self.reviews[0:2])

        # Remove punctuation elements from word lists
        self.reviews["Words"] = self.reviews["Words"].apply(
            lambda group: [self.remove_punctuation_from_word_list(sentence) for sentence in group]
        )
        print("\nAfter Removing Punctuation:")
        print(self.reviews[0:2])

        # Turn words to lowercase
        self.reviews["Words"] = self.reviews["Words"].apply(
            lambda group: [self.lowercase_words(sentence) for sentence in group]
        )
        print("\nAfter Making All Words Lowercase:")
        print(self.reviews[0:2])

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
