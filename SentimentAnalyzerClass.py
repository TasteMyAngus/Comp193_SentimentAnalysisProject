import nltk
import sklearn
import pandas as pd
import re
from spellchecker import SpellChecker
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class SentimentAnalyzer:
    def __init__(self, data_path, train=True, train_aspect_models=True):
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
        # Train aspect classifiers flag
        self.train_aspect_models = train_aspect_models

        # Lexicon-driven aspect configuration will be loaded from CSV
        self.aspect_config = None

        ###Testing things
        # Initialize spell checker
        self.spell_checker = SpellChecker()
        # Lemmatize
        self.lemmatizer = WordNetLemmatizer()
        # Hold aspect models and vectorizers
        self.aspect_models = {}
        self.aspect_vectorizers = {}

        # Load restaurant lexicon with sentiment
        try:
            self.load_lexicon("restaurant_lexicon_with_sentiment.csv")
        except Exception as e:
            print(f"Warning: failed to load lexicon CSV: {e}. Using empty lexicon.")
            self.lexicon_df = pd.DataFrame(columns=["term","aspect","sentiment"])
            self._build_aspect_config_from_lexicon()
        # Negation words and intensifiers for simple rules
        self.negators = {"not","no","never","none","n't"}
        self.intensifiers_pos = {"very","so","extremely","really","super","quite"}
        self.intensifiers_neg = {"too","barely","hardly"}


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
        # Scales weight of term down based on frequency
        vectorizer_tfid = TfidfVectorizer(
            stop_words=self.stop_words,
            token_pattern=r'\b\w+\b',  # Token pattern to include single-character words
            ngram_range=(1, 2),
            sublinear_tf=True,
            binary=True,
            #min_df=2,
            #max_df=.95,
        )
        
        X = vectorizer_tfid.fit_transform(reviews)
        return X
    def load_lexicon(self, csv_path):
        df = pd.read_csv(csv_path)
        # Normalize expected columns
        cols = {c.lower(): c for c in df.columns}
        # Expect: term, aspect, sentiment
        term_col = cols.get("term") or cols.get("word") or cols.get("token")
        aspect_col = cols.get("aspect") or cols.get("category")
        sent_col = cols.get("sentiment") or cols.get("polarity")
        if not term_col or not aspect_col or not sent_col:
            raise ValueError("CSV must contain columns for term, aspect, sentiment/polarity")
        df = df[[term_col, aspect_col, sent_col]].rename(columns={term_col:"term", aspect_col:"aspect", sent_col:"sentiment"})
        # Ensure lowercase and strip
        df["term"] = df["term"].astype(str).str.lower().str.strip()
        df["aspect"] = df["aspect"].astype(str).str.lower().str.strip()
        df["sentiment"] = df["sentiment"].astype(str).str.lower().str.strip()
        # Keep only known aspects
        df = df[df["aspect"].isin(["food","service","price","cost"])].copy()
        df.loc[df["aspect"]=="cost","aspect"] = "price"
        # Sentiment to numeric {positive:1, negative:0}
        df["label"] = df["sentiment"].map({"positive":1, "pos":1, "+":1, "negative":0, "neg":0, "-":0})
        # Drop rows with unknown label
        df = df.dropna(subset=["label"])
        # Lemmatize terms for better matching
        df["term_lemma"] = df["term"].apply(lambda t: self.lemmatizer.lemmatize(t))
        self.lexicon_df = df
        self._build_aspect_config_from_lexicon()

    def _build_aspect_config_from_lexicon(self):
        # Build aspect config structure from lexicon terms
        aspects = {a: {"keywords": set(), "phrases": set(), "windows": [5], "sent_threshold": 1} for a in ["food","service","price"]}
        for _, row in self.lexicon_df.iterrows():
            term = row["term"]
            aspect = row["aspect"]
            if " " in term:
                aspects[aspect]["phrases"].add(term)
            else:
                aspects[aspect]["keywords"].add(term)
        # Convert sets to lists
        for a in aspects:
            aspects[a]["keywords"] = list(aspects[a]["keywords"])
            aspects[a]["phrases"] = list(aspects[a]["phrases"])
        self.aspect_config = aspects

    def _vectorizer_for_aspect(self):
        return TfidfVectorizer(
            stop_words=self.stop_words,
            token_pattern=r'\b\w+\b',
            ngram_range=(1, 2),
            sublinear_tf=True,
            binary=True,
        )
    
    # Still needs to be integrated with lemmatization process
    def _wn_pos(treebank_tag):
            if treebank_tag.startswith('J'):
                return wordnet.ADJ
            elif treebank_tag.startswith('V'):
                return wordnet.VERB
            elif treebank_tag.startswith('N'):
                return wordnet.NOUN
            elif treebank_tag.startswith('R'):
                return wordnet.ADV
            else:
                return wordnet.NOUN


    # aspect presence methods
    def detect_aspect_presence(self, review_text, aspect):
        cfg = self.aspect_config[aspect]
        low = review_text.lower()
        # Tokenize for exact word matching
        token_list = low.split()
        tokens = set(token_list)
        # match keywords by token
        for kw in cfg["keywords"]:
            if kw in tokens:
                return 1
        # Phrase match by exact n-gram sequence
        for phrase in cfg["phrases"]:
            phrase_tokens = phrase.split()
            n = len(phrase_tokens)
            if n == 0:
                continue
            for i in range(0, len(token_list) - n + 1):
                if token_list[i:i+n] == phrase_tokens:
                    return 1
        return 0

    def build_presence_label(self):
        self.aspect_presence = {}
        for aspect in self.aspect_config:
            self.aspect_presence[aspect] = [self.detect_aspect_presence(txt, aspect) for txt in self.cleaned_texts]

    def train_aspect_classifiers(self):
        # Train a separate MultinomialNB per aspect on reviews where aspect is present
        self.aspect_models = {}
        self.aspect_vectorizers = {}
        for aspect in self.aspect_config.keys():
            presence = self.aspect_presence.get(aspect, [])
            if not presence:
                continue
            # Filter texts and labels where presence == 1
            texts = [t for t, p in zip(self.cleaned_texts, presence) if p == 1]
            labels = [label for label, p in zip(self.y, presence) if p == 1]
            if len(texts) < 5:
                # Not enough samples; skip training
                continue
            vect = self._vectorizer_for_aspect()
            X = vect.fit_transform(texts)
            X_train, X_test, y_train, y_test = self.split_data(X, labels, test_size=0.2, random_state=42)
            clf = MultinomialNB()
            clf.fit(X_train, y_train)
            self.aspect_models[aspect] = clf
            self.aspect_vectorizers[aspect] = vect
            # Simple report
            y_pred = clf.predict(X_test)
            print(f"\n[{aspect}] samples: {len(texts)} | Accuracy: {accuracy_score(y_test, y_pred):.3f}")

    def predict_aspect_sentiment(self, review_text):
        # Prepare single-review text like prepare_review_texts
        sentences = self.split_into_sentences(review_text)
        sent_words = [self.split_sentences_into_words(s) for s in sentences]
        sent_words = [self.remove_punctuation_from_word_list(w) for w in sent_words]
        sent_words = [self.lowercase_words(w) for w in sent_words]
        flat = " ".join(word for sentence in sent_words for word in sentence)
        flat_lemmas = " ".join(self.lemmatizer.lemmatize(w) for w in flat.split())
        token_list = flat.split()
        tokens = set(token_list)
        results = {}
        # Lexicon-based sentence-level scoring: sum positive/negative hits per aspect
        for aspect in self.aspect_config.keys():
            # Gather aspect terms
            lex_rows = self.lexicon_df[self.lexicon_df["aspect"] == aspect]
            pos_terms = set(lex_rows[lex_rows["label"] == 1]["term"].tolist())
            neg_terms = set(lex_rows[lex_rows["label"] == 0]["term"].tolist())
            # Also use lemmatized variants
            pos_terms_lem = set(lex_rows[lex_rows["label"] == 1]["term_lemma"].tolist())
            neg_terms_lem = set(lex_rows[lex_rows["label"] == 0]["term_lemma"].tolist())
            # Presence gate
            presence = self.detect_aspect_presence(flat, aspect)
            if presence == 0 and len(pos_terms | neg_terms) > 0:
                results[aspect] = "none"
                continue
            # Count hits using tokens/lemmas and exact phrase n-grams
            pos_hits = sum((term in tokens) for term in pos_terms) + sum((term in flat_lemmas) for term in pos_terms_lem)
            neg_hits = sum((term in tokens) for term in neg_terms) + sum((term in flat_lemmas) for term in neg_terms_lem)
            # Phrase n-grams
            def count_phrase_hits(phrases):
                c = 0
                for ph in phrases:
                    ph_toks = ph.split()
                    n = len(ph_toks)
                    for i in range(0, len(token_list) - n + 1):
                        if token_list[i:i+n] == ph_toks:
                            c += 1
                return c
            pos_hits += count_phrase_hits([t for t in pos_terms if " " in t])
            neg_hits += count_phrase_hits([t for t in neg_terms if " " in t])

            # Simple negation handling: if a negator appears within window of a hit, flip polarity
            window = 3
            def has_negator_near(term_tokens):
                n = len(term_tokens)
                for i in range(0, len(token_list) - n + 1):
                    if token_list[i:i+n] == term_tokens:
                        left = max(0, i - window)
                        right = min(len(token_list), i + n + window)
                        if any(tok in self.negators for tok in token_list[left:right]):
                            return True
                return False
            # Adjust hits for negation
            for ph in [t for t in pos_terms if " " in t]:
                if has_negator_near(ph.split()):
                    pos_hits -= 1
                    neg_hits += 1
            for ph in [t for t in neg_terms if " " in t]:
                if has_negator_near(ph.split()):
                    neg_hits -= 1
                    pos_hits += 1
            for t in pos_terms:
                if " " not in t and t in tokens:
                    idxs = [i for i, tok in enumerate(token_list) if tok == t]
                    for i in idxs:
                        left = max(0, i - window)
                        right = min(len(token_list), i + 1 + window)
                        if any(tok in self.negators for tok in token_list[left:right]):
                            pos_hits -= 1
                            neg_hits += 1
            for t in neg_terms:
                if " " not in t and t in tokens:
                    idxs = [i for i, tok in enumerate(token_list) if tok == t]
                    for i in idxs:
                        left = max(0, i - window)
                        right = min(len(token_list), i + 1 + window)
                        if any(tok in self.negators for tok in token_list[left:right]):
                            neg_hits -= 1
                            pos_hits += 1

            # Intensifiers: slightly weight hits
            if any(tok in self.intensifiers_pos for tok in token_list):
                pos_hits += 1
            if any(tok in self.intensifiers_neg for tok in token_list):
                neg_hits += 1
            if pos_hits == 0 and neg_hits == 0:
                # Fall back to ML model if available
                if aspect in self.aspect_models:
                    vect = self.aspect_vectorizers[aspect]
                    clf = self.aspect_models[aspect]
                    X = vect.transform([flat])
                    pred = clf.predict(X)[0]
                    results[aspect] = "positive" if pred == 1 else "negative"
                else:
                    results[aspect] = "none"
            else:
                results[aspect] = "positive" if pos_hits >= neg_hits else "negative"
        return results


    def split_data(self, X, y, test_size, random_state):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train, X_test, y_test):
            self.clf.fit(X_train, y_train)
            y_pred = self.clf.predict(X_test)
            print("\nModel Evaluation:")
            print("\nAccuracy:", accuracy_score(y_test, y_pred))


    # deprecated method
    # now using prepare_review_texts
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

    def prepare_review_texts(self):
        texts = []
        self.tokenized_reviews = []
        for review in self.reviews["Review"]:
            sentences = self.split_into_sentences(review)
            sent_words = [self.split_sentences_into_words(s) for s in sentences]
            sent_words = [self.remove_punctuation_from_word_list(w) for w in sent_words]
            sent_words = [self.lowercase_words(w) for w in sent_words]
            flat = " ".join(word for sentence in sent_words for word in sentence)
            texts.append(flat)
            self.tokenized_reviews.append(sent_words)
        self.cleaned_texts = texts
    
    

    def run(self):
        self.prepare_review_texts()
        X = self.vectorize_reviews(self.cleaned_texts)

        print("\nAfter Vectorization:")
        print(X.shape)
        print("\nNumber of labels: ", len(self.y))

        X_train, X_test, y_train, y_test = self.split_data(X, self.y, test_size=0.2, random_state=42)
        if self.train:
            self.train_model(X_train, y_train, X_test, y_test)

        # Build aspect presence labels and optionally train aspect-specific classifiers
        self.build_presence_label()
        for aspect, presence in self.aspect_presence.items():
            print(f"\nAspect '{aspect}' presence count: {sum(presence)} of {len(presence)}")

        if self.train_aspect_models:
            self.train_aspect_classifiers()
            # Demo prediction on first review
            if len(self.reviews) > 0:
                sample_text = self.reviews.loc[0, "Review"]
                preds = self.predict_aspect_sentiment(sample_text)
                print("\nSample aspect predictions on first review:")
                for a, v in preds.items():
                    print(f" - {a}: {v}")
            

if __name__ == "__main__":
    analyzer = SentimentAnalyzer("Restaurant_Reviews.tsv", True)
    analyzer.run()
