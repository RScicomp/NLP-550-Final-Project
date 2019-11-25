import numpy as np
import random
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt

np.random.seed(500)


def getData(file_path):
    """
    @param: file_path, str, path to the data file
    @return: an np array with each element as a line in the file
    """
    with open(file_path, errors="replace") as f:
        lines = f.readlines()
        dd = np.array(lines)
        return dd[1:]


def getCorpus(dataArray, threshold, target):
    total = dataArray.shape[0]
    firstGood = int(threshold*total)
    corpus = {"text": [], "label": []}
    corpus["text"] = np.array(dataArray)
    corpus["label"] = np.array(
        ["1" for i in range(firstGood)] + ["0" for i in range(total-firstGood)], dtype="str"
    )
    return corpus


# preprocessing
def preprocess_entry(entry, steps=("lower", "lemmatize", "stopwords")):
    if "lower" in steps:
        entry = entry.lower()
        entry = word_tokenize(entry)

    if "stopwords" in steps:
        entry = [
            word
            for word in entry
            if word not in stopwords.words("english") and word.isalpha()
        ]

    if ("lemmatize" in steps) and "stem" in steps:
        print("cannot use stem and lemmatization together")
        exit()
    elif "lemmatize" in steps:
        tag_map = defaultdict(lambda: wn.NOUN)
        tag_map["J"] = wn.ADJ
        tag_map["V"] = wn.VERB
        tag_map["R"] = wn.ADV
        # Initializing WordNetLemmatizer()
        word_Lemmatized = WordNetLemmatizer()
        # pos_tag function below will provide the 'tag'
        # i.e if the word is Noun(N) or Verb(V) or something else.
        Final_words = [
            word_Lemmatized.lemmatize(word, tag_map[tag[0]])
            for word, tag in pos_tag(entry)
        ]
    elif "stem" in steps:
        ps = PorterStemmer()
        Final_words = [ps.stem(word) for word in entry]
    else:
        Final_words = entry
    return str(Final_words)


def getVectorizer(corpus, method="tfidf"):
    """
    @param: corpus, map, with key 'final_text' map to the corpus used in training
    @return: a fitted vectorizer
    """
    vectorize_options = ("tfidf", "count")
    if method == vectorize_options[0]:
        vectorizer = TfidfVectorizer(max_features=10000)
        vectorizer.fit(corpus["final_text"])
        return vectorizer


def baseline_predict():
    roll = random.randint(1, 101)
    if roll > 50:
        return 0
    else:
        return 1


def baseline_predictor(corpus, test_size=0.3, title=None):
    Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(
        corpus["final_text"], corpus["label"], test_size=test_size
    )
    Encoder = LabelEncoder()
    Train_Y = Encoder.fit_transform(Train_Y)
    Test_Y = Encoder.fit_transform(Test_Y)
    predictions = [baseline_predict() for i in Test_Y]
    # Use accuracy_score function to get the accuracy
    final_accuracy = accuracy_score(predictions, Test_Y) * 100
    print("Baseline classifier", "Accuracy Score -> ", final_accuracy)
    return final_accuracy


def pipeline(corpus, vectorizer, test_size=0.3, classifier="NB", title=None):
    cls_options = ("NB", "SVM", "logistic")
    output_cls = {"NB": "Naive Bayes", "SVM": "SVM", "logistic": "logistic regression"}
    # split test and training set
    Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(
        corpus["final_text"], corpus["label"], test_size=test_size
    )
    Encoder = LabelEncoder()
    Train_Y = Encoder.fit_transform(Train_Y)
    Test_Y = Encoder.fit_transform(Test_Y)
    Train_X_vec = vectorizer.transform(Train_X)
    Test_X_vec = vectorizer.transform(Test_X)
    # fit the training dataset on the NB classifier
    if classifier == cls_options[0]:
        cls = naive_bayes.MultinomialNB()
    elif classifier == cls_options[1]:
        cls = svm.SVC(C=1.0, kernel="linear", degree=3, gamma="auto")
    elif classifier == cls_options[2]:
        cls = LogisticRegression()
    else:
        print("input classifier not supported")
        exit()
    
    cls.fit(Train_X_vec, Train_Y)
    # predict the labels on validation dataset
    predictions = cls.predict(Test_X_vec)
    # Use accuracy_score function to get the accuracy
    final_accuracy = accuracy_score(predictions, Test_Y) * 100
    print(output_cls[classifier], "Accuracy Score -> ", final_accuracy)
    return final_accuracy


if __name__ == "__main__":
    repeat = 3
    # initiate result map
    models = {"NB": 0, "SVM": 1, "logistic": 2, "baseline": 3}
    preprocessing = {
        "lemmatize_nostp": 0,
        "stem_nostp": 1,
        "nostp": 2,
        "lemmatize_stp": 3,
        "stem_stp": 4,
        "no_preprocess": 5,
    }
    steps = {
        "no_preprocess": ("lower"),
        "lemmatize_stp": ("lower", "lemmatize"),
        "nostp": ("lower", "stopwords"),
        "lemmatize_nostp": ("lower", "lemmatize", "stopwords"),
        "stem_nostp": ("lower", "stem", "stopwords"),
        "lemmatize_stp": ("lower", "lemmatize"),
        "stem_stp": ("lower", "stem"),
    }

    result = np.zeros((len(steps.keys()), len(models.keys())))
    # load data
    ddArray = getData("data/people.txt")
    corpus = getCorpus(ddArray, 0.3, 0)
    # preprocessing
    for preprocess_method in steps.keys():
        print(preprocess_method)
        corpus["final_text"] = [
            preprocess_entry(entry, steps=steps[preprocess_method])
            for entry in corpus["text"]
        ]
        vectorizer = getVectorizer(corpus)
        for model in models.keys():
            if model == "baseline":
                result[preprocessing[preprocess_method], models[model]] = (
                    sum(
                        [
                            baseline_predictor(
                                corpus, test_size=0.2, title=preprocess_method
                            )
                            for i in range(repeat)
                        ]
                    )
                    / repeat
                )
            else:
                result[preprocessing[preprocess_method], models[model]] = (
                    sum(
                        [
                            pipeline(
                                corpus,
                                vectorizer,
                                test_size=0.2,
                                classifier=model,
                                title=preprocess_method,
                            )
                            for i in range(repeat)
                        ]
                    )
                    / repeat
                )

    print(result)
