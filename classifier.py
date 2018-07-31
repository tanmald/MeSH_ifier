# -*- coding: utf-8 -*-

from nltk.corpus.reader import CategorizedPlaintextCorpusReader
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve, auc
from sklearn import metrics, preprocessing, svm
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, average_precision_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import string
import itertools
from plotting_learning_curves import plot_learning_curve


# PREPARE DATA FOR FUNCTION INPUT
def datainput(traincorpus, testcorpus):
    """
    Transforms the corpus to the correct pandas format.
    :param traincorpus, testcorpus: Corpus in CategorizedPlaintextCorpusReader format.
    :return: Two pandas datasets containing the text and label for each document of the corpus.
    """

    ## TRAIN CORPUS
    # X: a list or iterable of raw strings, each representing a document.
    X_train = [traincorpus.raw(fileid) for fileid in traincorpus.fileids()]
    # y: a list or iterable of labels, which will be label encoded.
    y_train = [traincorpus.categories(fileid)[0] for fileid in traincorpus.fileids()]

    # Transforming into Pandas DF
    columns = "Text Label".split()  # Declare the columns names
    # global traindata
    traindata = pd.DataFrame(list(zip(X_train, y_train)), columns=columns)  # load the dataset as a pandas data frame
    traindata = traindata.drop_duplicates(subset=['Text', 'Label'], keep=False)  # remove duplicate lines

    ## TEST CORPUS
    # X: a list or iterable of raw strings, each representing a document.
    X_test = [testcorpus.raw(fileid) for fileid in testcorpus.fileids()]
    # y: a list or iterable of labels, which will be label encoded.
    y_test = [testcorpus.categories(fileid)[0] for fileid in testcorpus.fileids()]

    # Transforming into Pandas DF
    columns = "Text Label".split()  # Declare the columns names
    testdata = pd.DataFrame(list(zip(X_test, y_test)), columns=columns)  # load the dataset as a pandas data frame
    testdata = testdata.drop_duplicates(subset=['Text', 'Label'], keep=False)  # remove duplicate lines

    return traindata, testdata


# DATA PREPROCESSING
def clean(doc):
    """
    Removes the punctuations, stopwords and normalizes the corpus.
    :param doc: A list of documents (strings)
    :return: A list of lists containing words from documents
    """
    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()

    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized


# CLASSIFY FUNCTION ANT TESTING
def classify(traindata, testdata, vectorizer=TfidfVectorizer(min_df=.1, max_df=0.9, max_features=None, preprocessor=clean,
             strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2),
             use_idf=True, smooth_idf=True, sublinear_tf=False), classifier=MultinomialNB(), learncurve=False, prcurve=False, roccurve=False):
    """
    Applies the classification algorithm.
    :param traindata: training corpus input, as provided by the "datainput" function.
    :param testdata: test corpus input, as provided by the "datainput" function.
    :param vectorizer: initializer for the vectorization algorithm.
    :param classifier: initializer for the classification algorithm.
    :param learncurve: boolean to compute learning curve plot.
    :param prcurve: boolean to compute precision/recall curve plot.
    :param roccurve: boolean to compute ROC curve plot.
    :return: classification report and confusion matrix.
    """

    # Split the data into training and test sets
    xtrainfull = traindata['Text']
    ytrainfull = traindata['Label']

    global xtest
    global ytest
    xtest = testdata['Text']
    ytest = testdata['Label']

    # Initialize the vectorizer.
    vect = vectorizer

    # 'vectorizer.fit_transform' creates the dictionary and vectorizes the input text at the same time.
    train_set = vect.fit_transform(xtrainfull)
    train_tags = ytrainfull

    # CROSS-VALIDATION
    # This cross validation prediction will print the predicted values of 'train_tags'
    predicted = cross_val_predict(clf, train_set, train_tags, cv=10)
    print('Validation set document classification accuracy = {}'.format(accuracy_score(train_tags, predicted) * 100))

    # The above metrics will show you how our estimator 'clf' works on 'train_set' data.
    # If the accuracies are very high it may be because of overfitting.

    # CLASSIFICATION
    # Now on using the full training data set and re-vectorize and retrain the classifier.
    # When vectorizing the test data, use '.transform()'
    # Note: the re-vectorization is necessary since the vectorizer is different using the full training set.
    test_set = vect.transform(xtest)

    # Get the tags with a better name just for clarification
    test_tags = ytest

    # Fit the entire training data to the estimator and then predict on test_tags
    clf.fit(train_set, train_tags)

    # Finally, use the classifier to predict on the test set.
    # To predict the tags, feed the vectorized 'test_set' to .predict()
    global predictions
    predictions = clf.predict(test_set)

    # Print the accuracy of the classifier
    print('Test set document classification accuracy = {}'.format(accuracy_score(predictions, test_tags) * 100))

    # For more detailed performance analysis of the results
    print(metrics.classification_report(test_tags, predictions, target_names=testcorpus.categories()))

    # Confusion Matrix
    print(confusion_matrix(test_tags, predictions))

    if learncurve == True:
        # LABEL ENCODING
        # Use the LabelEncoder from scikit-learn to convert text labels to integers, 0, 1 2
        lbl_enc = preprocessing.LabelEncoder()
        labels = lbl_enc.fit_transform(train_tags)

        # PLOT LEARNING CURVE
        X, y = train_set, labels

        title = "Learning Curves"
        # Cross validation with 100 iterations to get smoother mean test and train
        # score curves, each time with 20% data randomly selected as a validation set.
        cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

        estimator = clf
        plot_learning_curve(estimator, title, X, y, scoring='f1', ylim=None, cv=cv, n_jobs=4)
        plt.ylabel("Mean Cross Validation F1-Score")
        plt.xlabel("Number of Training Examples")
        plt.show()

    if prcurve == True:
        # LABEL ENCODING
        # Use the LabelEncoder from scikit-learn to convert text labels to integers, 0, 1 2
        lbl_enc = preprocessing.LabelEncoder()
        labels = lbl_enc.fit_transform(test_tags)

        y_score = clf.predict_proba(test_set)

        average_precision = average_precision_score(labels, y_score[:, 1])

        precision, recall, _ = precision_recall_curve(labels, y_score[:, 1])

        plt.step(recall, precision, color='b', alpha=0.2,
                 where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2,
                         color='b')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: Avg. P = {0:0.2f}'.format(
            average_precision))

    if roccurve == True:
        # LABEL ENCODING
        # Use the LabelEncoder from scikit-learn to convert text labels to integers, 0, 1 2
        lbl_enc = preprocessing.LabelEncoder()
        labels = lbl_enc.fit_transform(test_tags)

        y_score = clf.predict_proba(test_set)

        roc_score = roc_auc_score(labels, y_score[:, 1])

        average_precision = average_precision_score(labels, y_score[:, 1])

        precision, recall, _ = precision_recall_curve(labels, y_score[:, 1])

        plt.step(recall, precision, color='b', alpha=0.2,
                 where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2,
                         color='b')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: AUC={0:0.2f}'.format(
            average_precision))
        plt.savefig("{}_PRcurve.png".format(sys.argv[2]))

        fpr, tpr, _ = roc_curve(labels, y_score[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC Curve (AUROC = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic Example')
        plt.legend(loc="lower right")

        plt.savefig("{}_ROC curve.png".format(sys.argv[2]))


# CONFUSION MATRIX PLOT
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')


if __name__ == '__main__':
    print("\nStarting the Classifier. First, let's set everything up.")
    traincorpus_root = raw_input("Please specify the location of the training data: ")
    # traincorpus_root = '/Users/taniamaldonado/PycharmProjects/corpora/humanin/train4'
    traincorpus = CategorizedPlaintextCorpusReader(traincorpus_root, r".*_.*\.txt", cat_pattern=r'(\w+)_.*\.txt')

    testcorpus_root = raw_input("Please specify the location of the test data: ")
    # testcorpus_root = '/Users/taniamaldonado/PycharmProjects/corpora/humanin/test'
    testcorpus = CategorizedPlaintextCorpusReader(testcorpus_root, r".*_.*\.txt", cat_pattern=r'(\w+)_.*\.txt')

    try:
        traindata, testdata = datainput(traincorpus, testcorpus)
    except NameError:
        print "The training/test corpus is not defined, please check if the location is correct."

    print("\nPlease choose a classification algorithm:")
    print("1. Multinomial Naive Bayes")
    print("2. K Neighbors")
    print("3. Random Forest")
    print("4. Decision Trees")
    print("5. Logistic Regression \n")
    choice = raw_input("Enter a number to select the algorithm: ")

    if choice == "1":
        clf = MultinomialNB()
    elif choice == "2":
        clf = KNeighborsClassifier(3)
    elif choice == "3":
        clf = RandomForestClassifier()
    elif choice == "4":
        clf = DecisionTreeClassifier()
    elif choice == "5":
        clf = LogisticRegression(C=1.0, penalty='l1')

    print("\nDo you wish to see any of the above:")
    learnoption = raw_input("Learning curve plot: (Y/N) ")

    if learnoption == "Y" or learnoption == "y":
        learnchoice = True
    elif learnoption == "N" or learnoption == "n":
        learnchoice = False

    proption = raw_input("Precision/recall plot: (Y/N) ")

    if proption == "Y" or proption == "y":
        prchoice = True
    elif proption == "N" or proption == "n":
        prchoice = False

    rocoption = raw_input("ROC curve plot: (Y/N) ")

    if rocoption == "Y" or rocoption == "y":
        rocchoice = True
    elif rocoption == "N" or rocoption == "n":
        rocchoice = False

    confmatplot = raw_input("Confusion matrix plot: (Y/N) ")

    print("\nStarting the classifier...\n")
    classify(traindata, testdata, classifier=clf, learncurve=learnchoice, prcurve=prchoice, roccurve=rocchoice)

    if confmatplot == "Y" or confmatplot == "y":
        # Plot
        cnf_matrix = confusion_matrix(ytest, predictions)
        class_names = testcorpus.categories()
        plot_confusion_matrix(cnf_matrix, classes=class_names)
        plt.show()

    print("\nFinished!")