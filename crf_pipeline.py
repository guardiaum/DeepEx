import sys
import csv, io
import pickle
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from util.Constants import *
from util.common import *
from modules.SentenceClassifier import BootstrappingClassifier
from modules.Extractor import Extractor
from modules.WikipediaNavigator import WikipediaNavigator
from joblib import Parallel, delayed
import multiprocessing


def run(CLASS):
    '''
    # select class properties
    class_path = TRAIN_CLF_DIR + "/" + CLASS
    properties = [f.replace(".csv", "") for f in os.listdir(class_path) if
                  os.path.isfile(os.path.join(class_path, f))]

    print(properties)

    bt_clfs = []
    print("\nSTART GENERATING EXTRACTION TRAIN DATASETS")
    for property_name in properties:
        print("\nSTART GENRATING EXTRACTION TRAIN DATASET FOR %s" % property_name)

        bt_clf = BootstrappingClassifier()

        bt_clf.run(property_name, CLASS, ITERATION_NUMBER, STEP_THRESHOLD=0.15)

        bt_clfs.append([property_name, bt_clf])'''

    # select class properties
    class_path = TRAIN_EXT_DIR + "/" + CLASS
    properties = [f.replace(".csv", "") for f in os.listdir(class_path) if
                      os.path.isfile(os.path.join(class_path, f))]
    print(properties)

    extractor = Extractor()
    CRFs = extractor.trainCRFModel(properties, CLASS)

    articles_name = get_test_articles_title(CLASS)

    sent_clfs = get_sent_classifiers(CLASS)

    extraction_by_articles = run_pipeline(extractor, CLASS, CRFs, articles_name, sent_clfs)

    if extraction_by_articles is not None:
        outputfile = EXPERIMENTS_DIR + "/" + CLASS + '-deepex_CRF.csv'

        if os.path.exists(outputfile):
            os.remove(outputfile)

        keys = extraction_by_articles[0].keys()
        with open(outputfile, 'w') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(extraction_by_articles)


def run_pipeline(extractor, CLASS,  CRFs, articles_name, sentenceClassifiers):
    if len(articles_name) > 0:
        extractionByArticles = []

        for article in articles_name:
            print("-----------------------------------------------")
            print("ARTICLE: %s" % article.replace("|", "/"))

            extractionByArticles.append({'article': article})

            text = readFullArticleText(CLASS, article)

            if text is not None or len(text) > 0:
                sentences = np.array(sent_tokenize(text))

                sentences = np.array([sent for sent in sentences if len(sent) > 1])

                print("%s sentences" % sentences.shape[0])

                if sentences.shape[0] > 0:

                    for attribute, classifier in sentenceClassifiers:

                        vectorizer_path = MODELS + "/" + CLASS + "/clf-features/" + attribute + ".pkl"

                        transformer = TfidfTransformer()
                        loaded_vec = CountVectorizer(decode_error="replace",
                                                     vocabulary=pickle.load(open(vectorizer_path, "rb")))

                        if loaded_vec is not None:
                            sentences_features = transformer.fit_transform(loaded_vec.fit_transform(sentences))

                            predictedLabels = classifier.predict_proba(sentences_features)

                            predictedLabels = ['f' if pred[0] > pred[1] else 't' for pred in predictedLabels]

                            extractedValues = []
                            for i, predictedLabel in enumerate(predictedLabels):
                                predictedLabel = predictedLabel

                                if predictedLabel is 't':
                                    # run crf extractor
                                    crf = None
                                    for a, model in CRFs:
                                        if a == attribute:
                                            print("attribute: ", a)
                                            crf = model

                                    extractedDicts = extractor.runCRFExtractor(crf, sentences[i])

                                    if len(list(filter(None, extractedDicts))) > 0:
                                        extractedValues.append(extractedDicts)

                            tokens = selectTokensFromExtractedOutputs(extractedValues)

                            for d in extractionByArticles:
                                if d.get('article') == article:
                                    d.update({attribute: tokens.strip()})
                        else:
                            print("CANT FIND CLASSIFIER FOR %s" % attribute)
                else:
                    for attribute, classifier in sentenceClassifiers:
                        for d in extractionByArticles:
                            if d.get('article') == article:
                                d.update({attribute: 'NA'})
                    print("%s ~~~~~~~~~~~~~~~~~~~~ HAS NO SENTENCES" % article)
            else:
                for attribute, classifier in sentenceClassifiers:
                    for d in extractionByArticles:
                        if d.get('article') == article:
                            d.update({attribute: 'NA'})
                print("%s ~~~~~~~~~~~~~~~~~~~~ HAS NO TEXT" % article)

        return extractionByArticles
    else:
        return None


if __name__ == "__main__":
    if len(sys.argv) == 2:
        CLASS = sys.argv[1]
        run(CLASS)
    else:
        print("PLEASE INFORM CLASS")