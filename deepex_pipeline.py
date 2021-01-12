import sys
import importlib
import csv
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from util.Constants import *
from util.common import *
from modules.BLSTM import *
from util.util import *
from keras.optimizers import SGD, Nadam

def get_models(class_, type_):
    sent_clfs = get_sent_classifiers(class_)
    print('sent_clfs: {}'.format(len(sent_clfs)))

    attr_extractors = get_attr_extractors(class_, type_)
    print('attr_extractors: {}'.format(len(attr_extractors)))

    return sent_clfs, attr_extractors

def run_pipeline(class_, type_, extractors, articles_name, sentenceClassifiers):
    if len(articles_name) > 0:
        extractionByArticles = []

        for article in articles_name:
            print("-----------------------------------------------")
            print("ARTICLE: %s" % article.replace("|", "/"))

            extractionByArticles.append({'article': article})

            text = readFullArticleText(class_, article)

            if text is not None or len(text) > 0:
                sentences = np.array(sent_tokenize(text))

                sentences = np.array([sent for sent in sentences if len(sent) > 1])

                print("%s sentences" % sentences.shape[0])

                if sentences.shape[0] > 0:
                    df_sentences = pd.DataFrame(data={'sentence': sentences})

                    for attribute, classifier in sentenceClassifiers:
                        vectorizer_path = MODELS + "/" + class_ + "/clf-features/" + attribute + ".pkl"

                        transformer = TfidfTransformer()
                        loaded_vec = CountVectorizer(vocabulary=pickle.load(open(vectorizer_path, "rb")))

                        if loaded_vec is not None:
                            transformed = loaded_vec.transform(df_sentences.sentence.values)
                            sentences_features = transformer.fit_transform(transformed)

                            predictedLabels = classifier.predict_proba(sentences_features)

                            predictedLabels = ['f' if pred[0] > pred[1] else 't' for pred in predictedLabels]

                            extractedValues = []
                            for i, predictedLabel in enumerate(predictedLabels):

                                if predictedLabel is 't':

                                    # run crf extractor
                                    extractor = None
                                    for a, cnn_blstm in extractors:
                                        if attribute in a:
                                            extractor = cnn_blstm

                                    if extractor is not None:
                                        extractedDicts = predict_sentence(extractor, sentences[i], class_, attribute, type_)

                                        if len(list(filter(None, extractedDicts))) > 0:
                                            extractedValues.append(extractedDicts)
                                    else:
                                        print("NONE EXTRACTOR FOUND")

                            tokens = selectTokensFromExtractedOutputs(extractedValues)

                            for d in extractionByArticles:
                                if d.get('article') == article:
                                    d.update({attribute: tokens.strip()})
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
    if len(sys.argv) == 3:
        class_ = sys.argv[1]
        type_ = sys.argv[2]  # CNN_BLSTM / BLSTM / BLSTM_W2

        sent_clfs, attr_extractors = get_models(class_, type_)

        articles_name = get_test_articles_title(class_)

        #articles_name = articles_name[:3]

        extraction_by_articles = run_pipeline(class_, type_, attr_extractors, articles_name, sent_clfs)

        if extraction_by_articles is not None:
            outputfile = EXPERIMENTS_DIR + "/" + class_ + '-deepex_' + type_ + '.csv'

            if os.path.exists(outputfile):
                os.remove(outputfile)

            keys = extraction_by_articles[0].keys()
            with open(outputfile, 'w') as output_file:
                dict_writer = csv.DictWriter(output_file, keys)
                dict_writer.writeheader()
                dict_writer.writerows(extraction_by_articles)
    else:
        print("PLEASE INFORM CLASS")