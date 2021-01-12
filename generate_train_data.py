import sys
import codecs
import copy
import csv
from util.Constants import *
from util import threshold_similarity as threshold_sim
from nltk.tokenize import sent_tokenize, word_tokenize
from py_stringmatching.similarity_measure.soft_tfidf import SoftTfIdf
from py_stringmatching.similarity_measure.overlap_coefficient import OverlapCoefficient
from py_stringmatching.similarity_measure.jaro_winkler import JaroWinkler

'''
    soft_threshold: threshold similarity for the 
    soft tf-idf string matching inside the window of tokens
'''
def createDatasets(infobox_class, properties, soft_threshold):

    articles_text_dir = FULL_BASE_DIR + "/" + infobox_class + "/articles/"
    infoboxes_dir = FULL_BASE_DIR + "/" + infobox_class + "/infoboxes/"

    infoboxes_files = [f for f in os.listdir(infoboxes_dir) if
                  os.path.isfile(os.path.join(infoboxes_dir, f))]

    output_dir = TRAIN_CLF_DIR + "/" + infobox_class + "/"
    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir)
    else:
        os.rmdir(output_dir)
        os.makedirs(output_dir)

    # Ignores validation and test articles used for validate the model training and to fit parameters, respectivelly
    validation_articles = VALIDATION_ARTICLES_DIR + "/" + infobox_class + "-list"

    val_f = open(validation_articles, "r")
    articles_val = val_f.read().splitlines()
    print("validation articles", len(articles_val))

    test_articles = TEST_ARTICLES_DIR + "/" + infobox_class + "-list"
    test_f = open(test_articles, "r")
    articles_test = test_f.read().splitlines()
    print("test articles", len(articles_test))

    for f in infoboxes_files:

        if f in articles_val or f in articles_test:
            print("skipping validation/test article", f)
            continue

        infobox_tuples_f = codecs.open(infoboxes_dir+f, "r", encoding="utf8")
        article_text_f = codecs.open(articles_text_dir+f, 'r', encoding="utf8")
        article_text = article_text_f.read().replace("\n",' ')

        infoboxTuples = []
        for line in infobox_tuples_f:
            tuple = line.replace("\n", "").split("\t:\t")
            if len(tuple) == 2:
                infoboxTuples.append(tuple)

        # instantiate soft tf-idf and overlap coefficient [not used]
        soft_tfidf, oc = instantiateStringMatching(article_text, infoboxTuples)

        for tuple in infoboxTuples:

            if tuple[0] in properties:
                propName = tuple[0]
                outputFilePath = output_dir + propName + ".csv"

                data = do_sentence_soft_matching(soft_tfidf, soft_threshold, f, sent_tokenize(article_text), tuple)
                # data = doSentenceMatching(infobox_class, oc, soft_tfidf, f, sent_tokenize(article_text), tuple)

                with open(outputFilePath, 'a+') as csvfile:
                    spamwriter = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC)
                    spamwriter.writerows(data)
                    csvfile.close()


'''
    DO SENTENCE MATCHING WITH THRESHOLD LIMIT PASSED THROUGH PARAMETER
'''
def do_sentence_soft_matching(soft_tfidf, soft_threshold, entity, sentences, tuple):
    entityName = entity.replace("|", "/").replace("_", " ")

    data = []
    # try to match infobox tuple to sentences in text
    prop = tuple[0].replace("_", " ")
    value = tuple[1]

    for sentence in sentences:
        sentence_tokens = word_tokenize(sentence)

        ner = [[token, 'O'] for token in sentence_tokens]
        if len(sentence_tokens) > 3:
            soft_raw_score, ner = threshold_sim.get_similarity_and_set_ner(sentence_tokens, prop, value, soft_tfidf)

            # soft_raw_score = threshold_sim.get_max_similarity_on_window(sentence_tokens, prop, value, soft_tfidf)
            # soft_raw_score = soft_tfidf.get_raw_score(sentence_tokens, attributeTokens)

            if len(ner) == len(word_tokenize(sentence)):
                if soft_raw_score >= soft_threshold:
                    data.append([sentence, ner, entityName, value, 't'])
                elif 0 < soft_raw_score:
                    data.append([sentence, ner, entityName, value, 'o'])
                else:
                    data.append([sentence, ner, entityName, value, 'f'])
        else:
            data.append([sentence, ner, entityName, value, 'f'])

    return data

''' 
    DO SENTENCE MATCHING WITH INFOBOX TUPLES
    THRESHOLD SIMILARITY BASED ON SENTENCES FROM VALIDATION SET 
    SAVE DATASET TO FILE 
'''
def doSentenceMatching(infobox_class, oc, soft_tfidf, entityName, sentences, infoboxTuple):
    threshold_df = threshold_sim.run(infobox_class)
    print(threshold_df)

    entityName = entityName.replace("|", "/").replace("_"," ")

    data = []
    # try to match infobox tuple to sentences in text
    prop = infoboxTuple[0]
    value = infoboxTuple[1]

    soft_mean_threshold = threshold_df.loc[prop, ('soft_tf-idf', 'mean_soft')]
    soft_min_threshold = threshold_df.loc[prop, ('soft_tf-idf', 'min_soft')]
    soft_max_threshold = threshold_df.loc[prop, ('soft_tf-idf', 'max_soft')]
    oc_mean_threshold = threshold_df.loc[prop, ('overlap_coef', 'mean_oc')]
    oc_min_threshold = threshold_df.loc[prop, ('overlap_coef', 'min_oc')]
    oc_max_threshold = threshold_df.loc[prop, ('overlap_coef', 'max_oc')]

    attributeTokens = word_tokenize(prop.replace("_", " ") + " " + value)

    oc_threshold = oc_min_threshold
    if oc_threshold == 0:
        oc_threshold = oc_mean_threshold
    soft_threshold = soft_min_threshold
    if soft_threshold == 0:
        soft_threshold = soft_mean_threshold

    for sentence in sentences:
        sentence_tokens = word_tokenize(sentence)

        if len(sentence_tokens) > 3:
            soft_raw_score = soft_tfidf.get_raw_score(sentence_tokens, attributeTokens)
            oc_raw_score = oc.get_raw_score(sentence_tokens, attributeTokens)

            if oc_raw_score >= oc_threshold and soft_raw_score >= soft_threshold:
                data.append([sentence, entityName, value, 't'])
            elif 0 < oc_raw_score or 0 < soft_raw_score:
                data.append([sentence, entityName, value, 'o'])
            else:
                data.append([sentence, entityName, value, 'f'])
        else:
            data.append([sentence, entityName, value, 'f'])

    return data


def instantiateStringMatching(cleanedText, infoboxTuples):

    tokenizedTuples = [word_tokenize(prop.replace("_", " ") + " " + value) for prop, value in infoboxTuples]
    corpus = copy.deepcopy(tokenizedTuples)

    # tokenize text into sentences
    sentences = sent_tokenize(cleanedText.replace("\n", " ").replace("\t", " ").strip().rstrip())

    for sentence in sentences:
        tokenizedSent = word_tokenize(sentence)
        corpus.append(tokenizedSent)

    soft_tfidf = SoftTfIdf(corpus, sim_func=JaroWinkler().get_raw_score, threshold=0.8)
    oc = OverlapCoefficient()

    return soft_tfidf, oc

'''
    AUTOMATICALLY BUILDS TRAINING DATASETS FOR DEEPEX
    ACCORDING TO THE SOFT TF-IDF SIMILARITY AND WINDOW OF TOKENS
    
    IT RECIEVES AS INPUT THE NAME OF THE INFOBOX TEMPLATE (CLASS) AND 
    THE PROPERTIES FILES CONTAINING ALL PROPERTIES BELONGING TO THE DEFINED SCHEMA
    
    Default properties file: datasets/generate-datasets
'''
if __name__ == "__main__":
    if len(sys.argv) == 3:

        # read input
        infobox_class = sys.argv[1]
        properties_file = sys.argv[2]

        # selects all properties in the defined schema
        f = open(ROOT + "/" + properties_file, 'r')
        properties = [line.replace("\n", "") for line in f.readlines()]
        print(properties)

        print("\nSTART CREATING DATASETS FOR DEEPEX")

        createDatasets(infobox_class, properties, 0.5)

        print("\nENDED CREATING DATASETS FOR DEEPEX")
    else:
        print("PLEASE INFORM CLASS")