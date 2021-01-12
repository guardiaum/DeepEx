from util.Constants import *
from keras.models import load_model
import pickle


def get_test_articles_title(CLASS):
    class_path = TEST_ARTICLES_DIR + "/" + CLASS
    articles_name = [f for f in os.listdir(class_path) if
                     os.path.isfile(os.path.join(class_path, f))]
    return articles_name


def readFullArticleText(CLASS, article):
    file = open(TEST_ARTICLES_DIR + "/" + CLASS + "/" + article, "r")
    text = " ".join(line for line in file.readlines())
    return text


def get_sent_classifiers(CLASS):
    dump_dir = MODELS + "/" + CLASS + "/clf/" + str(ITERATION_NUMBER) + "/"

    sent_clfs = [f.replace(".sav", "") for f in os.listdir(dump_dir)]

    clfs = []
    for attribute in sent_clfs:
        dump_file = dump_dir + attribute + ".sav"
        print(">>> loading model for %s" % attribute)
        clf = pickle.load(open(dump_file, 'rb'))
        clfs.append([attribute, clf])

    return clfs

def get_attr_extractors(class_, type):

    dump_dir = MODELS + "/" + class_ + "/dl/"

    attr_extractors = [f.replace(".h5", "") for f in os.listdir(dump_dir)]

    extractors = []

    type_ = "-" + type + "_best_model"

    for extractor in attr_extractors:
        if type_ in extractor:
            print(">>> loading extractor model: %s" % dump_dir+extractor)
            ext = load_model(dump_dir + extractor + '.h5')
            extractors.append([extractor.replace(type_,""), ext])

    return extractors


# IF PREVIOUS OR NEXT TOKEN IS VALUE, RETURN ALL TOKENS
# ELSE IF VALUE LABELS ARE ISOLATED, RETURN THE ONE WITH HIGHER CONFIDENCE SCORE
def heuristicTokensSelection(labels, marginals, tokens):

    valuesList = []
    temp = []

    for i, currentLabel in enumerate(labels):
        if currentLabel == 'VALUE':  # add token to list if label is 'VALUE'
            temp.append(tokenSelection(i, marginals, tokens))
        elif currentLabel == 'O' and len(list(filter(None, temp))) == 1:
            # add found value to final  list and clears temp list when find a label 'O'
            valuesList.append([temp[0]])
            temp = []
        elif currentLabel == 'O' and len(list(filter(None, temp))) > 1:
            # add found values to final list and clears temp list of tokens
            extraction = []
            for t in temp:
                extraction.append(t)
            # add to list as a list of tokens (dicts) and clears temp list oof tokens
            valuesList.append(extraction)
            temp = []

    if len(valuesList) > 0:
        # iterates over a list of list of dicts and return a list of dicts with bigger confidence
        biggerConfidence = getHigherConfidenceFromList(valuesList)

        return biggerConfidence

    return None


def tokenSelection(i, marginals, tokens):
    confidence = marginals[i].get('VALUE')
    extraction = {
        'token': tokens[i],
        'confidence': confidence
    }
    return extraction


def selectTokensFromExtractedOutputs(extractedValues):
    tokens = ''
    if len(list(filter(None, extractedValues))) == 0:
        tokens = 'NA'
    elif len(list(filter(None, extractedValues))) == 1:
        for extractedDict in extractedValues[0]:
            tokens += extractedDict.get('token') + ' '
    elif len(list(filter(None, extractedValues))) > 1:
        extractedValue = getHigherConfidenceFromList(extractedValues)
        for extractedDict in extractedValue:
            tokens += extractedDict.get('token') + ' '
    return tokens


def getHigherConfidenceFromList(extractionsList):
    biggerConfidence = [{'token': 'NA', 'confidence': 0.0}]

    for extractions in extractionsList:

        if len(extractions) == 1:
            if len(extractions[0].get('token')) <= 2:
                continue

        localConfidence = 0.0
        for extraction in extractions:
            localConfidence += extraction.get('confidence')

        keptConfidence = 0.0
        for kept in biggerConfidence:
            keptConfidence += kept.get('confidence')

        if localConfidence > keptConfidence:
            biggerConfidence = extractions

    return biggerConfidence