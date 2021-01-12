import time
import re
import os
import sklearn_crfsuite
import pandas as pd
from util.Constants import *
from util.DatasetsUtil import *
from util.common import *
from util.util import *
from sklearn_crfsuite.metrics import flat_accuracy_score
from collections import Counter
from sklearn_crfsuite import metrics
from random import randrange
from copy import deepcopy
from nltk import word_tokenize
from nltk import pos_tag
import pickle
import nltk
nltk.download('maxent_ne_chunker')
nltk.download('words')


class Extractor(object):

    def runCRFExtractor(self, crf, sentence):

        if crf is not None:
            features_crf = Extractor().sent2features(sentence)
            predictedLabels = crf.predict_single(features_crf)
            predictedMarginals = crf.predict_marginals_single(features_crf)

            return self.extractFromCRFOutput(sentence, predictedLabels, predictedMarginals)
        else:
            print("CRF IS NONE")
            return []

    def extractFromCRFOutput(self, sentence, labels, marginals):
        tokens = word_tokenize(sentence)

        # composedValue indicates if attribute is composed by two or more tokens
        extractions = heuristicTokensSelection(labels, marginals, tokens)

        if extractions is not None and len(extractions) > 0:
            print(sentence)
            print(extractions)
            return extractions

        return []


    def trainCRFModel(self, attributes, CLASS):
        start_time = time.time()
        CRFs = []

        for attribute in attributes:
            print("Start training CRF for attribute: %s" % attribute)

            dump_dir = MODELS + "/" + CLASS + "/crf/"
            dump_file = dump_dir + attribute + ".sav"

            if os.path.exists(dump_dir) is False:
                os.makedirs(dump_dir)

            if os.path.isfile(dump_file):
                print("LOADING MODEL ALREADY TRAINED AND SAVED")
                crf = pickle.load(open(dump_file, 'rb'))
                CRFs.append([attribute, crf])
            else:
                train_file = TRAIN_EXT_DIR + "/" + CLASS + "/" + attribute + ".csv"

                samples = pd.read_csv(train_file, names=['sentence', 'ner', 'entity', 'value', 'label'],
                                      dtype={'sentence': str, 'entity': str, 'value': str, 'label': str},
                                      converters={'ner': eval}, encoding='utf-8')

                data = samples[samples['label'] == 't']
                subset = data[['ner', 'sentence']]

                sentences = subset['sentence'].values
                ner = subset['ner'].values

                sentences_final = []
                ner_final = []
                for i, s in enumerate(sentences):
                    s_tokens = word_tokenize(s)
                    if len(s_tokens) == len(ner[i]):
                        sentences_final.append(s)
                        ner_final.append(ner[i])
                    else:
                        print(">>> DIFFERENT LENGHTS FOR SENTENCE TOKENS AND NER")

                features = [self.sent2features(sent) for sent in sentences_final]
                labels = [n for n in ner_final]

                print('\nCount features: {}'.format(len(features)))
                print('Count labels: {}'.format(len(labels)))

                '''
                print(features[0])
                print([example for index, example in samples.iterrows() if example['label'] == 't'][0])
                print(labels[0])
                print(type(features))
                print(type(labels))
                print("\n")
                
                Removes occurences of ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] from training CRFs
                
                keptFeatures = []
                keptLabels = []
                
                for index, crf_labels in enumerate(labels):
                    keep = False
                    for ner in crf_labels:
                        if ner == 'VALUE':
                            keep = True

                    if keep is True:
                        print(crf_labels)
                        keptFeatures.append(features[index])
                        keptLabels.append(labels[index])

                print("\nAFTER CLEANING NEGATIVE OCCURENCES")
                print('Count kept features: {}'.format(len(keptFeatures)))
                print('Count kept labels: {}'.format(len(keptLabels)))'''

                X_train = deepcopy(features)
                y_train = deepcopy(labels)

                val_df = get_val_data3(CLASS, attribute)
                val_subset = val_df[['ner', 'sentence']]
                val_sentences = val_subset['sentence'].values
                val_ner = val_subset['ner'].values

                test_sentences = []
                test_labels = []
                for i, s in enumerate(val_sentences):
                    s_tokens = word_tokenize(s)
                    if len(s_tokens) == len(val_ner[i]):
                        test_sentences.append(s)
                        test_labels.append(val_ner[i])
                    else:
                        print(">>> DIFFERENT LENGHTS FOR VALIDATION SENTENCE TOKENS AND NER")

                X_test = [self.sent2features(sent) for sent in test_sentences]
                y_test = [n for n in test_labels]

                print('\nCount features: {}'.format(len(X_test)))
                print('Count labels: {}'.format(len(y_test)))
                #X_train, y_train, X_test, y_test = self.splitTrainingData(X, y, 0.75)

                crf = sklearn_crfsuite.CRF(
                    algorithm='lbfgs',
                    c1=0.1,
                    c2=0.1,
                    max_iterations=30,
                    all_possible_transitions=True,
                )
                crf.fit(X_train, y_train)
                pickle.dump(crf, open(dump_file, 'wb'))

                labels = list(crf.classes_)
                print("CRF CLASSES: ")
                print(labels)

                y_pred = crf.predict(X_test)

                sorted_labels = sorted(
                    labels,
                    key=lambda name: (name[1:], name[0])
                )
                print(metrics.flat_classification_report(
                    y_test, y_pred, labels=sorted_labels, digits=3
                ))

                print(">>> flat accuracy: %.3f" % flat_accuracy_score(y_test, y_pred))

                print("Top likely transitions:")
                self.print_transitions(Counter(crf.transition_features_).most_common(20))

                print("\nTop positive:")
                self.print_state_features(Counter(crf.state_features_).most_common(30))

                '''print("\nTop negative:")
                self.print_state_features(Counter(crf.state_features_).most_common()[-30:])'''
                CRFs.append([attribute, crf])

        elapsed_time = time.time() - start_time
        print("Returned CRF Extractors: {}".format(len(CRFs)))
        print("CRF extractors: {}".format([name for (name, crf) in CRFs]))
        print("Elapsed time: {}".format(elapsed_time))
        return CRFs

    def print_state_features(self, state_features):
        for (attr, label), weight in state_features:
            print("%0.6f %-8s %s" % (weight, label, attr))

    def print_transitions(self, trans_features):
        for (label_from, label_to), weight in trans_features:
            print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

    def splitTrainingData(self, features, labels, division_ratio):
        trainSize = int(len(features) * division_ratio)
        x_train = []
        y_train = []
        x_test = []
        y_test = []

        while len(x_train) < trainSize and len(y_train) < trainSize:
            index = randrange(len(features))
            x_train.append(features.pop(index))
            y_train.append(labels.pop(index))

        while len(features) > 0:
            x_test.append(features.pop())
            y_test.append(labels.pop())

        return x_train, y_train, x_test, y_test

    def label2features(self, label, sent):
        tokens = word_tokenize(sent)
        postags = pos_tag(tokens)
        label = label.encode("utf-8")
        label_tk = word_tokenize(label.lower())
        return ['VALUE' if tags[0].encode("utf-8").lower() in label_tk else 'O' for tags in postags]

    def sent2features(self, sent):
        tokens = word_tokenize(sent)
        tags = pos_tag(tokens)

        return [self.word2features(tags, i) for i in range(len(tags))]

    def word2features(self, sent, i):
        word = sent[i][0]
        postag = sent[i][1]

        # features object definition with Token and postag added. Others default.
        features = {
            'token': word,
            'postag': postag,
            'np_chunk': 'none',
            'start_capital': False,
            'single_capital': False,
            'capital_period': False,
            'all_capital_period': False,
            'contains_number': False,
            'two_digits': False,
            'four_digits': False,
            'dollar_sign': False,
            'underline': False,
            'percentage': False,
            'purely_numeric': False,
            'number_type': False,
            'stop_word': False
        }

        # NP Chunk tag
        grammar = r"""NP:
        {<.*>+}          # Chunk everything
        }<VBD|IN>+{      # Chink sequences of VBD and IN
        """
        cp = nltk.RegexpParser(grammar)
        tree = cp.parse(sent)

        features['np_chunk'] = [subtree.label()
                                if sent[i][0] in [token for (token, tag) in subtree.leaves()] else 'none'
                                for subtree in tree.subtrees()][0]

        '''
            print(features['np_chunk'])
            for subtree in tree.subtrees():
                if sent[i][0] in [token for (token, tag) in subtree.leaves()]:
                    features['np_chunk'] = subtree.label()
                else:
                    features['np_chunk'] = 'none'
        '''

        # First token of sentence
        if i == 0:
            features['first'] = word

        # In first and second half of sentence
        if i < len(sent)/2:
            features['first_half'] = word
        elif i >= len(sent)/2:
            features['second_half'] = word

        # String normalization
        normalization = self.wordNormalization(word)
        features['normalization'] = normalization

        # Previous Tokens (window size = 5)
        previous_tokens = self.getTokensInWindow(sent, i, 5, "prev")

        features['previous_tokens'] = previous_tokens

        # Next Tokens (window size = 5)
        next_tokens = self.getTokensInWindow(sent, i, 5, "next")

        features['next_tokens'] = next_tokens

        # First letter capitalized
        if word[0].isupper():
            features['start_capital'] = True

        # Single capital
        if len(word)==1 and word.isupper():
            features['single_capital'] = True

        # Starts capital end period
        if word[0].isupper() and word[len(word)-1] == '.':
            features['capital_period'] = True

        # All capital end period
        capital_period_pattern = re.compile('^[A-Z]*\.$')
        if capital_period_pattern.match(word) is not None:
            features["all_capital_period"] = True

        # Contains at least one digit
        one_number_pattern = re.compile('[0-9]+')
        if one_number_pattern.match(word) is not None:
            features['contains_number'] = True

        # Two digits
        two_digits_pattern = re.compile('^[0-9]{2}$')
        if two_digits_pattern.match(word) is not None:
            features['two_digits'] = True

        # Four digits
        four_digits_pattern = re.compile('^[0-9]{4}$')
        if four_digits_pattern.match(word) is not None:
            features['four_digits'] = True

        # Contains dollar sign
        dollar_sign_pattern = re.compile('\$')
        if dollar_sign_pattern.match(word) is not None:
            features['dollar_sign'] = True

        # Contains uniderline
        underline_pattern = re.compile('\_')
        if underline_pattern.match(word) is not None:
            features['underline'] = True

        # Contains percentage
        percentage_pattern = re.compile('\%')
        if percentage_pattern.match(word) is not None:
            features['percentage'] = True

        # Purely numeric
        purely_numeric_pattern = re.compile('^\d+$')
        if purely_numeric_pattern.match(word) is not None:
            features['purely_numeric'] = True

        # Number type
        number_type_pattern = re.compile('(\d+((\.|,)*\d+)+((,)*\d+)*)*')
        if number_type_pattern.match(word) is not None:
            features['number_type'] = True

        # Stop word
        stop_words = ['the', 'a', 'of']
        if word in stop_words:
            features['stop_word'] = True

        # print(features)

        return features

    # capital to "A"
    # lowercase to "a"
    # digit to "1"
    # others to "0"
    def wordNormalization(self, word):
        normalization = ''
        digit_pattern = re.compile('\d')

        for character in word:
            if character.isupper():
                normalization += "A"
            elif character.islower():
                normalization += "a"
            elif digit_pattern.match(character):
                normalization += "1"
            else:
                normalization += "0"
        return normalization

    # return previous tokens in sentence including current index
    def getTokensInWindow(self, sent, current_index, window_size, type):

        returnedTokens = []
        for i in range(1, window_size):
            if len(returnedTokens) < window_size:
                index = 0
                if type == 'prev':
                    index = current_index - i
                elif type == 'next':
                    index = current_index + i

                if 0 <= index < len(sent):
                    returnedTokens.append(sent[index][0])
        return returnedTokens
