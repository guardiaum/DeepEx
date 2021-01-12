import numpy as np
import pandas as pd
import tensorflow as tf
from nltk import word_tokenize
from nltk import pos_tag
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import StratifiedShuffleSplit
from keras.utils import Progbar


'''
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
'''
def read_file(property_name, prop_path):
    data = pd.read_csv(prop_path, names=['sentence', 'ner', 'entity', 'value', 'label'],
                       dtype={'sentence': str, 'entity': str, 'value': str, 'label': str},
                       converters={'ner': eval}, encoding='utf-8')
    data['property'] = property_name
    data = data[data['label'] == 't']
    subset = data[['ner', 'sentence']]

    sentences = subset['sentence'].values
    ner = subset['ner'].values

    tagged_data = []
    for i, s in enumerate(sentences):
        s_tokens = word_tokenize(s)
        if len(s_tokens) == len(ner[i]):
            tags = [''] * len(ner[i])
            for j, token in enumerate(s_tokens):
                tags[j] = [token, ner[i][j]]
            tagged_data.append(tags)
        else:
            print(">>> DIFFERENT LENGHTS FOR SENTENCE TOKENS AND NER")
    return tagged_data
    #return split_data_get_features(subset)


'''
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
'''
def read_validation_file(file, class_):
    data = pd.read_csv(file, dtype=str)
    data = data[data['property'] == class_]
    subset = data[['property', 'value', 'sentence']]

    data = subset[['sentence']].values
    values = subset[['value']].values
    labels = subset[['property']].values

    return [sentence2features(labels[index][0], values[index][0], row[0]) for index, row in enumerate(data)]


def get_data_from_class(data, class_):
    return data[data['property'] == class_]


def sentence2features(prop, value, sent):
    try:
        prop_tk = word_tokenize(prop.replace("_", " ").lower())
        value_tk = word_tokenize(value.lower())
        tokens = word_tokenize(sent)
        postags = pos_tag(tokens)
        return [[token, 'PROP' if token.lower() in prop_tk else 'VALUE' if token.lower() in value_tk else 'O'] for token, tag in postags]
    except TypeError:
        print("error trying tokenize sentence")


def split_data_get_features(subset):

    data = subset['sentence'].values
    ner = subset['ner'].values

    tagged_data = []
    for i, d in enumerate(data):
        d_tokens = word_tokenize(d)

        if len(d_tokens) == len(ner[i]):
            tags = [''] * len(ner[i])
            for j, token in enumerate(d_tokens):
                tags[j] = [token, ner[i][j]]
            tagged_data.append(tags)
        else:
            print(">>> DIFFERENT LENGHTS FOR SENTENCE TOKENS AND NER")

    train_size = int(len(tagged_data) - (len(tagged_data) * 0.25))

    return tagged_data[0:train_size], tagged_data[train_size:len(tagged_data)]


''' 
Gets characters information and adds to sentences
Returns a matrix where the row is the sentence and 
each column is composed by token setence, characters information from tokens and label for token
'''
def add_char_information_in(sentences):
    for i, sentence in enumerate(sentences):
        for j, data in enumerate(sentence):
            chars = [c for c in data[0]]  # data[0] is the token
            sentences[i][j] = [data[0], chars, data[1]]  # data[1] is the annotation (label)

    return sentences


def create_dictionaries(sentences):
    label_set = set()
    dictionary = {}
    for sentence in sentences:
        for token, char, label in sentence:
            label_set.add(label)
            dictionary[token.lower()] = True
    return label_set, dictionary


def labels_mappings(label_set):
    label2Idx = {}
    for label in label_set:
        label2Idx[label] = len(label2Idx)
    return label2Idx


def get_casing(word, caseLookup):
    casing = 'other'

    numDigits = 0
    for char in word:
        if char.isdigit():
            numDigits += 1

    digitFraction = numDigits / float(len(word))

    if word.isdigit():
        casing = 'numeric'
    elif digitFraction > 0.5:
        casing = 'mainly_numeric'
    elif word.islower():
        casing = 'allLower'
    elif word.isupper():
        casing = 'allUpper'
    elif word[0].isupper():
        casing = 'initialUpper'
    elif numDigits > 0:
        casing = 'contains_digit'

    return caseLookup[casing]


def create_matrices(sentences, word2Idx, label2Idx, case2Idx, char2Idx):
    unknownIdx = word2Idx['UNKNOWN_TOKEN']
    paddingIdx = word2Idx['PADDING_TOKEN']

    dataset = []

    maxlen = 1200
    for sentence in sentences:
        wordcount = 0
        for word, char, label in sentence:
            wordcount += 1
        maxlen = max(maxlen, wordcount)

    for sentence in sentences:
        wordIndices = []
        caseIndices = []
        charIndices = []
        labelIndices = []

        for word, char, label in sentence:
            if word in word2Idx:
                wordIdx = word2Idx[word]
            elif word.lower() in word2Idx:
                wordIdx = word2Idx[word.lower()]
            else:
                wordIdx = unknownIdx

            charIdx = []
            for x in char:
                if x not in char2Idx:
                    charIdx.append(char2Idx[' '])
                else:
                    charIdx.append(char2Idx[x])

            wordIndices.append(wordIdx)
            charIndices.append(charIdx)
            caseIndices.append(get_casing(word, case2Idx))
            labelIndices.append(label2Idx[label])

        while len(wordIndices) < maxlen:
            wordIndices.append(paddingIdx)
        while len(charIndices) < maxlen:
            charIndices.append([char2Idx['PADDING']])
        while len(caseIndices) < maxlen:
            caseIndices.append(case2Idx['PADDING_TOKEN'])

        dataset.append([wordIndices, caseIndices, charIndices, labelIndices])

    print("sentence maxlen: %s" % maxlen)

    return dataset


def padding(sentences_matrix):

    maxlen = 52
    for sentence in sentences_matrix:
        char = sentence[2]
        for x in char:
            maxlen = max(maxlen, len(x))

    for i, sentence in enumerate(sentences_matrix):
        sentences_matrix[i][2] = pad_sequences(sentences_matrix[i][2], maxlen, padding='post')

    print("maxlen: %s" % maxlen)
    return sentences_matrix


def create_batches(data):
    l = []
    for i in data:
        l.append(len(i[0]))  # appends sentence size
    l = set(l)  # converts to set to remove duplicates
    batches = []
    batch_len = []
    z = 0
    for i in l:  # for each different sentence size
        for batch in data:  # for each sample
            if len(batch[0]) == i:  # if sentence size on batch is equal to the ones kept on 'l'
                batches.append(batch)
                z += 1
        batch_len.append(z)
    return batches, batch_len


def iterate_minibatches(dataset, batch_len):
    start = 0
    for i in batch_len:
        tokens = []
        casing = []
        char = []
        labels = []
        data = dataset[start:i]
        start = i
        for dt in data:
            t, c, ch, l = dt
            l = np.expand_dims(l, -1)
            tokens.append(t)
            casing.append(c)
            char.append(ch)
            labels.append(l)
        yield np.asarray(labels), np.asarray(tokens), np.asarray(casing), np.asarray(char)


def tag_dataset(model, dataset):
    correctLabels = []
    predLabels = []
    sentences = []
    b = Progbar(len(dataset))
    for i, data in enumerate(dataset):
        tokens, casing, char, labels = data
        tokens = np.asarray([tokens])
        casing = np.asarray([casing])
        char = np.asarray([char])
        pred = model.predict([tokens, casing, char], verbose=False)[0]
        pred = pred.argmax(axis=-1)
        correctLabels.append(labels)
        sentences.append(tokens)
        predLabels.append(pred)
        b.update(i)
    return predLabels, correctLabels, sentences