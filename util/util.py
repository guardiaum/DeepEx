import pandas as pd
import numpy as np
import json
from keras.preprocessing.sequence import pad_sequences
from nltk import word_tokenize
from util.Constants import *
from util.common import *


def get_train_data():
    
    train_sentences = TRAIN_EXT_DIR + "/"+INFOBOX_CLASS+"/" + PROPERTY_NAME +".csv"

    data = pd.read_csv(train_sentences, names=['sentence', 'ner', 'entity', 'value', 'label'],
                       dtype={'sentence': str, 'entity': str, 'value': str, 'label': str},
                       converters={'ner': eval}, encoding='utf-8')
    data['property'] = PROPERTY_NAME
    data = data[data['label'] == 't']

    print(data.keys())

    return data


def get_train_data2(class_, property_name):
    train_sentences = TRAIN_EXT_DIR + "/" + class_ + "/" + property_name + ".csv"

    data = pd.read_csv(train_sentences, names=['sentence', 'ner', 'entity', 'value', 'label'],
                       dtype={'sentence': str, 'entity': str, 'value': str, 'label': str},
                       converters={'ner': eval}, encoding='utf-8')
    data['property'] = PROPERTY_NAME
    data = data[data['label'] == 't']

    print(data.keys())

    return data


def get_test_data():
    test_sentences = TEST_ARTICLES_DIR + "/test-labeled-sentences.csv"

    df_test = pd.read_csv(test_sentences,
                          dtype={'property': str, 'entity': str, 'class': str, 'value': str, 'sentence': str},
                          converters={'ner': eval}, encoding='utf-8')

    df_test.replace(["NaN"], np.nan, inplace=True)
    df_test.dropna(inplace=True)

    print(df_test.keys())

    return df_test[df_test['property']==PROPERTY_NAME]


def get_test_data2(property_name):
    test_sentences = TEST_ARTICLES_DIR + "/test-labeled-sentences.csv"

    df_test = pd.read_csv(test_sentences,
                          dtype={'property': str, 'entity': str, 'class': str, 'value': str, 'sentence': str},
                          converters={'ner': eval}, encoding='utf-8')

    df_test.replace(["NaN"], np.nan, inplace=True)
    df_test.dropna(inplace=True)

    print(df_test.keys())

    return df_test[df_test['property']==property_name]


def get_val_data():
    val_sentences = VALIDATION_ARTICLES_DIR + "/validation-labeled-sentences.csv"

    df_val = pd.read_csv(val_sentences,
                         dtype={'property': str, 'entity': str, 'class': str, 'value': str, 'sentence': str},
                         converters={'ner': eval}, encoding='utf-8')

    df_val.replace(["NaN"], np.nan, inplace=True)
    df_val.dropna(inplace=True)

    print(df_val.keys())

    return df_val[df_val['property']==PROPERTY_NAME]


def get_val_data2(property_name):
    val_sentences = VALIDATION_ARTICLES_DIR + "/validation-labeled-sentences.csv"

    df_val = pd.read_csv(val_sentences,
                         dtype={'property': str, 'entity': str, 'class': str, 'value': str, 'sentence': str},
                         converters={'ner': eval}, encoding='utf-8')

    df_val.replace(["NaN"], np.nan, inplace=True)
    df_val.dropna(inplace=True)

    print(df_val.keys())

    return df_val[df_val['property']==property_name]


def get_val_data3(class_, property_name):
    val_sentences = VALIDATION_ARTICLES_DIR + "/validation-labeled-sentences.csv"

    df_val = pd.read_csv(val_sentences,
                         dtype={'property': str, 'entity': str, 'class': str, 'value': str, 'sentence': str},
                         converters={'ner': eval}, encoding='utf-8')

    df_val.replace(["NaN"], np.nan, inplace=True)
    df_val.dropna(inplace=True)

    print(df_val.keys())

    return df_val[(df_val['class']==class_) & (df_val['property']==property_name)]

def read_dicts(class_, property_name):
    case2Idx = None
    word2Idx = None
    char2Idx = None
    label2Idx = None

    path = "embeddings/dicts/" + class_ + "/" + property_name + "-"

    with open(path + "case2Idx.json", 'r') as f:
        case2Idx = json.load(f)
    
    with open(path + "word2Idx.json", 'r') as f:
        word2Idx = json.load(f)
    
    with open(path + "char2Idx.json", 'r') as f:
        char2Idx = json.load(f)
    
    with open(path + "label2Idx.json", 'r') as f:
        label2Idx = json.load(f)
    
    if case2Idx is not None and word2Idx is not None and char2Idx is not None and label2Idx is not None:
        return word2Idx, case2Idx, char2Idx, label2Idx
    else:
        print("ERROR READING DICTIONARIES")
        exit(0)


def split_data(data):
    
    tokens = []
    casings = []
    chars = []
    labels = []
    for t, c, ch, l in data:
        tokens.append(t)
        casings.append(c)
        chars.append(ch)
        labels.append(np.asarray(np.expand_dims(l, -1)))
    
    X = [np.asarray(tokens), np.asarray(casings), np.asarray(chars)]
    Y = [labels]
    
    return X, Y


def split_data_blstm(data):
    tokens = []
    labels = []
    for t, c, ch, l in data:
        tokens.append(t)
        labels.append(np.asarray(np.expand_dims(l, -1)))

    X = [np.asarray(tokens)]
    Y = [labels]
    return X, Y


def split_data_blstm_w2(data):
    tokens = []
    casings = []
    labels = []
    for t, c, ch, l in data:
        tokens.append(t)
        casings.append(c)
        labels.append(np.asarray(np.expand_dims(l, -1)))

    X = [np.asarray(tokens), np.asarray(casings)]
    Y = [labels]
    return X, Y

def define_dicts(words):

    label2Idx = {'O': 0, 'VALUE': 1, 'PROP': 2}  #, 'PAD': 0}

    # mapping for token cases
    case2Idx = {'numeric': 0, 'allLower': 1, 'allUpper': 2, 'initialUpper': 3, 'other': 4, 'mainly_numeric': 5,
                'contains_digit': 6, 'PADDING_TOKEN': 7}
    caseEmbeddings = np.identity(len(case2Idx), dtype='float32')  # identity matrix used
    #caseEmbeddings = caseEmbeddings[0:8]

    # read GLoVE word embeddings
    word2Idx = {}
    wordEmbeddings = []

    fEmbeddings = open("embeddings/glove.6B/glove.6B.50d.txt", encoding="utf-8")

    # loop through each word in embeddings
    for line in fEmbeddings:
        split = line.strip().split(" ")
        word = split[0]  # embedding word entry

        if len(word2Idx) == 0:  # add padding+unknown
            word2Idx["PADDING_TOKEN"] = len(word2Idx)
            vector = np.zeros(len(split) - 1)  # zero vector for 'PADDING' word
            wordEmbeddings.append(vector)

            word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
            vector = np.random.uniform(-0.25, 0.25, len(split) - 1)
            wordEmbeddings.append(vector)

        if word.lower() in words:
            vector = np.array([float(num) for num in split[1:]])
            wordEmbeddings.append(vector)  # word embedding vector
            word2Idx[word] = len(word2Idx)  # corresponding word dict

    wordEmbeddings = np.array(wordEmbeddings)

    # dictionary of all possible characters
    char2Idx = {"PADDING": 0, "UNKNOWN": 1}
    for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|<>":
        char2Idx[c] = len(char2Idx)

    path = "embeddings/dicts/"+INFOBOX_CLASS+"/"+PROPERTY_NAME+"-"

    with open(path + "case2Idx.json", 'w') as f:
        json.dump(case2Idx, f)
    
    with open(path + "word2Idx.json", 'w') as f:
        json.dump(word2Idx, f)
    
    with open(path + "char2Idx.json", 'w') as f:
        json.dump(char2Idx, f)
    
    with open(path + "label2Idx.json", 'w') as f:
        json.dump(label2Idx, f)
    
    return case2Idx, caseEmbeddings, word2Idx, wordEmbeddings, char2Idx, label2Idx


def embed_sentences(sentences, class_, property_name):
    word2Idx, case2Idx, char2Idx, label2Idx = read_dicts(class_, property_name)
    data, sentences_maxlen, words_maxlen = create_matrices(sentences, word2Idx, label2Idx, case2Idx, char2Idx)
    return data

    
def embed(sentences):
    """Create word- and character-level embeddings"""

    labelSet = set()
    words = {}

    # unique words and labels in data
    for dataset in sentences:
        for sentence in dataset:
            for token, char, label in [sentence]:
                # token ... token, char ... list of chars, label ... BIO labels
                labelSet.add(label)
                words[token.lower()] = True

    case2Idx, caseEmbeddings, word2Idx, wordEmbeddings, char2Idx, label2Idx = define_dicts(words)

    # format: [[wordindices], [label indices], [caseindices], [padded word indices]]
    data, sentences_maxlen, words_maxlen = create_matrices(sentences, word2Idx, label2Idx, case2Idx, char2Idx)

    return data, case2Idx, caseEmbeddings, word2Idx, wordEmbeddings, char2Idx, label2Idx, sentences_maxlen, words_maxlen


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


def prepare_data(data):
    # unique_properties = df_test['property'].unique()
    # for prop in unique_properties:
    #print("CLASS: {}\n".format(prop))
    #prop_data = data.loc[data['property'] == prop]
    tagged_data = tag_data(data)
    sentences = add_char_information_in(tagged_data)
    return embed(sentences)


def tag_data(test_data):
    subset = test_data[['ner', 'sentence']]
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

def createBatches(data):
    l = []
    for i in data:
        l.append(len(i[0]))
    l = set(l)
    batches = []
    batch_len = []
    z = 0
    for i in l:
        for batch in data:
            if len(batch[0]) == i:
                batches.append(batch)
                z += 1
        batch_len.append(z)
    return batches,batch_len

def createBatches2CNN_BLSTM(X, y):
    l = []
    for i in X[0]:
        l.append(len(i))
    l = set(l)
    
    batches = []
    batch_len = []
    z = 0
    for i in l:
        for words, case, char, label in zip(X[0], X[1], X[2], y[0]):
            if len(words) == i:
                batches.append([words, case, char, label])
                z += 1
        batch_len.append(z)
    return batches, batch_len


def createBatches2BLSTM_W2(X, y):
    l = []
    for i in X[0]:
        l.append(len(i))
    l = set(l)

    batches = []
    batch_len = []
    z = 0
    for i in l:
        for words, case, label in zip(X[0], X[1], y[0]):
            if len(words) == i:
                batches.append([words, case, label])
                z += 1
        batch_len.append(z)
    return batches, batch_len


def createBatches2BLSTM(X, y):
    l = []
    for i in X[0]:
        l.append(len(i))
    l = set(l)

    batches = []
    batch_len = []
    z = 0
    for i in l:
        for words, label in zip(X[0], y[0]):
            if len(words) == i:
                batches.append([words, label])
                z += 1
        batch_len.append(z)
    return batches, batch_len

def iterate_minibatches_CNN_BLSTM(dataset, batch_len):
    while True:
        for dt in dataset:
            t, c, ch, l = dt
            l = np.expand_dims(l, axis=0)
            ch = np.expand_dims(ch, axis=0)
            c = np.expand_dims(c, axis=0)
            t = np.expand_dims(t, axis=0)

            yield ({'Words_input': np.asarray(t), 'Casing_input': np.asarray(c), 'Character_input': np.asarray(ch)}, {'Softmax_layer': l})       


def iterate_minibatches_BLSTM_w2(dataset, batch_len):
    while True:
        for dt in dataset:
            t, c, l = dt
            l = np.expand_dims(l, axis=0)
            c = np.expand_dims(c, axis=0)
            t = np.expand_dims(t, axis=0)

            yield ({'Words_input': np.asarray(t), 'Casing_input': np.asarray(c)}, {'Softmax_layer': l})


def iterate_minibatches_BLSTM(dataset, batch_len):
    while True:
        for dt in dataset:
            t, l = dt
            l = np.expand_dims(l, axis=0)
            t = np.expand_dims(t, axis=0)

            yield ({'Words_input': np.asarray(t)}, {'Softmax_layer': l})


def create_matrices(sentences, word2Idx, label2Idx, case2Idx, char2Idx):
    
    sentences_maxlen = 1200
    for sentence in sentences:
        wordcount = 0
        for word, char, label in sentence:
            wordcount += 1
        sentences_maxlen = max(sentences_maxlen, wordcount)

    '''PADDING FOR SENTENCES AND EMBED OF WORDS, WORD CASING AND CHARACTERS'''
    dataset = []
    
    words_maxlen = 50
    for sentence in sentences:
        wordIndices = []
        caseIndices = []
        charIndices = []
        labelIndices = []

        for word, char, label in sentence:
            if word in word2Idx:
                wordIdx = word2Idx[word]
                words_maxlen = max(words_maxlen, len(word))
            elif word.lower() in word2Idx:
                wordIdx = word2Idx[word.lower()]
                words_maxlen = max(words_maxlen, len(word))
            else:
                wordIdx = word2Idx['UNKNOWN_TOKEN']
            
            charIdx = []
            for x in char:
                if x not in char2Idx:
                    charIdx.append(char2Idx[' '])
                else:
                    charIdx.append(char2Idx[x])

            wordIndices.append(wordIdx)
            caseIndices.append(get_casing(word, case2Idx))
            charIndices.append(charIdx)
            labelIndices.append(label2Idx[label])
        
        dataset.append([np.asarray(wordIndices), np.asarray(caseIndices), np.asarray(charIndices), np.asarray(labelIndices)])
   
    '''PADDING FOR CHARACTERS'''
    for i, sentence in enumerate(dataset):
       dataset[i][2] = pad_sequences(dataset[i][2], words_maxlen, padding='post')
    
    return dataset, sentences_maxlen, words_maxlen


def sentence2input(sentence, class_, property_name, type_="CNN_BLSTM"):
    word2Idx, case2Idx, char2Idx, label2Idx = read_dicts(class_, property_name)
    
    wordIndices = []
    caseIndices = []
    charIndices = []

    words_maxlen = 50
    for word, char in sentence:
        if word in word2Idx:
            wordIdx = word2Idx[word]
        elif word.lower() in word2Idx:
            wordIdx = word2Idx[word.lower()]
        else:
            wordIdx = word2Idx['UNKNOWN_TOKEN']

        words_maxlen = max(words_maxlen, len(word))

        charIdx = []
        for x in char:
            if x not in char2Idx:
                charIdx.append(char2Idx[' '])
            else:
                charIdx.append(char2Idx[x])

        wordIndices.append(wordIdx)
        caseIndices.append(get_casing(word, case2Idx))
        charIndices.append(charIdx)

    dataset = [wordIndices, caseIndices, charIndices]

    '''PADDING FOR CHARACTERS'''
    dataset[2] = pad_sequences(dataset[2], words_maxlen, padding='post')

    tks = [dataset[0]]
    casings = [dataset[1]]
    chars = [dataset[2]]

    X_predict = None
    if type_ == "CNN_BLSTM":
        X_predict = [np.array(tks), np.array(casings), np.array(chars)]
    elif type_ == "BLSTM_W2":
        X_predict = [np.array(tks), np.array(casings)]
    elif type_ == "BLSTM":
        X_predict = np.array(tks)
    
    return X_predict, label2Idx


def predict_sentence(model, raw_sentence, class_, property_name, type_="CNN_BLSTM"):

    tokens = word_tokenize(raw_sentence)

    sentence = []
    for i, tk in enumerate(tokens):
        chars = [c for c in tk]
        sentence.append([tk, chars])

    X_predict, label2Idx = sentence2input(sentence, class_, property_name, type_)

    idx2Label = {v: k for k, v in label2Idx.items()}

    percents = model.predict(X_predict)[0]

    prediction = percents.argmax(axis=-1)
    #print(prediction)

    marginals = []
    for row in percents:
        temp = {}
        for i, perc in enumerate(row):
            label = idx2Label[i]
            temp[label] = perc
        marginals.append(temp)

    # convert back labels representation to label value, add token
    label_preds = []
    token_words = []
    for i, p in enumerate(prediction):
        label = idx2Label[p]
        label_preds.append(label)
        token_words.append(tokens[i])

    #print(label_preds)

    extractions = heuristicTokensSelection(label_preds, marginals, token_words)

    if extractions is not None and len(extractions) > 0:
        print(raw_sentence)
        print(extractions)
        return extractions
    else:
        return []