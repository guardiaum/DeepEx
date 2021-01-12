import numpy as np
import tensorflow as tf
import os
from keras.models import Model, load_model, Sequential
from keras.layers import TimeDistributed, Conv1D, Dense, Embedding, Input, Dropout, LSTM, Bidirectional, MaxPooling1D, \
    Flatten, concatenate
from keras.utils import plot_model
from keras.initializers import RandomUniform
from util.CNN_BLSTM_util import *
from util.CNN_BLSTM_validation import compute_f1
from util.common import *


class BLSTM(object):

    def __init__(self, class_, type, propertyName, EPOCHS, DROPOUT, DROPOUT_RECURRENT, LSTM_STATE_SIZE, CONV_SIZE, OPTIMIZER):

        self.epochs = EPOCHS
        self.dropout = DROPOUT
        self.dropout_recurrent = DROPOUT_RECURRENT
        self.lstm_state_size = LSTM_STATE_SIZE
        self.conv_size = CONV_SIZE
        self.optimizer = OPTIMIZER
        # save model
        self.modelName = "{}_{}_{}_{}_{}_{}_{}_{}".format(type, propertyName, self.epochs,
                                                       self.dropout,
                                                       self.dropout_recurrent,
                                                       self.lstm_state_size,
                                                       self.conv_size,
                                                       self.optimizer.__class__.__name__
                                                       )
        self.model_path = "./models/" + class_ + "/dl/" + self.modelName + ".h5"
        exists = os.path.isdir("./models/" + class_ + "/dl/")
        if exists is False:
            os.makedirs("./models/" + class_ + "/dl/")

        self.trainSentences = []  #, self.devSentences = [], []
        self.train_set = []  # self.dev_set = [], []
        self.train_batch = []  # self.train_batch_len = [], []
        # self.dev_batch, self.dev_batch_len = [], []
        self.model = None


    def loadData(self, property_name, property_path):
        """Load data and add character information"""
        self.trainSentences = read_file(property_name, property_path)
        # self.devSentences = read_validation_file('./datasets/validation-sentences.csv', property_name)

    def addCharInfo(self):
        # format: [['EU', ['E', 'U'], 'B-ORG\n'], ...]
        print("train: {}".format(len(self.trainSentences)))
        self.trainSentences = add_char_information_in(self.trainSentences)
        # self.devSentences = add_char_information_in(self.devSentences)

    def embed(self):
        """Create word- and character-level embeddings"""

        labelSet = set()
        words = {}

        # unique words and labels in data
        for dataset in self.trainSentences:
            for sentence in dataset:
                for token, char, label in [sentence]:
                    # token ... token, char ... list of chars, label ... BIO labels
                    labelSet.add(label)
                    words[token.lower()] = True

        # mapping for labels
        self.label2Idx = {'O': 0, 'VALUE': 1, 'PROP': 2}
        #for label in labelSet:
        #    self.label2Idx[label] = len(self.label2Idx)

        # mapping for token cases
        self.case2Idx = {'numeric': 0, 'allLower': 1, 'allUpper': 2, 'initialUpper': 3, 'other': 4, 'mainly_numeric': 5,
                    'contains_digit': 6, 'PADDING_TOKEN': 7}
        self.caseEmbeddings = np.identity(len(self.case2Idx), dtype='float32')  # identity matrix used

        # read GLoVE word embeddings
        self.word2Idx = {}
        self.wordEmbeddings = []

        fEmbeddings = open("embeddings/glove.6B/glove.6B.50d.txt", encoding="utf-8")

        # loop through each word in embeddings
        for line in fEmbeddings:
            split = line.strip().split(" ")
            word = split[0]  # embedding word entry

            if len(self.word2Idx) == 0:  # add padding+unknown
                self.word2Idx["PADDING_TOKEN"] = len(self.word2Idx)
                vector = np.zeros(len(split) - 1)  # zero vector for 'PADDING' word
                self.wordEmbeddings.append(vector)

                self.word2Idx["UNKNOWN_TOKEN"] = len(self.word2Idx)
                vector = np.random.uniform(-0.25, 0.25, len(split) - 1)
                self.wordEmbeddings.append(vector)

            if word.lower() in words:
                vector = np.array([float(num) for num in split[1:]])
                self.wordEmbeddings.append(vector)  # word embedding vector
                self.word2Idx[word] = len(self.word2Idx)  # corresponding word dict

        self.wordEmbeddings = np.array(self.wordEmbeddings)

        # dictionary of all possible characters
        self.char2Idx = {"PADDING": 0, "UNKNOWN": 1}
        for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|<>":
            self.char2Idx[c] = len(self.char2Idx)

        # format: [[wordindices], [caseindices], [padded word indices], [label indices]]
        self.train_set = padding(create_matrices(self.trainSentences, self.word2Idx, self.label2Idx, self.case2Idx, self.char2Idx))
        # self.dev_set = padding(create_matrices(self.devSentences, self.word2Idx, self.label2Idx, self.case2Idx, self.char2Idx))

        self.idx2Word = {v: k for k, v in self.word2Idx.items()}
        self.idx2Label = {v: k for k, v in self.label2Idx.items()}


    def createBatches(self):
        """Create batches"""
        self.train_batch, self.train_batch_len = create_batches(self.train_set)
        # self.dev_batch, self.dev_batch_len = create_batches(self.dev_set)


    def tag_dataset(self, dataset, model):
        """Tag data with numerical values"""
        correctLabels = []
        predLabels = []
        for i, data in enumerate(dataset):
            tokens, casing, char, labels = data
            tokens = np.asarray([tokens])
            casing = np.asarray([casing])
            char = np.asarray([char])
            pred = model.predict([tokens, casing, char], verbose=False)[0]
            pred = pred.argmax(axis=-1)  # Predict the classes

            correctLabels.append(labels)
            predLabels.append(pred)

        return predLabels, correctLabels

    def predict(self, model, sentence):
        # put samples (sentences) in the format [token, label] --- ignore label
        tokens = word_tokenize(sentence)
        sample = [[token, 'O'] for token in tokens]

        # put samples in format [token, [chars], [label]] --- find a way to ignore label info
        sample = add_char_information_in([sample])
        sample = padding(create_matrices(sample, self.word2Idx, self.label2Idx, self.case2Idx, self.char2Idx))

        tokens_np, casing, char, label = sample[0]
        tokens_np = np.asarray([tokens_np])
        casing = np.asarray([casing])
        char = np.asarray([char])

        pred = model.predict([tokens_np], verbose=False)[0]

        percent = pred

        marginals = []
        for row in percent:
            temp = {}
            for i, perc in enumerate(row):
                label = self.idx2Label[i]
                temp[label] = perc
            marginals.append(temp)

        prediction = pred.argmax(axis=-1)

        # convert back labels representation to label value, add token
        label_preds = []
        token_words = []
        for i, p in enumerate(prediction):
            label = self.idx2Label[p]
            label_preds.append(label)
            token_words.append(tokens[i])

        extractions = heuristicTokensSelection(label_preds, marginals, token_words)

        if extractions is not None and len(extractions) > 0:
            print(sentence)
            print(extractions)
            return extractions

        return []

    def buildModel(self):
        """Model layers"""

        # word-level input
        words_input = Input(shape=(None,), dtype='int32', name='words_input')
        words = Embedding(input_dim=self.wordEmbeddings.shape[0], output_dim=self.wordEmbeddings.shape[1],
                          weights=[self.wordEmbeddings],
                          trainable=False)(words_input)

        # concat & BLSTM
        output = Bidirectional(LSTM(self.lstm_state_size,
                      return_sequences=True,
                      dropout=self.dropout,  # on input to each LSTM block
                      recurrent_dropout=self.dropout_recurrent  # on recurrent input signal
                      ), name="BLSTM")(words)

        output = TimeDistributed(Dense(len(self.label2Idx), activation='softmax'), name="Softmax_layer")(output)

        # set up model
        self.model = Model(inputs=[words_input], outputs=[output])

        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=self.optimizer)

        self.init_weights = self.model.get_weights()


    def train(self):
        """Default training"""

        if self.verify_model_exists() is False:

            for epoch in range(self.epochs):
                print("\nEpoch {}/{}".format(epoch, self.epochs))
                for i, batch in enumerate(iterate_minibatches(self.train_batch, self.train_batch_len)):
                    labels, tokens, casing, char = batch
                    self.model.train_on_batch([tokens], labels)

                # compute F1 scores
                '''predLabels, correctLabels = self.tag_dataset(self.dev_batch, self.model)
                pre_dev, rec_dev, f1_dev = compute_f1(predLabels, correctLabels, self.idx2Label)
                print("prec dev ", round(pre_dev, 4))
                print("rec dev", round(rec_dev, 4))
                print("f1 dev ", round(f1_dev, 4))'''

            self.model.save(self.model_path)
            print("Model weights saved.")
            print("Training finished.")

        # self.model.set_weights(self.init_weights)  # clear model
        print("Model weights cleared.")

    def verify_model_exists(self):
        exists = os.path.isfile(self.model_path)
        if exists:
            print("model exists")
            print("loading...")
            self.model = load_model(self.model_path)
            return True
        return False

    print("Class initialized.")
