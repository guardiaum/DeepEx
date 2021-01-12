import sys
import argparse
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import TimeDistributed, Conv1D, Dense, Embedding, Input, Dropout, LSTM, Bidirectional, MaxPooling1D, \
    Flatten, concatenate
from keras.initializers import RandomUniform
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from keras.models import load_model
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, randint
from util.util import *


def data():
    class_ = sys.argv[1]
    property_name = sys.argv[2]

    train_df = get_train_data2(class_, property_name)
    print(train_df.shape)

    train_data, case2Idx, caseEmbeddings, word2Idx, wordEmbeddings, \
    char2Idx, label2Idx, sentences_maxlen, words_maxlen = prepare_data(train_df)

    val_df = get_val_data2(property_name)
    print(val_df.shape)

    val_data = embed_sentences(add_char_information_in(tag_data(val_df)), class_, property_name)

    X_train, Y_train = split_data(train_data)
    X_val, Y_val = split_data(val_data)

    return X_train, Y_train, X_val, Y_val, caseEmbeddings, wordEmbeddings, label2Idx, char2Idx, words_maxlen


def model(X_train, Y_train, X_val, Y_val, caseEmbeddings, wordEmbeddings, label2Idx, lstm_state_size):
    # word-level input
    words_input = Input(shape=(None,), dtype='int32', name='Words_input')
    words = Embedding(input_dim=wordEmbeddings.shape[0], output_dim=wordEmbeddings.shape[1],
                      weights=[wordEmbeddings], trainable=False)(words_input)

    # case-info input
    casing_input = Input(shape=(None,), dtype='int32', name='Casing_input')
    casing = Embedding(input_dim=caseEmbeddings.shape[0], output_dim=caseEmbeddings.shape[1],
                       weights=[caseEmbeddings], trainable=False)(casing_input)

    # concat & BLSTM
    output = concatenate([words, casing])

    output = Bidirectional(LSTM(lstm_state_size,
                                return_sequences=True,
                                dropout=0.6,  # on input to each LSTM block
                                recurrent_dropout=0.25  # on recurrent input signal
                                ), name="BLSTM")(output)

    output = TimeDistributed(Dense(len(label2Idx), activation='softmax'), name="Softmax_layer")(output)

    # set up model
    model = Model(inputs=[words_input, casing_input], outputs=[output])

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='nadam', metrics=['accuracy'])

    model.summary()

    # fit model
    train_batch, train_batch_len = createBatches2BLSTM_W2(X_train, Y_train)
    val_batch, val_batch_len = createBatches2BLSTM_W2(X_val, Y_val)

    model.fit_generator(iterate_minibatches_BLSTM_w2(train_batch, train_batch_len),
                        steps_per_epoch=len(train_batch),
                        # class_weight=class_weight_vect,
                        epochs=10, verbose=2, validation_steps=len(val_batch),
                        validation_data=iterate_minibatches_BLSTM_w2(val_batch, val_batch_len))

    # score, acc = model.evaluate(X_val, Y_val, verbose=0)
    score, acc = model.evaluate_generator(generator=iterate_minibatches_BLSTM_w2(val_batch, val_batch_len), steps=len(val_batch),
                                          verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


if __name__ == "__main__":
    if len(sys.argv) == 4:
        class_ = sys.argv[1]
        property_name = sys.argv[2]
        lstm_state_size = int(sys.argv[3])

        X_train, Y_train, X_val, Y_val, caseEmbeddings, wordEmbeddings, label2Idx, char2Idx, words_maxlen = data()

        result = model(X_train, Y_train, X_val, Y_val, caseEmbeddings, wordEmbeddings, label2Idx, lstm_state_size)

        best_model = result['model']

        best_model.save('models/' + class_ + '/dl/' + property_name + '-BLSTM_W2_best_model.h5')

        model = load_model('models/' + class_ + '/dl/' + property_name + '-BLSTM_W2_best_model.h5')

        test_df = get_test_data2(property_name)
        print(test_df.shape)

        test_data = embed_sentences(add_char_information_in(tag_data(test_df)), class_, property_name)

        X_test, Y_test = split_data(test_data)

        test_batch, test_batch_len = createBatches2BLSTM_W2(X_test, Y_test)

        print("Evalutation of best performing model:")
        score, acc = model.evaluate_generator(generator=iterate_minibatches_BLSTM_w2(test_batch, test_batch_len),
                                              steps=len(test_batch), verbose=0)
        print("acc on test: ", acc)
    else:
        print("INFORM PARAMETERS")