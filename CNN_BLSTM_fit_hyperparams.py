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
    train_df = get_train_data()
    #found = train_df['ner']
    #hasValue = [True if 'VALUE' in x else False for i, x in enumerate(found)]
    #train_df = train_df[np.array(hasValue)]
    print(train_df.shape)

    train_data, case2Idx, caseEmbeddings, word2Idx, wordEmbeddings, \
    char2Idx, label2Idx, sentences_maxlen, words_maxlen = prepare_data(train_df)

    val_df = get_val_data()
    print(val_df.shape)

    val_data = embed_sentences(add_char_information_in(tag_data(val_df)), INFOBOX_CLASS, PROPERTY_NAME)

    X_train, Y_train = split_data(train_data)
    X_val, Y_val = split_data(val_data)

    return X_train, Y_train, X_val, Y_val, caseEmbeddings, wordEmbeddings, label2Idx, char2Idx, sentences_maxlen, words_maxlen


def model(X_train, Y_train, X_val, Y_val, caseEmbeddings, wordEmbeddings, label2Idx, char2Idx, sentences_maxlen,
          words_maxlen):
    temp = []
    for item in Y_train[0]:
        flatten = [i for sublist in item for i in sublist]
        for i in flatten:
            temp.append(i)

    temp = np.asarray(temp)
    print('labels', np.unique(np.ravel(temp, order='C')))

    # word-level input
    words_input = Input(shape=(None,), dtype='int32', name='Words_input')
    words = Embedding(input_dim=wordEmbeddings.shape[0], output_dim=wordEmbeddings.shape[1],
                      weights=[wordEmbeddings], trainable=False)(words_input)

    # case-info input
    casing_input = Input(shape=(None,), dtype='int32', name='Casing_input')
    casing = Embedding(input_dim=caseEmbeddings.shape[0], output_dim=caseEmbeddings.shape[1],
                       weights=[caseEmbeddings], trainable=False)(casing_input)

    # character input
    character_input = Input(shape=(None, words_maxlen,), name="Character_input")
    embed_char_out = TimeDistributed(
        Embedding(input_dim=len(char2Idx), output_dim=50,
                  embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)),
        name="Character_embedding")(character_input)

    dropout = Dropout(0.5)(embed_char_out)

    # CNN
    conv1d_out = TimeDistributed(
        Conv1D(kernel_size={{choice([3, 5])}}, filters=10,
               padding='same', activation='tanh', strides=1),
        name="Convolution")(dropout)
    maxpool_out = TimeDistributed(MaxPooling1D({{choice([10, 25, 50])}}), name="Maxpool")(conv1d_out)
    char = TimeDistributed(Flatten(), name="Flatten")(maxpool_out)
    char = Dropout(0.5)(char)

    # concat & BLSTM
    output = concatenate([words, casing, char])

    output = Bidirectional(LSTM({{choice([100, 200, 300])}},
                                return_sequences=True,
                                dropout=0.5,  # on input to each LSTM block
                                recurrent_dropout=0.25  # on recurrent input signal
                                ), name="BLSTM")(output)

    output = TimeDistributed(Dense(len(label2Idx), activation='softmax'), name="Softmax_layer")(output)

    # set up model
    model = Model(inputs=[words_input, casing_input, character_input], outputs=[output])

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='nadam', metrics=['accuracy'])

    model.summary()

    train_batch, train_batch_len = createBatches2CNN_BLSTM(X_train, Y_train)
    val_batch, val_batch_len = createBatches2CNN_BLSTM(X_val, Y_val)

    model.fit_generator(iterate_minibatches_CNN_BLSTM(train_batch, train_batch_len),
                        steps_per_epoch=len(train_batch),
                        # class_weight=class_weight_vect,
                        epochs=10, verbose=2, validation_steps=len(val_batch),
                        validation_data=iterate_minibatches_CNN_BLSTM(val_batch, val_batch_len))

    # score, acc = model.evaluate(X_val, Y_val, verbose=0)
    score, acc = model.evaluate_generator(generator=iterate_minibatches_CNN_BLSTM(val_batch, val_batch_len), steps=len(val_batch),
                                          verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


if __name__ == "__main__":

    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials(),
                                          functions=[createBatches2CNN_BLSTM, iterate_minibatches_CNN_BLSTM])

    plot_model(best_model, to_file='models/' + PROPERTY_NAME + '-CNN_BLSTM_best_model.png', show_shapes=True, show_layer_names=True)
    best_model.save('models/'+INFOBOX_CLASS+'/dl/' + PROPERTY_NAME + '-CNN_BLSTM_best_model.h5')

    print("Best performing model chosen hyper-parameters:")
    print(best_run)

    model = load_model('models/'+INFOBOX_CLASS+'/dl/' + PROPERTY_NAME + '-CNN_BLSTM_best_model.h5')

    test_df = get_test_data()
    print(test_df.shape)

    test_data = embed_sentences(add_char_information_in(tag_data(test_df)), INFOBOX_CLASS, PROPERTY_NAME)

    X_test, Y_test = split_data(test_data)

    temp = []
    for item in Y_test[0]:
        flatten = [i for sublist in item for i in sublist]
        for i in flatten:
            temp.append(i)

    temp = np.asarray(temp)
    print('labels', np.unique(np.ravel(temp, order='C')))

    test_batch, test_batch_len = createBatches2CNN_BLSTM(X_test, Y_test)

    print("Evalutation of best performing model:")
    score, acc = model.evaluate_generator(generator=iterate_minibatches_CNN_BLSTM(test_batch, test_batch_len),
                                          steps=len(test_batch), verbose=0)
    print("acc on test: ", acc)
