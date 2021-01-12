import sys
import pandas as pd
import numpy as np
import codecs
import io
from util.Constants import *
from generate_train_data import *
from nltk.tokenize import sent_tokenize, word_tokenize

def run(CLASS):
    df = pd.read_csv(DATASETS + "/validation-sentences.csv", encoding="utf-8")

    pd.set_option('display.max_columns', 50)
    pd.set_option('display.max_colwidth', -1)
    #print(df[['entity', 'property', 'value']].sample(4))

    df_class = df[df['class'] == CLASS]
    entities_class = df_class.entity.unique()
    properties = df_class.property.unique()
    #print(entities_class)

    data = []
    for entity in entities_class:
        text = readText(CLASS, entity)
        text = text.replace("\n", " ")

        if text is not None or len(text) == 0:
            entity_tuples = df_class[df_class['entity'] == entity]

            tuples = []
            sentences = []
            for i in range(0, len(entity_tuples)):
                prop = entity_tuples.iloc[i, entity_tuples.columns.get_loc('property')]
                value = entity_tuples.iloc[i, entity_tuples.columns.get_loc('value')]
                sentence = entity_tuples.iloc[i, entity_tuples.columns.get_loc('sentence')]
                tuples.append([prop, value])
                sentences.append([prop, sentence])

            # instantiate soft tf-idf and overlap coefficient
            soft_tfidf, oc = instantiateStringMatching(text, tuples)

            for i, tuple in enumerate(tuples):

                if tuple[0] == sentences[i][0]:
                    sentence = sentences[i][1]
                    sentence_tokens = word_tokenize(sentence)
                    attribute_tokens = word_tokenize(tuple[0].replace("_", " ") + " " + tuple[1])

                    soft_raw_score = get_max_similarity_on_window(sentence_tokens, attribute_tokens, soft_tfidf)
                    oc_raw_score = 0.0
                    #soft_raw_score = soft_tfidf.get_raw_score(sentence_tokens, attribute_tokens)
                    #oc_raw_score = oc.get_raw_score(sentence_tokens, attribute_tokens)
                    data.append({'prop': tuple[0], 'soft_tf-idf': soft_raw_score, 'overlap_coef': oc_raw_score})

    temp_df = pd.DataFrame.from_dict(data)

    final_df = temp_df.groupby('prop').agg({"soft_tf-idf": {
        "median_soft": "median", "mean_soft": "mean", "max_soft": "max", "min_soft": "min"},
        "overlap_coef": {"median_oc": "median", "mean_oc": "mean", "max_oc": "max", "min_oc": "min"}})

    final_df.to_csv(EXPERIMENTS_DIR + "/" + CLASS + "-threshold_validation.csv")

    return final_df


def token_sliding_window(tokens, size):
    for i in range(len(tokens) - size + 1):
        yield tokens[i: i + size]


def get_similarity_and_set_ner(sentence_tokens, prop, value, soft_tfidf):

    big_score = get_max_similarity_on_window(sentence_tokens, prop, value, soft_tfidf)

    propTokens = word_tokenize(prop)
    valueTokens = word_tokenize(value)

    # get ner annotations according to soft tf-idf measure
    kept_index_prop = [-1] * len(propTokens)
    bigger_token_score = [0.0] * len(propTokens)
    for i, prop_token in enumerate(propTokens):
        for j, token in enumerate(sentence_tokens):
            score = soft_tfidf.get_raw_score([prop_token], [token])
            if score > bigger_token_score[i]:
                bigger_token_score[i] = score
                kept_index_prop[i] = j

    kept_index_value = [-1] * len(valueTokens)
    bigger_value_score = [0.0] * len(valueTokens)
    for i, value_token in enumerate(valueTokens):
        for j, token in enumerate(sentence_tokens):
            score = soft_tfidf.get_raw_score([value_token], [token])
            if score > bigger_value_score[i]:
                bigger_value_score[i] = score
                kept_index_value[i] = j

    ner = [''] * len(sentence_tokens)
    for index, token in enumerate(sentence_tokens):
        if index in kept_index_prop:
            ner[index] = 'PROP'
            continue
        if index in kept_index_value:
            ner[index] = 'VALUE'
            continue
        ner[index] = 'O'

    '''print('\nkept prop: {}'.format(kept_index_prop))
    print('kept value: {}'.format(kept_index_value))
    print('Sentence: {}'.format(sentence_tokens))
    print('NER: {}'.format(ner))
    print('{}: {}'.format(prop, value))
    print('Bigger window score: {}'.format(big_score))'''
    return big_score, ner


def get_max_similarity_on_window(sentence_tokens, prop, value, soft_tfidf):
    attributeTokens = word_tokenize(prop + " " + value)

    big_score = 0.0
    for window in token_sliding_window(sentence_tokens, 5):
        score = soft_tfidf.get_raw_score(window, attributeTokens)
        # print("{} : {} = {}".format(window, attribute_tokens, score))
        if score > big_score:
            big_score = score
    return big_score

def readText(CLASS, entity):
    validation_path = VALIDATION_ARTICLES_DIR + "/" + CLASS + "/" + entity
    article_text_f = codecs.open(validation_path, 'r', encoding="utf-8")
    article_text = article_text_f.read().replace("\n", ' ')
    return article_text


if __name__ == "__main__":
    if len(sys.argv) == 2:
        CLASS = sys.argv[1]
        run(CLASS)
    else:
        print("PLEASE INFORM CLASS")