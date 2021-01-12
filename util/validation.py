import pandas as pd
import numpy as np
import nltk

class_ = 'airline'
execution = class_+'-ipopulator'


def multivalued(extracted_value, expected_value):
    subvalues = extracted_value.split('/')
    subvalues = map(str.strip, subvalues)
    subvalues = list(filter(lambda a: a != 'NA', subvalues))

    if len(subvalues)==0:  # for ipopulator cases where all extractions for a value can be NA
        confusion_matrix['FN'] = confusion_matrix['TP'] + 1
    else:
        exp_values = expected_value.split('/')
        ext_values = extracted_value.split('/')

        ext_values = list(filter(lambda x: 'NA' not in x, ext_values))
        ext = ext_values[0]
        #print(exp_values)
        #print(ext_values)
        #print("\n")
        general = 0
        for exp in exp_values:
            positive = 0
            #for ext in ext_values:
            similarity = nltk.jaccard_distance(set(ext), set(exp))
            if similarity <= 0.5:
                positive += 1

            if positive != 0:
                general += 1

        if general != 0:
            if (general / len(set(exp_values))) >= 0.5:
                confusion_matrix['TP'] = confusion_matrix['TP'] + 1
            else:
                confusion_matrix['FP'] = confusion_matrix['FP'] + 1
        else:
            confusion_matrix['FP'] = confusion_matrix['FP'] + 1


extraction = pd.read_csv("../results/extractions/{}.csv".format(execution), dtype=np.dtype('str'))
truth = pd.read_csv("../results/ground-truth/{}.csv".format(class_), dtype=np.dtype('str'))

file = open("../results/output/{}.log".format(execution), 'w')

extraction = extraction.replace('\n','', regex=True)
truth = truth.replace('\n','', regex=True)

properties = extraction.columns.values.tolist()

properties.remove('article')
print(properties)


'''
    CONFUSION MATRIX
'''
all_props_conf_matrix = []

for prop in properties:

    confusion_matrix = None

    if not any(d['PROP'] == prop for d in all_props_conf_matrix):
        confusion_matrix = {'PROP': prop, 'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    else:
        print("confusion exits")
        for d in all_props_conf_matrix:
            if d['PROP'] == prop:
                confusion_matrix = d

    if confusion_matrix is not None:
        for i, article in extraction.iterrows():
            #print(article['article'])
            expected = truth[truth['article'] == article['article']]
            #print(expected)
            extracted_value = article[prop]
            expected_value = expected[prop].values[0]

            extractedIsNull = pd.Series(article[prop]).isnull().values.any()
            expectedIsNull = expected[prop].isnull().values.any()

            if expectedIsNull and extractedIsNull:
                confusion_matrix['TN'] = confusion_matrix['TN'] + 1
            elif expectedIsNull and extractedIsNull == False and all(i not in extracted_value for i in ['/', ',']):
                confusion_matrix['FP'] = confusion_matrix['FP'] + 1
            elif expectedIsNull == False and extractedIsNull:
                confusion_matrix['FN'] = confusion_matrix['FN'] + 1
            elif expectedIsNull == False and extractedIsNull == False:

                '''
                    for multivalued cases it verifies each returned value
                    if the jaccard similarity between the individual value and the expected
                    is minor than 0.5, it is a positive extraction.
                    The proportion of positive extractions is obtained and if at least 50 perc
                    of the values are positives than we have a [TP] else [FP]
                '''
                if any(i in extracted_value for i in ['/', ',']):
                    multivalued(extracted_value, expected_value)
                    continue
                else:
                    similarity = nltk.jaccard_distance(set(extracted_value), set(expected_value))
                    print(similarity)

                    if similarity > 0.5:
                        confusion_matrix['FP'] = confusion_matrix['FP'] + 1
                    elif similarity <= 0.5:
                        confusion_matrix['TP'] = confusion_matrix['TP'] + 1

        all_props_conf_matrix.append(confusion_matrix)


'''
    METRICS (PRECISION, RECALL, F-SCORE)
'''
for dict in all_props_conf_matrix:
    prop = dict['PROP']
    tp = dict['TP']
    fp = dict['FP']
    fn = dict['FN']

    try:
        precision = float(tp) / (float(tp) + float(fp))
        dict['precision'] = precision
    except ZeroDivisionError:
        dict['precision'] = 0.0

    try:
        recall = float(tp) / (float(tp) + float(fn))
        dict['recall'] = recall
    except ZeroDivisionError:
        dict['recall'] = 0.0

    try:
        f_score = 2 * dict['precision'] * dict['recall'] / (float(dict['precision']) + float(dict['recall']))
        dict['f_score'] = f_score
    except ZeroDivisionError:
        dict['f_score'] = 0.0

    print(dict)
    file.write('\n' + str(dict))


'''
    METRICS (MACRO, MICRO)
'''

macro_precision = 0.0
macro_recall = 0.0
macro_f_score = 0.0

df = pd.DataFrame.from_dict(all_props_conf_matrix)
macro_precision = df["precision"].mean()
macro_recall = df["recall"].mean()
macro_f_score = (2 * macro_precision * macro_recall) / (macro_precision + macro_recall)

print('\nMACRO PRECISION: {}'.format(macro_precision))
file.write('\n\nMACRO PRECISION: {}'.format(macro_precision))
print('MACRO RECALL: {}'.format(macro_recall))
file.write('\nMACRO RECALL: {}'.format(macro_recall))
print('MACRO F-SCORE: {}'.format(macro_f_score))
file.write('\nMACRO F-SCORE: {}'.format(macro_f_score))

micro_precision = df["TP"].sum() / (df["TP"].sum() + df["FP"].sum())
micro_recall = df["TP"].sum() / (df["TP"].sum() + df["FN"].sum())
micro_f_score = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)

print('\nMICRO PRECISION: {}'.format(micro_precision))
file.write('\n\nMICRO PRECISION: {}'.format(micro_precision))
print('MICRO RECALL: {}'.format(micro_recall))
file.write('\nMICRO RECALL: {}'.format(micro_recall))
print('MICRO F-SCORE: {}'.format(micro_f_score))
file.write('\nMICRO F-SCORE: {}'.format(micro_f_score))


print('\n\nPRECISION list')
file.write('\n\nPRECISION list\n')
for dict in all_props_conf_matrix:
    print('{}\t:\t{}'.format(dict['PROP'], dict['precision']))
    file.write('{}\t:\t{}\n'.format(dict['PROP'], dict['precision']))


print('\n\nRECALL list')
file.write('\n\nRECALL list\n')
for dict in all_props_conf_matrix:
    print('{}\t:\t{}'.format(dict['PROP'], dict['recall']))
    file.write('{}\t:\t{}\n'.format(dict['PROP'], dict['recall']))


print('\n\nF-SCORE list')
file.write('\n\nF-SCORE list\n')
for dict in all_props_conf_matrix:
    print('{}\t:\t{}'.format(dict['PROP'], dict['f_score']))
    file.write('{}\t:\t{}\n'.format(dict['PROP'], dict['f_score']))

file.close()
