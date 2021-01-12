import numpy as np
import tensorflow as tf


# Method to compute the accruarcy. Call predict_labels to get the labels for the dataset
def compute_f1(predictions, correct, idx2Label):
    label_pred = []    
    for sentence in predictions:
        label_pred.append([idx2Label[element] for element in sentence])
        
    label_correct = []    
    for sentence in correct:
        label_correct.append([idx2Label[element] for element in sentence])

    rec = compute_precision(label_correct, label_pred)
    prec = compute_precision(label_pred, label_correct)

    f1 = 0
    if (rec+prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec)

    return prec, rec, f1

def compute_precision(guessed_sentences, correct_sentences):
    assert(len(guessed_sentences) == len(correct_sentences))

    correctCount = 0
    count = 0

    for sentenceIdx in range(len(guessed_sentences)):
        guessed = guessed_sentences[sentenceIdx]
        correct = correct_sentences[sentenceIdx]

        # print('\n')
        # print('guessed: {}'.format(guessed))
        # print('correct: {}'.format(correct))

        assert(len(guessed) == len(correct))

        idx = 0
        while idx < len(guessed):

            if guessed[idx] == 'VALUE':
                count += 1

                if guessed[idx] == correct[idx]:
                    idx += 1
                    correctCount += 1
                else:
                    idx += 1
            else:  
                idx += 1
    
    precision = 0
    if count > 0:    
        precision = float(correctCount) / count
        
    return precision