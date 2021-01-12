import os
import csv
import pickle
import pandas as pd
import numpy as np
from util.Constants import *
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.utils import resample


class SentenceClassifier(object):

    def __init__(self):
        self.count_vec = CountVectorizer(decode_error="replace")
        self.tf_transformer = TfidfTransformer()

    def generate_initial_datasets(self, data):
        # CLASSES PROPORTION
        print(pd.DataFrame(data['label'].value_counts()))
        # SUBSET TRAIN AND OTHER DATASETS
        df_other = data.loc[data['label'] == 'o']
        df_train = data.loc[data['label'].isin(['t', 'f'])]
        return df_train, df_other


    def fit_transform_labels(self, df):
        global le
        # ENCODE LABEL VALUES
        le = LabelEncoder()
        y = le.fit_transform(df.label)
        return y


    def fit_samples(self, df):
        # EXTRACTING FEATURES FROM TEXT
        #count_vec = CountVectorizer()
        self.count_vec.fit(df.sentence)
        print('\nFit BoW vectors, df shape: (%s,%s)' % df.shape)
        # COMPUTE TF-IDF
        #tf_transformer = TfidfTransformer()
        # tf_transformer.fit(bow_vec)
        #print('Fit Tf-idf, df shape: (%s,%s)' % df.shape)

    # input: raw dataset of samples
    # returns a dataframe with extracted features from sentence
    def transform_samples(self, df):
        # EXTRACT FEATURES FROM OTHER
        bow_vec = self.count_vec.transform(df.sentence)
        print('Transform BoW, df shape: (%s,%s)' % bow_vec.shape)
        tf_idf = self.tf_transformer.fit_transform(bow_vec)
        print('Transform Tf-idf, df shape: (%s,%s)' % tf_idf.shape)
        return tf_idf

    def train_and_test_classifier(self, CLASS, property_name, iter, X, y):

        dump_dir = MODELS + "/" + CLASS + "/clf/" + str(iter) + "/"
        dump_file = dump_dir + property_name + ".sav"

        if os.path.exists(dump_dir) is False:
            os.makedirs(dump_dir)

        if os.path.exists(dump_file) is False:

            try:
                # train classifier
                clf = svm.SVC(probability=True, random_state=42)

                clf.fit(X, y)
                # save model
                print("saving model...")
                if os.path.exists(dump_dir) is False:
                    os.makedirs(dump_dir)
                pickle.dump(clf, open(dump_file, 'wb'))

                return clf
            except ValueError as er:
                print("ERROR trying to train classifier: {} {}".format(er.args, er.message))
                return None
        else:
            print("loading model...")
            clf = pickle.load(open(dump_file, 'rb'))
            return clf
        return None


    def get_new_true_samples(self, df_other, true_predicted_index):
        # creates a new dataframe only with new predicted samples
        new_true_samples = df_other.loc[df_other.index[true_predicted_index], :]
        new_true_samples.label = 't'

        pd.set_option('display.max_colwidth', -1)

        if new_true_samples.shape[0] > 5:
            print(new_true_samples[['sentence', 'value', 'label']].sample(5))
        else:
            print(new_true_samples[['sentence', 'value', 'label']])

        return new_true_samples


    def get_index_from_true_probabilities(self, THRESHOLD, o_pred_proba):
        # selects true predicted samples according to probability
        # false: 0
        # true: 1
        o_pred = [1 if pred[1] >= THRESHOLD else 0 for pred in o_pred_proba]

        # get indexes for each true predicted sample
        true_predicted_index = [i for i, pred in enumerate(o_pred) if pred is not 0]

        print("\nTrue predicted samples: %s" % len(true_predicted_index))

        return true_predicted_index


    def run(self, CLASS, property_name, THRESHOLD=0.9):

        # DEFINE PATH TO FILE
        input_file = TRAIN_CLF_DIR + "/" + CLASS + "/" + property_name + ".csv"

        # LOAD FILE
        data = pd.read_csv(input_file, names=['sentence', 'ner', 'entity', 'value', 'label'],
                           dtype={'sentence': str, 'entity': str, 'value': str, 'label': str},
                           converters={'ner': eval}, encoding='utf-8')

        # split datasets according to label
        df_train, df_other = self.generate_initial_datasets(data)

        # write output to datasets for training extractors
        output_dir = TRAIN_EXT_DIR + "/" + CLASS + "/"
        output_file = output_dir + property_name + ".csv"

        if os.path.exists(output_dir) is False:
            os.makedirs(output_dir)

        if os.path.exists(output_file):
            print("TRAIN EXTRACTION DATASET FOR %s ALREADY EXISTS" % property_name)
        else:
            print("\n==============================================================")

            print("downsample df_train")
            df_train_downsampled = self.downsampling(df_train)

            # split df_train in train and test sets
            print("Transform df_train")

            # fit BOW vectors and instantiate Tf-idf transformer
            self.fit_samples(data)

            X = self.transform_samples(df_train_downsampled)
            y = self.fit_transform_labels(df_train_downsampled)

            clf = svm.SVC(probability=True, random_state=42)

            clf.fit(X, y)

            if clf is not None:
                print("\nPredict samples on df_other")
                other_sent = self.transform_samples(df_other)

                o_pred_proba = clf.predict_proba(other_sent)

                # subset samples predicted true
                true_predicted_indexes = self.get_index_from_true_probabilities(THRESHOLD, o_pred_proba)

                new_true_samples = self.get_new_true_samples(df_other, true_predicted_indexes)

                if len(true_predicted_indexes) > 0:
                    # drop predicted positive samples from df_other
                    df_other = df_other.drop(df_other.index[true_predicted_indexes])

                if new_true_samples.shape[0] > 0:
                    # append newly predicted positive samples to df_train
                    df_train = df_train.append(new_true_samples)

            print("\nFINISHING ITERATION")
            print("df_other: {}".format(df_other.shape))
            print("df_train: {}".format(df_train.shape))
            print("\ndf_train labels frequency")
            print(pd.DataFrame(df_train['label'].value_counts()))
            print("\ndf_other labels frequency")
            print(pd.DataFrame(df_other['label'].value_counts()))
            print("==============================================================")

            df_other['label'] = 'f'
            df_train = df_train.append(df_other)

            print("downsample df_train")
            new_df_train_downsampled = self.downsampling(df_train)

            # retrain classifier
            print("Transform df_train")
            X_new = self.transform_samples(new_df_train_downsampled)
            y_new = self.fit_transform_labels(new_df_train_downsampled)

            if os.path.exists(MODELS + "/" + CLASS + "/clf-features/") is False:
                os.makedirs(MODELS + "/" + CLASS + "/clf-features/")

            pickle.dump(self.count_vec.vocabulary_,
                        open(MODELS + "/" + CLASS + "/clf-features/" + property_name + ".pkl", "wb"))

            # split df_train in train and test sets
            print("\nSplit df_train on train and test sets. Train classifier.")
            clf = self.train_and_test_classifier(CLASS, property_name, 1, X_new, y_new)

            unique_labels = df_train['label'].unique()

            if 't' not in unique_labels:
                print("Property missing true labels: {}".format(property_name))
            else:
                # save final df_train to file
                print("\nFINISHING CREATE TRAIN EXTRACTION DATASET FOR %s" % property_name)
                print("writing to file...")

                df_train = df_train[df_train['label'] == 't']

                data = []
                for index, row in df_train.iterrows():
                    data.append([row['sentence'], row['ner'], row['entity'], row['value'], row['label']])

                with open(output_file, 'w+') as csvfile:
                    spamwriter = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC)
                    spamwriter.writerows(data)
                    csvfile.close()
                # df_train.to_csv(output_file, mode='w', index=False, header=False, quoting=csv.QUOTE_NONNUMERIC)

        print("\nFINISHED GENERATING EXTRACTION TRAIN DATASETS")


    def saveProbaDistribution2File(self, CLASS, df_other, iter, o_pred_proba, property_name):
        df_sent = pd.DataFrame(df_other['sentence'], columns=['sentence'], dtype=np.object)
        df_prob = pd.DataFrame(o_pred_proba, columns=['f', 't'], dtype=np.str)

        df_concat = pd.DataFrame(data={"sentence": df_sent['sentence'].values, "f": df_prob['f'].values, "t": df_prob['t'].values})

        proba_dir = PROBA_DISTR_DIR + "/" + CLASS + "/"
        if os.path.exists(proba_dir) is False:
            os.makedirs(proba_dir)

        df_concat.to_csv(proba_dir + property_name + "_" + str(iter) + ".csv", header=True, index=False)


    def get_classifier(self, CLASS, property_name, last_iteration):
        dump_dir = MODELS + "/" + CLASS + "/"
        dump_file = dump_dir + property_name + "-sent-clf-" + last_iteration + ".sav"

        print("loading model for %s..." % property_name)
        clf = pickle.load(open(dump_file, 'rb'))

        return clf


    def downsampling(self, df):
        df_class_f = df[df.label == 'f']
        df_class_t = df[df.label == 't']

        majority = None
        minority = None
        if df_class_t.shape[0] > df_class_f.shape[0]:  # downsample class t
            majority = df_class_t
            minority = df_class_f
        elif df_class_f.shape[0] > df_class_t.shape[0]:  # downsample class f
            majority = df_class_f
            minority = df_class_t
        elif df_class_f.shape[0] == df_class_t.shape[0]:
            return df

        # Downsample majority class
        if majority is not None and minority is not None:
            print("Majority samples size: %s" % majority.shape[0])
            print("Minority samples size: %s" % minority.shape[0])

            majority_downsampled = resample(majority,
                                            replace=False,
                                            n_samples=minority.shape[0],
                                            random_state=42)

            print("After downsampling: %s" % majority_downsampled.shape[0])

            df_downsampled = pd.concat([majority_downsampled, minority])
            return df_downsampled

        return df

    def upsampling(self, df):
        df_class_f = df[df.label == 'f']
        df_class_t = df[df.label == 't']

        majority = None
        minority = None
        if df_class_t.shape[0] > df_class_f.shape[0]:  # downsample class t
            majority = df_class_t
            minority = df_class_f
        elif df_class_f.shape[0] > df_class_t.shape[0]:  # downsample class f
            majority = df_class_f
            minority = df_class_t
        elif df_class_f.shape[0] == df_class_t.shape[0]:
            return df

        if majority is not None and minority is not None:
            print("Majority samples size: %s" % majority.shape[0])
            print("Minority samples size: %s" % minority.shape[0])

            minority_upsampled = resample(minority,
                                            replace=True,
                                            n_samples=majority.shape[0],
                                            random_state=42)

            print("After upsampling: %s" % minority_upsampled.shape[0])

            df_upsampled = pd.concat([minority_upsampled, majority])
            return df_upsampled
        return df