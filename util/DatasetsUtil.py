import unicodecsv as csv
import os
import io


class DatasetsUtil(object):

    @staticmethod
    def readDataset4Attribute(datasets_dir, attribute):

        examples = []
        datasetPath = os.path.join(datasets_dir, attribute + '.csv')

        with io.open(datasetPath, "rb") as datasetCodec:
            reader = csv.reader(datasetCodec, quoting=csv.QUOTE_ALL)
            data = {i: v for (i, v) in enumerate(reader)}
            for i in data:
                examples.append([data[i][0], data[i][1], data[i][2], data[i][3]])
        return examples

    @staticmethod
    def selectDatasetFields(examples, indexes):
        filteredFields = []

        for example in examples:
            temp = []
            for i in indexes:
                temp.append(example[i])
            filteredFields.append(temp)

        return filteredFields