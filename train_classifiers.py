from util.Constants import *
from modules.SentenceClassifier import SentenceClassifier


def run(class_):
    # select class properties
    class_path = TRAIN_CLF_DIR + "/" + class_
    properties = [f.replace(".csv", "") for f in os.listdir(class_path) if
                  os.path.isfile(os.path.join(class_path, f))]

    print(properties)

    print("\nSTART GENERATING EXTRACTION TRAIN DATASETS")
    for property_name in properties:
        print("\nSTART GENRATING EXTRACTION TRAIN DATASET FOR %s" % property_name)

        bt_clf = SentenceClassifier()

        bt_clf.run(class_, property_name)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        CLASS = sys.argv[1]
        run(CLASS)
        print("FINISHED TRAINING CLASSIFIERS AND FILTERING DATASETS FOR TRAINING EXTRACTORS")
    else:
        print("PLEASE INFORM CLASS")