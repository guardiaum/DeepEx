import os, sys

# SET CONSTANT VARIABLES
INFOBOX_CLASS = 'us_county'
PROPERTY_NAME = 'state'

ROOT = os.path.dirname(os.path.dirname(__file__))

MODELS = ROOT + "/models"
DATASETS = ROOT + "/datasets"
EXPERIMENTS_DIR = ROOT + "/experiments"
TRAIN_CLF_DIR = DATASETS + "/train_clf"
TRAIN_EXT_DIR = DATASETS + "/train_extractor"
VALIDATION_ARTICLES_DIR = DATASETS + "/articles-validation"
TEST_ARTICLES_DIR = DATASETS + "/articles-test"
FULL_BASE_DIR = DATASETS +"/full-base"
PROBA_DISTR_DIR = DATASETS + "/proba_distributions"
WIKIPEDIA_CHUNKS_DIR = ROOT + "/infoboxes-extractor/wikipedia-dump/wikipedia-xml-chunks/"

ITERATION_NUMBER = 1