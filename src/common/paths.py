import os

JPEG_EXT = '.jpg'
DATA_ROOT = '/data/'
APP_DATA_ROOT = '/app/data/'
# TRAIN_DIR = os.path.join(DATA_ROOT, 'train')
# TEST_DIR = os.path.join(DATA_ROOT, 'test')
TRAIN_TF_RECORDS = os.path.join(DATA_ROOT, 'data/train_example.tfrecords')
TEST_TF_RECORDS = os.path.join(DATA_ROOT, 'data/test.tfrecords')
# LABELS = os.path.join(DATA_ROOT, 'train', 'labels.csv')
CATEGORIES = os.path.join(APP_DATA_ROOT, 'category_names.csv')
IMAGENET_GRAPH_DEF = os.path.join(DATA_ROOT, 'frozen/inception/classify_image_graph_def.pb')
INCEPTION_MODEL_DIR = os.path.join(DATA_ROOT, "frozen/inception/")
TEST_PREDICTIONS = os.path.join(DATA_ROOT, 'outputs/predictions.csv')
METRICS_DIR = os.path.join(DATA_ROOT, 'metrics')
# TRAIN_CONFUSION = os.path.join(METRICS_DIR, 'training_confusion.csv')
FROZEN_MODELS_DIR = os.path.join(DATA_ROOT, 'frozen')
CHECKPOINTS_DIR = os.path.join(DATA_ROOT, 'checkpoints')
GRAPHS_DIR = os.path.join(DATA_ROOT, 'graphs')
SUMMARY_DIR = os.path.join(DATA_ROOT, 'summary')
