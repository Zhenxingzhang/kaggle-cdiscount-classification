CLASSES_COUNT = 5270
INCEPTION_CLASSES_COUNT = 2048
INCEPTION_OUTPUT_FIELD = 'inception_output'
LABEL_ONE_HOT_FIELD = 'label_one_hot'
IMAGE_RAW_FIELD = 'img_raw'
INCEPTION_INPUT_STRING_TENSOR = 'DecodeJpeg/contents:0'
INCEPTION_INPUT_ARRAY_TENSOR = 'DecodeJpeg:0'
INCEPTION_OUTPUT_TENSOR = 'pool_3:0'
OUTPUT_NODE_NAME = 'output_node'
OUTPUT_TENSOR_NAME = OUTPUT_NODE_NAME + ':0'
HEAD_INPUT_NODE_NAME = 'x'
HEAD_INPUT_TENSOR_NAME = HEAD_INPUT_NODE_NAME + ':0'

DEV_SET_SIZE = 3
TRAIN_SAMPLE_SIZE = 3

# name of the models being referenced by all other scripts
CURRENT_MODEL_NAME = 'nn_2_2048_1024_5000'
# sets up number of layers and number of units in each layer for
# the "head" dense neural network stacked on top of the Inception
# pre-trained models.
HEAD_MODEL_LAYERS = [INCEPTION_CLASSES_COUNT, 1024, CLASSES_COUNT]