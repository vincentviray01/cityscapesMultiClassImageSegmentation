from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
import argparse

from functions import *

# LIMIT GPU USAGE - OPTIONAL
# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# for device in gpu_devices:
    # tf.config.experimental.set_memory_growth(device, True)

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#   try:
#     tf.config.set_logical_device_configuration(
#         gpus[0],
#         [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
#     logical_gpus = tf.config.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Virtual devices must be set before GPUs have been initialized
#     print(e)


print("Is GPU available: " + str(tf.test.is_gpu_available()))

# VARIABLES
trainingImagesPath = "./cityscapes_data/train"
testingImagesPath = "./cityscapes_data/val"

parser = argparse.ArgumentParser(description="Train U-NET")
parser.add_argument('-i', '--img_size', type=int, metavar='', required=True, help="The resolution of the training and generated images")
parser.add_argument('-b', '--batch_size', type=int, metavar='', required=True, help="Batch size when training")
parser.add_argument('-e', '--epochs', type=int, required=True, help="The number of epochs to train for")
parser.add_argument('-n', '--num_classes', type=int, metavar='', required=True, help="The number of unique labels a pixel can take from")
group = parser.add_mutually_exclusive_group()
group.add_argument('-q', '--quiet', action='store_true', help='print quiet')
group.add_argument('-v', '--verbose', action='store_true', help ='print verbose')
args = parser.parse_args()

image_size = args.img_size # originally 128
batch_size = args.batch_size
epochs = args.epochs
num_classes = args.num_classes # originally 10
quiet = args.quiet == True
verbose = not quiet

adamLearningRate = 0.1

trainingImagesFilePaths = getImages(trainingImagesPath)
testingImagesFilePaths = getImages(testingImagesPath)

trainingSetSize = len(trainingImagesFilePaths)
testingSetSize = len(testingImagesFilePaths)

trainingImages = imagePathsToNumpyArrays(trainingImagesFilePaths)
testingImages = imagePathsToNumpyArrays(testingImagesFilePaths)

trainingImagesWithoutMask, trainingImagesWithMask = splitImageAndMask(trainingImages)
testingImagesWithoutMask, testingImagesWithMask = splitImageAndMask(testingImages)

if verbose:
    showSampleImagesAndMasks(testingImagesWithoutMask, testingImagesWithMask)

kMeansClusterCenters, trainingImagesWithMaskLabels, testingImagesWithMaskLabels = getNewLabelsForTrainAndTestMasks(trainingImagesWithMask, testingImagesWithMask, num_classes=num_classes)

trainingImagesWithMaskLabelsCategorized = to_categorical(trainingImagesWithMaskLabels, num_classes=num_classes)
testingImagesWithMaskLabelsCategorized = to_categorical(testingImagesWithMaskLabels, num_classes=num_classes)

# GET MODEL AND COMPILE
input_shape = (image_size, image_size, 3)
model = getModel(inputShape=input_shape, verbose=verbose, classes=num_classes)
model.compile(optimizer=Adam(learning_rate=adamLearningRate), loss=sparse_categorical_crossentropy, metrics=['accuracy'], sample_weight_mode='temporal')

# PLOT MODEL PNG
if verbose:
    plotModelPNG(model)

sample_weights = getSampleWeights(trainingImagesWithMaskLabels, num_classes, trainingSetSize, image_size)
callbacksList = getCallbacks()

model.load_weights("savedModelUpdated4")
# model.fit(trainingImagesWithoutMask,np.asarray(trainingImagesWithMaskLabels).reshape(-1, image_size**2),batch_size=batch_size,epochs=epochs,callbacks=callbacksList, verbose=1, sample_weight=sample_weights)
model.save_weights('savedModelUpdated4')

# MAKE PREDICTIONS
if verbose:
    showPredictionsAndTruths(model, kMeansClusterCenters, testingImagesWithoutMask, testingImagesWithMaskLabels, num_classes)

print("FINISHED")