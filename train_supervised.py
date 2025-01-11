import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
import json
from sklearn.model_selection import train_test_split
import argparse
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD
# from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.callbacks import BaseLogger
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.losses import KLDivergence, CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from model_semi import build_resnet_18, build_resnet_34, build_resnet_50, build_resnet_101, build_resnet_152
from data_util import get_data, get_classes, clean_data, generator_labeled_data_one_clip, data_augmentation_labeled_one_clip, generator_test_data

def parse_args():
    parser = argparse.ArgumentParser(description='Semi-supervised learning')
    parser.add_argument('--model', type=str, default='res18', help='res18/res34/res50/res101/res152')
    parser.add_argument('--clip_len', type=int, default=16, help='clip length')
    parser.add_argument('--crop_size', type=int, default=224, help='crop size')
    parser.add_argument('--dataset', type=str, default='ucf101', help='ucf101/hmdb51/kinetics100/kinetics400/minisomething')
    parser.add_argument('--percent', type=int, default=5, help='percent of labeled data')
    parser.add_argument('--switch', type=int, default=10, help='swap value')
    parser.add_argument('--gpu', type=str, default='0', help='GPU id')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--confidence', type=float, default=0.8, help='confidence factor')
    parser.add_argument('--drop_rate', type=float, default=0.5, help='drop rate')
    parser.add_argument('--reg_factor', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--epochs', type=int, default=500, help='number of total epochs to run')
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--start_epoch', type=int, default=1, help='manual epoch number (useful on restarts)')
    args = parser.parse_args()
    return args


def build_model(model_name, input_shape, num_classes, reg_factor=1e-4, activation='softmax', drop_rate=None):
    if model_name == 'res18':
        model = build_resnet_18(input_shape, num_classes, reg_factor, activation=activation, drop_rate=drop_rate)
    elif model_name == 'res34':
        model = build_resnet_34(input_shape, num_classes, reg_factor, activation=activation, drop_rate=drop_rate)
    elif model_name == 'res50':
        model = build_resnet_50(input_shape, num_classes, reg_factor, activation=activation, drop_rate=drop_rate)
    elif model_name == 'res101':
        model = build_resnet_101(input_shape, num_classes, reg_factor, activation=activation, drop_rate=drop_rate)
    else:
        model = build_resnet_152(input_shape, num_classes, reg_factor, activation=activation, drop_rate=drop_rate)
    return model

def build_callbacks(save_path, filepath, every, startAt, monitor, mode, has_teacher=True):
    earlyStopping = EarlyStopping(monitor=monitor, patience=30, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=10, verbose=1, factor=0.1, min_lr=1e-8)

    jsonName = 'normal.json'
    jsonPath = os.path.join(save_path, "output")
    checkpoint_path = os.path.join(save_path, 'checkpoints')

    saveLog = SaveLog(jsonPath=jsonPath, jsonName=jsonName, startAt=startAt, verbose=1)
    epoch_checkpoint = EpochCheckpoint(checkpoint_path, every=every, startAt=startAt)
    model_checkpoint = ModelCheckpoint(filepath, verbose=1, monitor=monitor, mode=mode, save_best_only=True, save_weights_only=True)
    # cb = [earlyStopping, reduce_lr, model_checkpoint]
    cb = [model_checkpoint, epoch_checkpoint, reduce_lr, saveLog, earlyStopping]

    return cb

def training_supervised(train_dataset, test_dataset, model_name, input_shape, classes_list, lr_init, weight_model_path,
                        start_epoch=1, reg_factor=1e-4, save_path='save_model', epochs=200, batch_size=16):
    # Prepare data for training and testing
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    data_train = tf.data.Dataset.from_generator(generator_labeled_data_one_clip, (tf.float32, tf.float32),
                                                (tf.TensorShape(input_shape), tf.TensorShape([len(classes_list)])),
                                                args=[train_dataset, classes_list, input_shape[0], input_shape[1]])

    data_test = tf.data.Dataset.from_generator(generator_test_data, (tf.float32, tf.float32),
                                               (tf.TensorShape(input_shape), tf.TensorShape([len(classes_list)])),
                                               args=[test_dataset, classes_list, input_shape[0], input_shape[1]])

    data_train = data_train.map(data_augmentation_labeled_one_clip, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(buffer_size=AUTOTUNE)
    data_test = data_test.batch(batch_size).prefetch(buffer_size=AUTOTUNE)

    # Optimization and Loss
    sgd = SGD(learning_rate=lr_init, momentum=0.9, nesterov=True)

    # Callbacks
    cb = build_callbacks(save_path=save_path, filepath=weight_model_path, every=1, startAt=start_epoch,
                               monitor='val_accuracy', mode='max', has_teacher=False)

    # Build model
    model = build_model(model_name, input_shape, len(classes_list), reg_factor=reg_factor, activation='softmax')
    if start_epoch > 1:
        path = os.path.join(save_path, 'checkpoints', 'epoch_' + str(start_epoch) + '.h5')
        print('----------------------------Load model weights-------------------------------')
        model.load_weights(path)

    model.summary(line_length=150)

    loss = CategoricalCrossentropy()
    model.compile(optimizer=sgd, loss=loss, metrics=['accuracy'])
    model.fit(data_train, epochs=epochs - start_epoch + 1, verbose=1, steps_per_epoch= len(train_dataset)//batch_size,
                   validation_data=data_test, validation_steps=len(test_dataset)//batch_size,
                   callbacks=cb)

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)  # Choose GPU for training

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=config)

    input_shape = (args.clip_len, args.crop_size, args.crop_size, 3)
    model_name = args.model
    reg_factor = args.reg_factor
    batch_size = args.batch_size
    epochs = args.epochs
    lr_init = args.lr
    start_epoch = args.start_epoch
    percent = args.percent

    # Read dataset
    if percent == 1:
        save_path = './save_model/Normal_1percent_with_aug_TSM/'
        train_dataset = get_data('train_labeled_1percent.csv')
    elif percent == 5:
        save_path = './save_model/Normal_5percent_with_aug_TSM/'
        train_dataset = get_data('train_labeled_5percent.csv')
    elif percent == 10:
        save_path = './save_model/Normal_10percent_with_aug_TSM/'
        train_dataset = get_data('train_labeled_10percent.csv')
    elif percent == 20:
        save_path = './save_model/Normal_20percent_with_aug_TSM/'
        train_dataset = get_data('train_labeled_20percent.csv')
    elif percent == 40:
        save_path = './save_model/Normal_40percent_with_aug_TSM/'
        train_dataset = get_data('train_labeled_40percent.csv')
    elif percent == 50:
        save_path = './save_model/Normal_50percent_with_aug_TSM/'
        train_dataset = get_data('train_labeled_50percent.csv')
    else:
        save_path = './save_model/Normal_60percent_with_aug_TSM/'
        train_dataset = get_data('train_labeled_60percent.csv')

    # Create folders for callback
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(os.path.join(save_path, "output")):
        os.mkdir(os.path.join(save_path, "output"))
    if not os.path.exists(os.path.join(save_path, "checkpoints")):
        os.mkdir(os.path.join(save_path, "checkpoints"))

    # Write all config to file
    f = open(os.path.join(save_path, 'config.txt'), "w")
    f.write('input shape: ' + str(input_shape) + '\n')
    f.write('model name: ' + model_name + '\n')
    f.write('reg factor: ' + str(reg_factor) + '\n')
    f.write('batch size: ' + str(batch_size) + '\n')
    f.write('batch size: ' + str(batch_size) + '\n')
    f.write('numbers of epochs: ' + str(epochs) + '\n')
    f.write('lr init: ' + str(lr_init) + '\n')
    f.write('start epoch: ' + str(start_epoch) + '\n')
    f.close()

    # Read dataset
    test_dataset = get_data('test.csv')
    classes_list = get_classes(train_dataset)
    print('Number of classes:', len(classes_list))
    print('Train set:', len(train_dataset))
    print('Test set:', len(test_dataset))

    weight_model_path = os.path.join(save_path, "best_" + model_name + "_percent_" + str(percent) + ".h5")

    train_dataset = clean_data(train_dataset, args.clip_len + 1, classes=classes_list, MAX_FRAMES=3000)
    test_dataset = clean_data(test_dataset, args.clip_len + 1, classes=classes_list, MAX_FRAMES=3000)
    print('Train set after clean:', len(train_dataset))
    print('Test set after clean:', len(test_dataset))

    _, test_dataset_small = train_test_split(test_dataset, test_size=0.2, random_state=2019)
    test_dataset = test_dataset_small
    print('Sub Test set after clean:', len(test_dataset))

    # Training
    training_supervised(train_dataset, test_dataset, model_name, input_shape, classes_list, lr_init, weight_model_path,
                        start_epoch=start_epoch, reg_factor=reg_factor, save_path=save_path, epochs=epochs, batch_size=batch_size)

class SaveLog(BaseLogger):
    def __init__(self, jsonPath=None, jsonName=None, startAt=0, verbose=0):
        super(SaveLog, self).__init__()
        self.jsonPath = jsonPath
        self.jsonName = jsonName
        self.jsonfile = os.path.join(self.jsonPath, self.jsonName)
        self.startAt = startAt
        self.verbose = verbose

    def on_train_begin(self, logs={}):
        # initialize the history dictionary
        self.H = {}

        # if the JSON history path exists, load the training history
        if self.jsonfile is not None:
            if os.path.exists(self.jsonfile):
                self.H = json.loads(open(self.jsonfile).read())

    def on_epoch_end(self, epoch, logs={}):
        # loop over the logs and update the loss, accuracy, etc.
        # for the entire training process
        for (k, v) in logs.items():
            l = self.H.get(k, [])
            l.append(float(v))
            self.H[k] = l

        # check to see if the training history should be serialized
        # to file
        if self.jsonfile is not None:
            f = open(self.jsonfile, "w")
            f.write(json.dumps(self.H))
            f.close()

class EpochCheckpoint(Callback):
    def __init__(self, outputPath, every=5, startAt=1):
        # call the parent constructor
        super(Callback, self).__init__()

        # store the base output path for the model, the number of
        # epochs that must pass before the model is serialized to
        # disk and the current epoch value
        self.outputPath = outputPath
        self.every = every
        self.intEpoch = startAt

    def on_epoch_end(self, epoch, logs={}):
        # check to see if the model should be serialized to disk
        if (self.intEpoch) % self.every == 0:
            # Save current model weight
            p = os.path.sep.join([self.outputPath, "epoch_{}.h5".format(self.intEpoch)])
            self.model.save_weights(p, overwrite=True)

            # Delete old model weight
            old_p = os.path.sep.join([self.outputPath, "epoch_{}.h5".format(self.intEpoch - self.every)])
            if os.path.exists(old_p):
                os.remove(old_p)

        # increment the internal epoch counter
        self.intEpoch += 1

if __name__ == '__main__':
    print(tf.__version__)
    main()
