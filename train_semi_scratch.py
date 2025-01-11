import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
import random
import numpy as np
import argparse
import json
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import KLDivergence, CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import BaseLogger

from model_semi import build_resnet_18, build_resnet_34, build_resnet_50, build_resnet_101, build_resnet_152
from data_util import generator_labeled_data_one_clip, generator_unlabeled_data_for_swap_learning, generator_test_data, get_data, get_classes, clean_data
from data_util import data_augmentation_labeled_one_clip, data_augmentation_unlabeled_swap_learning
# from data_util import EpochCheckpoint, ModelCheckpoint

from sklearn.model_selection import train_test_split
from call_back_semi import EpochCheckpoint, ModelCheckpoint_and_Reduce_LR, Switch_Models


def parse_args():
    parser = argparse.ArgumentParser(description='Training semi-supervised learning')
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

class Semi_Model(Model):
    def __init__(self, student_1, student_2, num_classes):
        super(Semi_Model, self).__init__()
        self.student_1 = student_1
        self.student_2 = student_2
        self.num_classes = num_classes

    def compile(self, optimizer, ce_loss, confidence=0.8):
        super(Semi_Model, self).compile()
        self.optimizer = optimizer
        self.ce_loss = ce_loss
        self.confidence = confidence
        self.accuracy_s1 = CategoricalAccuracy()
        self.accuracy_s2 = CategoricalAccuracy()
        self.accuracy_full = CategoricalAccuracy()
        self.val_s1_acc = CategoricalAccuracy()
        self.val_s2_acc = CategoricalAccuracy()
        self.val_accuracy = CategoricalAccuracy()

    @tf.function
    def train_step(self, data):
        # Unpack data
        data_labeled, data_unlabeled = data
        x_labeled, y = data_labeled[0], data_labeled[1]
        x_unlabeled_no_aug, x_unlabel_aug = data_unlabeled[0], data_unlabeled[1]

        # ---------------------------------- Training for student 1--------------------------------
        with tf.GradientTape() as s1_tape:
            # Forward pass of student 1
            z_labeled_s1 = self.student_1(x_labeled, training=True)

            # Compute loss for label data
            ce_loss_s1 = self.ce_loss(y, tf.nn.softmax(z_labeled_s1, axis=1))

            # loss_s1 = ce_loss_labeled_s1 + self.lamda * ce_loss_unlabeled_s1 + (1 - self.lamda) * kld_loss_unlabeld_s1
            loss_s1 = ce_loss_s1
            loss_s1 += sum(self.student_1.losses)

        # Compute gradients for student 1
        trainable_vars_1 = self.student_1.trainable_variables
        gradients_1 = s1_tape.gradient(loss_s1, trainable_vars_1)
        # Update weights for student 1
        self.optimizer.apply_gradients(zip(gradients_1, trainable_vars_1))

        z_unlabled_s1 = self.student_1(x_unlabeled_no_aug, training=False)
        logits = tf.stop_gradient(z_unlabled_s1)
        pseudo_labels = tf.nn.softmax(logits)
        pseudo_mask = tf.cast(tf.reduce_max(pseudo_labels, axis=1) >= self.confidence, dtype=tf.float32)
        pseudo_labels_one_hot = tf.one_hot(tf.argmax(logits, axis=1), self.num_classes)

        # ----------------------------------- Training for student 2-----------------------------------
        with tf.GradientTape() as s2_tape:
            # Forward pass of student 2
            z_unlabeled_s2 = self.student_2(x_unlabel_aug, training=True)

            # Compute loss for label data
            ce_loss_unlabeled_s2 = self.ce_loss(pseudo_labels_one_hot, tf.nn.softmax(z_unlabeled_s2, axis=1), sample_weight=pseudo_mask)

            loss_s2 = ce_loss_unlabeled_s2
            loss_s2 += sum(self.student_2.losses)

        # Compute gradients for student 2
        trainable_vars_2 = self.student_2.trainable_variables
        gradients_2 = s2_tape.gradient(loss_s2, trainable_vars_2)
        # Update weights for student 2
        self.optimizer.apply_gradients(zip(gradients_2, trainable_vars_2))

        # Update the metrics configured in `compile()
        self.accuracy_full.update_state(y, tf.stop_gradient(tf.nn.softmax(z_labeled_s1, axis=1)))

        # Return a dict of performance
        results = {}
        results.update({"loss_S1": loss_s1, "loss_S2": loss_s2, "accuracy_full": self.accuracy_full.result()})
        return results

    @tf.function
    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction_s1 = self.student_1(x, training=False)
        y_prediction_s2 = self.student_2(x, training=False)

        # Calculate the loss for student 1
        student1_loss = self.ce_loss(y, tf.nn.softmax(y_prediction_s1, axis=1))
        loss_s1 = student1_loss + sum(self.student_1.losses)

        # Calculate the loss for student 2
        student2_loss = self.ce_loss(y, tf.nn.softmax(y_prediction_s2, axis=1))
        loss_s2 = student2_loss + sum(self.student_2.losses)

        y_prediction_full = y_prediction_s1 + y_prediction_s2

        # Update the metrics.
        self.val_s1_acc.update_state(y, tf.nn.softmax(y_prediction_s1, axis=1))
        self.val_s2_acc.update_state(y, tf.nn.softmax(y_prediction_s2, axis=1))
        self.val_accuracy.update_state(y, tf.nn.softmax(y_prediction_full, axis=1))

        # Return a dict of performance
        results = {}
        results.update({"loss_S1": loss_s1, "loss_S2": loss_s2, "s1_accuracy": self.val_s1_acc.result(),
                        "s2_accuracy": self.val_s2_acc.result(),
                        "accuracy": self.val_accuracy.result()})
        return results

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


def build_callbacks_new(save_path, every, startAt, train_dataset, monitor='val_accuracy', mode='max', confidence=0.7,
                    lr_init=0.01, batch_size=16, epochs=200, warmup_epoch=10, num_epoch_switch=1):
    earlyStopping = EarlyStopping(monitor=monitor, patience=40, verbose=1)

    checkpoint_path = os.path.join(save_path, 'checkpoints')
    epoch_checkpoint = EpochCheckpoint(outputPath=checkpoint_path, every=every, startAt=startAt)

    jsonName = 'log_results.json'
    jsonPath = os.path.join(save_path, "output")
    Checkpoint_LR = ModelCheckpoint_and_Reduce_LR(folderpath=save_path, jsonPath=jsonPath, jsonName=jsonName, startAt=startAt,
                                               lr_init=lr_init, monitor=monitor, mode=mode, factor=0.1, patience=20,
                                               verbose=1)

    switch_models = Switch_Models(num_epoch_switch=num_epoch_switch, startAt=startAt, verbose=1)
    cb = [earlyStopping, epoch_checkpoint, Checkpoint_LR, switch_models]

    return cb


def train_semi(train_labeled_dataset, train_unlabeled_dataset, test_dataset, model_name, classes_list, input_shape, lr_init,
               batch_size=16, reg_factor=5e-4, drop_rate=0.5, epochs=200, confidence=0.7,
               start_epoch=1, save_path='save_model', every=1, num_epoch_switch=1):

    # Prepare data for training phase
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    ds_labeled = tf.data.Dataset.from_generator(generator_labeled_data_one_clip,
                                                (tf.float32, tf.float32),
                                                (tf.TensorShape(input_shape), tf.TensorShape([len(classes_list)])),
                                                args=[train_labeled_dataset, classes_list, input_shape[0], input_shape[1]])
    ds_labeled = ds_labeled.map(data_augmentation_labeled_one_clip, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE).batch(batch_size)
    # ds_labeled = ds_labeled.batch(batch_size//3).prefetch(tf.data.experimental.AUTOTUNE)

    ds_unlabeled = tf.data.Dataset.from_generator(generator_unlabeled_data_for_swap_learning,
                                                  (tf.float32, tf.float32),
                                                (tf.TensorShape(input_shape), tf.TensorShape(input_shape)),
                                                args=[train_unlabeled_dataset, classes_list, input_shape[0], input_shape[1]])
    ds_unlabeled = ds_unlabeled.map(data_augmentation_unlabeled_swap_learning, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE).batch(batch_size*3)
    # ds_unlabeled = ds_unlabeled.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    ds_train = tf.data.Dataset.zip((ds_labeled, ds_unlabeled))


    ds_test = tf.data.Dataset.from_generator(generator_test_data, (tf.float32, tf.float32),
                                               (tf.TensorShape(input_shape), tf.TensorShape([len(classes_list)])),
                                               args=[test_dataset, classes_list, input_shape[0], input_shape[1]])
    ds_test = ds_test.prefetch(AUTOTUNE).batch(batch_size)

    # Build model
    student_1 = build_model(model_name, input_shape, len(classes_list), reg_factor=reg_factor, activation=None, drop_rate=drop_rate)
    student_2 = build_model(model_name, input_shape, len(classes_list), reg_factor=reg_factor, activation=None, drop_rate=drop_rate)

    # student_1.summary(line_length=150)

    # Load weight for student model
    if start_epoch > 1:
        print('--------------------------START LOAD WEIGHT FROM CURRENT EPOCH---------------------------------')
        path1 = os.path.join(save_path, 'checkpoints', 's1_epoch_' + str(start_epoch) + '.h5')
        student_1.load_weights(path1)
        path2 = os.path.join(save_path, 'checkpoints', 's2_epoch_' + str(start_epoch) + '.h5')
        student_2.load_weights(path2)
        print('--------------------------LOAD WEIGHT COMPLETED---------------------------------')


    # Build callbacks
    callback = build_callbacks_new(save_path, every=every, startAt=start_epoch, train_dataset=train_unlabeled_dataset,
                               monitor='val_accuracy', mode='max', lr_init=lr_init, batch_size=batch_size, confidence=confidence,
                               epochs=epochs, warmup_epoch=10, num_epoch_switch=num_epoch_switch)

    # Build optimizer and loss function
    optimizer = SGD(learning_rate=lr_init, momentum=0.9, nesterov=True)
    ce_loss = CategoricalCrossentropy(label_smoothing=0.1)

    # Build model
    Model = Semi_Model(student_1, student_2, len(classes_list))
    Model.compile(
        optimizer,
        ce_loss,
        confidence=confidence
    )

    # Training
    Model.fit(ds_train, epochs=epochs - start_epoch + 1, verbose=1,
              steps_per_epoch=len(train_unlabeled_dataset) // (batch_size),
              validation_data=ds_test,
              validation_steps=len(test_dataset) // (batch_size),
              callbacks=callback
              )

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
    num_epoch_switch = args.switch
    drop_rate = args.drop_rate
    every = 1
    percent = args.percent
    confidence = args.confidence

    # Read dataset
    if percent==1:
        save_path = './save_model/swap_semi_1percent/'
        # train_labeled_dataset = get_data('mini_Kinetics_100_train_labeled_1percent.csv')
        # train_unlabeled_dataset = get_data('mini_Kinetics_100_train_unlabeled_1percent.csv')
        train_labeled_dataset = get_data('train_labeled_1percent.csv')
        train_unlabeled_dataset = get_data('train_unlabeled_1percent.csv')
    elif percent==5:
        save_path = './save_model/swap_semi_5percent/'
        # train_labeled_dataset = get_data('mini_Kinetics_100_train_labeled_5percent.csv')
        # train_unlabeled_dataset = get_data('mini_Kinetics_100_train_unlabeled_5percent.csv')
        train_labeled_dataset = get_data('train_labeled_5percent.csv')
        train_unlabeled_dataset = get_data('train_unlabeled_5percent.csv')
    elif percent==10:
        save_path = './save_model/swap_semi_10percent/'
        # train_labeled_dataset = get_data('mini_Kinetics_100_train_labeled_10percent.csv')
        # train_unlabeled_dataset = get_data('mini_Kinetics_100_train_unlabeled_10percent.csv')
        train_labeled_dataset = get_data('train_labeled_10percent.csv')
        train_unlabeled_dataset = get_data('train_unlabeled_10percent.csv')
    elif percent==20:
        save_path = './save_model/swap_semi_20percent/'
        train_labeled_dataset = get_data('train_labeled_20percent.csv')
        train_unlabeled_dataset = get_data('train_unlabeled_20percent.csv')
    else:
        save_path = './save_model/swap_semi_50percent/'
        train_labeled_dataset = get_data('train_labeled_50percent.csv')
        train_unlabeled_dataset = get_data('train_unlabeled_50percent.csv')

    # test_dataset = get_data('mini_Kinetics_100_test.csv')
    test_dataset = get_data('test.csv')
    classes_list = get_classes(train_labeled_dataset)
    print('Number of classes:', len(classes_list))
    print('Train labeled set:', len(train_labeled_dataset))
    print('Train unlabeled set:', len(train_unlabeled_dataset))
    print('Test set:', len(test_dataset))

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
    f.write('numbers of epochs: ' + str(epochs) + '\n')
    f.write('lr init: ' + str(lr_init) + '\n')
    f.write('num_epoch_switch: ' + str(num_epoch_switch) + '\n')
    f.write('start epoch: ' + str(start_epoch) + '\n')
    f.write('Drop rate: ' + str(drop_rate) + '\n')
    f.write('Percent: ' + str(percent) + '\n')
    f.write('confidence: ' + str(confidence) + '\n')
    f.close()

    train_labeled_dataset = clean_data(train_labeled_dataset, input_shape[0] * 2 + 2, classes=classes_list, MAX_FRAMES=3000)
    train_unlabeled_dataset = clean_data(train_unlabeled_dataset, input_shape[0] * 2 + 2, classes=None, MAX_FRAMES=3000)
    test_dataset = clean_data(test_dataset, input_shape[0] + 1, classes=classes_list, MAX_FRAMES=3000)
    random.shuffle(test_dataset)
    random.shuffle(train_labeled_dataset)
    random.shuffle(train_unlabeled_dataset)
    print('Train labeled set after clean:', len(train_labeled_dataset))
    print('Train unlabled set after clean:', len(train_unlabeled_dataset))
    print('Test set after clean:', len(test_dataset))


    # _, test_dataset_small = train_test_split(test_dataset, test_size=0.1)
    # test_dataset = test_dataset_small
    # print('Sub Test set after clean:', len(test_dataset))

    # train_labeled_dataset = train_labeled_dataset[:1000]
    # train_unlabeled_dataset = train_unlabeled_dataset[:1000]
    # test_dataset = test_dataset[:100]


    # --------------------------------------Continuous training ----------------------------------------
    train_semi(train_labeled_dataset, train_unlabeled_dataset, test_dataset, model_name, classes_list, input_shape, lr_init,
               batch_size=batch_size, reg_factor=reg_factor, drop_rate=drop_rate, confidence=confidence,
               epochs=epochs, start_epoch=start_epoch, save_path=save_path, every=every, num_epoch_switch=num_epoch_switch)


if __name__ == '__main__':
    print(tf.__version__)
    main()

