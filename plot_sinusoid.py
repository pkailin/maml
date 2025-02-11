import csv
import numpy as np
import pickle
import random
import tensorflow as tf

from data_generator import DataGenerator
from maml import MAML
import matplotlib.pyplot as plt
import math
import numpy as np 

from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

'''
For MAML run with: python plot_sinusoid.py --datasource=sinusoid --logdir=logs/sine/ --metatrain_iterations=70000 --norm=None --update_batch_size=10
For baseline run with: python plot_sinusoid.py --datasource=sinusoid --logdir=logs/sine/baseline/ --pretrain_iterations=70000 --metatrain_iterations=0 --norm=None --update_batch_size=10

'''

## Dataset/method options
flags.DEFINE_string('datasource', 'sinusoid', 'sinusoid or omniglot or miniimagenet')
flags.DEFINE_integer('num_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')
# oracle means task id is input (only suitable for sinusoid)
flags.DEFINE_string('baseline', None, 'oracle, or None')

## Training options
flags.DEFINE_integer('pretrain_iterations', 0, 'number of pre-training iterations.')
flags.DEFINE_integer('metatrain_iterations', 15000, 'number of metatraining iterations.') # 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer('meta_batch_size', 25, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 5, 'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_float('update_lr', 1e-3, 'step size alpha for inner gradient update.') # 0.1 for omniglot
flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.')

## Model options
flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or None')
flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
flags.DEFINE_bool('conv', True, 'whether or not to use a convolutional network, only applicable in some cases')
flags.DEFINE_bool('max_pool', False, 'Whether or not to use max pooling rather than strided convolutions')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', '/tmp/data', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', True, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('test_set', False, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_integer('train_update_batch_size', -1, 'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1, 'value of inner gradient step step during training. (use if you want to test with a different value)') # 0.1 for omniglot

def test_and_plot(model, sess, K, data_generator, update_lr=0.01, num_updates=10):
    """
    Tests the trained MAML model on a single sinusoid task and plots the results.
    """
    task_outputbs = []

    # batch_x, batch_y, amp, phase = data_generator.generate(train=False) # for sinusoid 
    batch_x, batch_y, scale, shift = data_generator.generate(train=False) # for sigmoid 

    num_classes = 1
    inputa = batch_x[:, :num_classes*K, :] # first K points in inputa (training set)
    inputb = batch_x[:, num_classes*K:, :]
    labela = batch_y[:, :num_classes*K, :] # next 200 - K points in inputb (test set)
    labelb = batch_y[:, num_classes*K: , :]

    feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb, model.meta_lr: 0.0}
    
    result = sess.run(model.result, feed_dict) # result is [task_outputa, task_outputbs, task_lossa, task_lossesb]
    
    outputbs = np.squeeze(np.array(result[1]))
    print(outputbs.shape)

    values_labelb = np.squeeze(labelb) 
    values_inputb = np.squeeze(inputb)
    values_labela = np.squeeze(labela) 
    values_inputa = np.squeeze(inputa) 

    sorted_indices = np.argsort(values_inputb)
    inputb_sorted = values_inputb[sorted_indices]
    labelb_sorted = values_labelb[sorted_indices]

    for i in range(outputbs.shape[0]):

        outputb_sorted = outputbs[i][sorted_indices]

        plt.figure(figsize=(6, 4))
        plt.scatter(values_inputa, values_labela, marker='x', color='r', label='Train Data')
        plt.plot(inputb_sorted, labelb_sorted, 'g-', label='Ground Truth')
        # plt.plot(inputb_sorted, outputb_sorted, 'b--', label='MAML Prediction') # for MAML
        plt.plot(inputb_sorted, outputb_sorted, 'b--', label='Baseline Prediction') # for Baseline Model 
        
        # plt.title(f'Sinusoid Regression (Amplitude: {amp[0]:.2f}, Phase: {phase[0]:.2f})') # for sinusoid 
        plt.title(f'Sigmoid Regression (Scale: {scale:.2f}, Shift: {shift:.2f})') # for sigmoid 

        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        # plt.savefig('./plots/gradstep_MAML' + str(i) + '.png') # for MAML
        plt.savefig('./plots/gradstep_baseline' + str(i) + '.png') # for Baseline Model


def plot_sinusoid(): 
    FLAGS.meta_batch_size = 1 # number of unique amp and phase 

    # Create a data generator for sinusoid regression
    # data_generator = DataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size)
    data_generator = DataGenerator(200, FLAGS.meta_batch_size)
    dim_input = data_generator.dim_input
    dim_output = data_generator.dim_output # for MAML/pre-train 
    # dim_output = 3 # for oracle 

    test_num_updates = 10

    # Initialize the MAML model
    model = MAML(dim_input=dim_input, dim_output=dim_output, test_num_updates=test_num_updates)

    tf_data_load = False
    input_tensors = None

    model.construct_model(input_tensors=input_tensors, prefix='metatrain_')
    model.summ_op = tf.summary.merge_all()

    # Define TensorFlow session
    saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)
    sess = tf.InteractiveSession()

    if FLAGS.train_update_batch_size == -1:
        FLAGS.train_update_batch_size = FLAGS.update_batch_size
    if FLAGS.train_update_lr == -1:
        FLAGS.train_update_lr = FLAGS.update_lr

    # For sinusoid 
    # model_file = "logs/sine//cls_5.mbs_25.ubs_10.numstep1.updatelr0.001oraclenonorm\model69999" # for oracle 
    # model_file = "logs/sine//cls_5.mbs_25.ubs_10.numstep1.updatelr0.001nonorm\model69999" # for MAML
    # model_file = "logs/sine/baseline//cls_5.mbs_25.ubs_10.numstep1.updatelr0.001nonorm\model69999" # for baseline pretrained model 

    # For sigmoid 
    # model_file = "logs/sigmoid//cls_5.mbs_25.ubs_10.numstep1.updatelr0.001nonorm\model69999" # for MAML
    # model_file = "logs/sigmoid/baseline//cls_5.mbs_25.ubs_10.numstep1.updatelr0.001nonorm\model69999" # for baseline pretrained model

    # For sigmoid (4 hidden layers)
    # model_file = "logs/sigmoid_4hidd//cls_5.mbs_25.ubs_10.numstep1.updatelr0.001nonorm\model69999" # for MAML
    model_file = "logs/sigmoid_4hidd/baseline//cls_5.mbs_25.ubs_10.numstep1.updatelr0.001nonorm\model69999" # for baseline pretrained model
    
    print("Restoring model weights from " + model_file)
    saver.restore(sess, model_file)
    print("Model restored successfully!")

    # Call the function to test and plot adaptation
    test_and_plot(model, sess, 5, data_generator=data_generator) 

if __name__ == "__main__":
    plot_sinusoid()