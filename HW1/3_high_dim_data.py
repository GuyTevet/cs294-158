import os
import pickle
import matplotlib.pyplot as plt
from copy import copy
import numpy as np
import logging
import tensorflow as tf
import shutil
from tqdm import tqdm
import nets
import argparse

logger = logging.getLogger('log')
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

num_epochs = 25


def get_data():
    pkl_name = './mnist-hw1.pkl'

    if not os.path.exists((pkl_name)):
        raise FileNotFoundError(
            'Please download the file mnist-hw1.pkl from https://drive.google.com/file/d/1hm077GxmIBP-foHxiPtTxSNy371yowk2/view and put it in this directory.')

    with open(pkl_name, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')

    return data

def show_sample(data, idx, key='train'):
    plt.imshow(data[key][idx] * 64) # uint2 -> uint8

def get_shuffled_train(data):
    train = copy(data['train'])
    np.random.shuffle(train) # in place shuffle
    return train

def sample_and_save(sess, model, data, save_path, label=''):

    # sample
    num_images = 16
    logger.info('Sampling {} images from the model'.format(num_images))
    samples = model.sample(sess, num_images)
    # samples = np.zeros([num_images, 28, 28, 3])

    # save samples
    plt.figure(figsize=(12, 12))
    plt.suptitle(label, fontsize=32)
    for k in range(4):
        for l in range(4):
            plt.subplot(4, 4, 4 * k + l + 1)
            plt.imshow(samples[4 * k + l] * 64)
            plt.axis('off')
            plt.grid(False)
    plt.savefig(save_path)

def train(data, model, config):

    num_samples = data['train'].shape[0]
    num_batches = num_samples // model.batch_size

    with model.sess as sess:

        # tensorboard
        log_dir = os.path.join(config.log_path, config.experiment_name)
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        writer = tf.summary.FileWriter(log_dir, graph=sess.graph)

        # saver
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=999999)

        # init
        tf.global_variables_initializer().run()
        step = 0

        sample_path = os.path.join(experiment_path, 'init.png')
        sample_and_save(sess, model, data, sample_path, 'On init')

        for epoch in range(num_epochs):
            logger.info('Epoch {} out of {}'.format(epoch + 1, num_epochs))
            logger.info('shuffling samples')
            epoch_samples = get_shuffled_train(data)

            for i in tqdm(range(num_batches)):

                batch = epoch_samples[i * model.batch_size : (i+1) * model.batch_size]

                if writer is None:
                    model.train_on_batch(sess, batch, summarize=False)
                else:
                    loss, summary, _ = model.train_on_batch(sess, batch, summarize=True)
                    writer.add_summary(summary, step)

                step += 1
            sample_path = os.path.join(experiment_path, 'epoch.png'.format(epoch))
            sample_and_save(sess, model, data, sample_path, 'Epoch {}'.format(epoch))

        # save checkpoint
        logger.info('#' * 20)
        logger.info('saving model...')
        save_file = os.path.join(log_dir, '{}_epoch_{}'.format(config.experiment_name, num_epochs))
        saver.save(sess, save_file)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="SeqGAN Train for real text datasets")

    ######################################################################################
    #  General
    ######################################################################################
    parser.add_argument('experiment_name', type=str, help='experiment name')
    parser.add_argument('--log_path', type=str, default='./log/',  help='tensorboard log path')
    parser.add_argument('--num_epochs', type=int, default=25,  help='num of epochs')
    parser.add_argument('--gpu_inst', type=str, default='', help='choose GPU instance. empty string == run on CPU []')

    config = parser.parse_args()
    experiment_path = os.path.join(config.log_path,config.experiment_name)

    #choose GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_inst

    # #check valid name
    # if os.path.isdir(experiment_path):
    #     raise NameError('experiment_name [{}] already exists - choose another one!'.format(config.experiment_name))

    # print config
    args_dict = vars(config)
    config_file = os.path.join(experiment_path,'config_' + config.experiment_name + '.txt')
    if not os.path.isdir(experiment_path):
        os.makedirs(experiment_path)
    with open(config_file,'w') as f:
        for arg in args_dict.keys():
            s = "%0s :\t\t\t%0s"%(arg,str(args_dict[arg]))
            print(s)
            f.write(s + '\n')

    # run
    data = get_data()
    model = nets.pixelCNN()
    model.build('model', summarize=True)
    train(data, model, config)

