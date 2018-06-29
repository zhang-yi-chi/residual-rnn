from __future__ import print_function

import numpy as np
from scipy.misc import imsave
import tensorflow as tf

from data_loader import ImageLoader
from model import ResidualRNN
from msssim import msssim


flags = tf.app.flags
flags.DEFINE_string('path', 'imgs', 'dataset path')
flags.DEFINE_integer('epochs', 100, 'number of epochs')
flags.DEFINE_integer('display', 10, 'print step')
flags.DEFINE_integer('batch_size', 64, 'batch size of training dataset')
flags.DEFINE_float('lr', 1e-3, 'learning rate')
flags.DEFINE_integer('iters', 16, 'number of iterations of model')
flags.DEFINE_string('summaries_dir', 'tmp', 'tensorboard')

opts = flags.FLAGS

with tf.device('/cpu:0'):
    dataset = ImageLoader(path=opts.path, batch_size=opts.batch_size)
    iterator = tf.data.Iterator.from_structure(
        dataset.data.output_types, dataset.data.output_shapes)
    next_batch = iterator.get_next()

# Ops for initializing the two different iterators
training_init_op = iterator.make_initializer(dataset.data)
batches_per_epoch = dataset.batches_per_epoch

# Model
model = ResidualRNN(opts.batch_size, opts.iters)
train_loss = model.get_loss()
tf.summary.scalar('loss', train_loss)
compressed = model.get_compress()
# train_op = tf.train.MomentumOptimizer(opts.lr, 0.9).minimize(train_loss)
train_op = tf.train.AdamOptimizer(opts.lr).minimize(train_loss)
merged = tf.summary.merge_all()

saver = tf.train.Saver()

# Start Tensorflow session
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(opts.summaries_dir, sess.graph)
    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Add the model graph to TensorBoard
    # writer.add_graph(sess.graph)

    print("Start training... step per epoch: {}".format(
        batches_per_epoch))
    # print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
    #                                                   filewriter_path))

    best_ms_ssim = 0
    # Loop over number of epochs
    for epoch in range(opts.epochs):
        print("Epoch number: {}".format(epoch + 1))

        # Initialize iterator with the training dataset
        sess.run(training_init_op)
        tmp_loss = 0

        for step in range(batches_per_epoch):
            # get next batch of data
            img_batches = sess.run(next_batch)
            _, summary, cur_train_loss = sess.run([train_op, merged, train_loss], feed_dict={
                model.inputs: img_batches})
            train_writer.add_summary(summary, epoch * batches_per_epoch + step)
            tmp_loss += cur_train_loss
            if (step + 1) % opts.display == 0:
                tmp_loss /= opts.display
                print("Epoch number: {} step {} loss {}".format(
                    epoch + 1, step + 1, tmp_loss))
                tmp_loss = 0

        compressed_img = compressed.eval(
            feed_dict={model.inputs: img_batches}).clip(0, 255).astype(np.uint8)
        ms_ssim = msssim(compressed_img, img_batches)
        print("Epoch number: {} ms-ssim {}".format(epoch + 1, ms_ssim))
        if ms_ssim > best_ms_ssim:
            saver.save(sess, 'save/model')
            best_ms_ssim = ms_ssim
        compressed_img = np.concatenate(compressed_img, axis=1)
        original_img = np.concatenate(img_batches, axis=1).astype(np.uint8)
        output_img = np.concatenate((original_img, compressed_img), axis=0)
        imsave('eval/epoch_{}.jpg'.format(epoch + 1), output_img)
