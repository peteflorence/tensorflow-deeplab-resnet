"""Run DeepLab-ResNet on a given image.

This script computes a segmentation mask for a given image.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time

from PIL import Image

import tensorflow as tf
import numpy as np

from deeplab_resnet import DeepLabResNetModel, ImageReader, decode_labels, prepare_label

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network Inference.")
    parser.add_argument("--train_set", type=str, default="drill",
                        help="Number of classes to predict (including background).")
    args = parser.parse_args()
    train_set = args.train_set

    #test_set = "drill_11_test_scenes"

    NUM_CLASSES = 8
    SAVE_DIR = './testing_softmax_output/'
    DATA_DIR = '/'
    DATA_LIST_PATH = '/home/peteflo/spartan/src/CorlDev/experiments/sixobjects_multi_test_scenes.txt.imglist.txtdownsampled10.txt'
    DATA_DIRECTORY = ''
    IGNORE_LABEL = 255
    RESTORE_FROM = './snapshots_' + train_set + '/model.ckpt-20000'
   
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Where to save predicted mask.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    return parser.parse_args()

def load(saver, sess, ckpt_path):
    '''Load trained weights.

    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    '''
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()
    
    # Create queue coordinator.
    coord = tf.train.Coordinator()

    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            args.data_dir,
            args.data_list,
            None, # No defined input size.
            False, # No random scale.
            False, # No random mirror.
            args.ignore_label,
            IMG_MEAN,
            coord)
        image, label = reader.image, reader.label
    image_batch, label_batch = tf.expand_dims(image, dim=0), tf.expand_dims(label, dim=0) # Add one batch dimension.

    # Create network.
    net = DeepLabResNetModel({'data': image_batch}, is_training=False, num_classes=args.num_classes)

    # Which variables to load.
    restore_var = tf.global_variables()

    # Predictions.
    raw_output = net.layers['fc1_voc12']
    raw_output = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3,])
    #raw_output = tf.argmax(raw_output, dimension=3)
    #pred = tf.expand_dims(raw_output, dim=3) # Create 4-d tensor.
    pred = raw_output

    # Set up TF session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)
    sess.run(tf.local_variables_initializer())

    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    if args.restore_from is not None:
        load(loader, sess, args.restore_from)

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    with open(args.data_list) as f:
        content = f.readlines()

    content = [x.strip() for x in content]
    
    for index, value in enumerate(content):
        print("outputting "+str(index))
    	img = tf.image.decode_png(tf.read_file(value.split()[0]), channels=3)
    	# Convert RGB to BGR.
    	img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    	img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    	# Extract mean.
    	img -= IMG_MEAN 
    	# Predictions.
    	raw_output = net.layers['fc1_voc12']
    	raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(img)[0:2,])
    	pred = raw_output_up
        #raw_output_up = tf.argmax(raw_output_up, dimension=3)
    	#pred = tf.expand_dims(raw_output_up, dim=3)
    	# Perform inference.
    	preds = sess.run(pred)
        softmax = preds[0, :, :, :]
        print(softmax.shape)
        print(type(softmax))
        processed_probabilities = softmax.transpose((2, 0, 1))
        print(processed_probabilities.shape)
        print(type(processed_probabilities))
        im_preds = Image.fromarray(np.uint8(preds[0, :, :, 0]))

    	msk = decode_labels(preds, num_classes=args.num_classes)
    	im = Image.fromarray(msk[0])
    	if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        im_preds.save(args.save_dir +str(index).zfill(8) +'_predlabels_'+args.train_set+'.png')
    	im.save(args.save_dir +str(index).zfill(8) +'_pred_'+args.train_set+'.png')

if __name__ == '__main__':
    main()
