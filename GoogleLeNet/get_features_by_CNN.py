# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple image classification with Inception.

Run image classification with Inception trained on ImageNet 2012 Challenge data
set.

This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. It outputs human readable
strings of the top 5 predictions along with their probabilities.

Change the --image_file argument to any jpg image to compute a
classification of that image.

Please see the tutorial and website for a detailed description of how
to use this script to perform image recognition.

https://tensorflow.org/tutorials/image_recognition/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import os.path
import re
import sys
import tarfile
import cv2
import numpy as np
# from six.moves import urllib
import tensorflow as tf
import shutil
FLAGS = tf.app.flags.FLAGS
class_num = 3
image_path = '/media/chang/fe3dd8af-5577-42cb-95fb-4bd30a47cc9e/dataset/dynamic_texture/dynamicNew_gamma/jpg'
# classify_image_graph_def.pb:
#   Binary representation of the GraphDef protocol buffer.
# imagenet_synset_to_human_label_map.txt:
#   Map from synset ID to a human readable string.
# imagenet_2012_challenge_label_map_proto.pbtxt:
#   Text representation of a protocol buffer mapping a label to synset ID.
tf.app.flags.DEFINE_string(
    'model_dir', 'Graph/GoogleLeNet',
    """Path to classify_image_graph_def.pb, """
    """imagenet_synset_to_human_label_map.txt, and """
    """imagenet_2012_challenge_label_map_proto.pbtxt.""")
tf.app.flags.DEFINE_string('image_file', '',
                           """Absolute path to image file.""")
tf.app.flags.DEFINE_integer('num_top_predictions', 5,
                            """Display this many predictions.""")

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long

feature_path = '/media/chang/fe3dd8af-5577-42cb-95fb-4bd30a47cc9e/dataset/dynamic_texture/dynamicNew_gamma/google_npy_tfrecords/train'

class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_synset_to_human_label_map.txt')
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    """Loads a human readable English name for each softmax node.

    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.

    Returns:
      dict from integer node ID to human-readable string.
    """
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]


def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile('classify_image_graph_def.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(frame_image_path):
  """Runs inference on an image.

  Args:
    image: Image file name.

  Returns:
    Nothing
  """
  with tf.Session() as sess:
    #train_writer = tf.train.SummaryWriter('./Graph',sess.graph)
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.
  # maybe_download_and_extract()
    
    featrue_maps = []
    for i in os.listdir(frame_image_path):

      each_frame_jpg = os.path.join(frame_image_path, i)

      image_data = tf.gfile.FastGFile(each_frame_jpg, 'rb').read()

      # last_conv_layer = sess.graph.get_tensor_by_name('mixed_10/join:0')
      last_conv_layer = sess.graph.get_tensor_by_name('pool_3/_reshape:0')
      featrue_map = sess.run(last_conv_layer,
                           {'DecodeJpeg/contents:0': image_data})
      featrue_maps.append(featrue_map)

    featrue_maps = np.squeeze(featrue_maps)
    print('feature_maps.shape = ',featrue_maps.shape)
    return featrue_maps
    # Creates node ID --> English string lookup.
    # node_lookup = NodeLookup()

    # top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
    # for node_id in top_k:
    #   human_string = node_lookup.id_to_string(node_id)
    #   score = predictions[node_id]
    #   print('%s (score = %.5f)' % (human_string, score))
    


def maybe_download_and_extract():
  """Download and extract model tar file."""
  dest_directory = FLAGS.model_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    # filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def get_features_matrix(frame_matrix):
  maybe_download_and_extract()
  if os.path.exists(image_path):
    shutil.rmtree(image_path)
  os.mkdir(image_path)
  i=0
  for each_frame in frame_matrix:

    cv2.imwrite('/media/wyd/h51/frame_image/frame_'+str(i)+'.jpg', each_frame)
    i+=1

  with tf.Graph().as_default():
    
    featrue_maps = run_inference_on_image()
    # each_frame_jpg = frame_image_path+i
    # featrue_map = run_inference_on_image(each_frame_jpg)
    # featrue_maps.append(featrue_map)
    # featrue_maps = np.squeeze(featrue_maps)
    # print(featrue_maps.shape)
  return featrue_maps


def save_feat_into_disk(file_handle, feat, class_index, save_path, split_name):
    num_example = feat.shape[0]
    for i in range(num_example):
        file_path = os.path.join(save_path, split_name, str(class_index))
        if os.path.exists(file_path):
            print("Exists!")
        else:
            os.mkdir(file_path)
        # save
        # feat_name = file_path + '/feat_' + str(i) + '.jpg'
        feat_name = file_path + '/feat_' + str(i) + '.npy'
        file_handle.write(feat_name + ' ' + str(class_index) + '\n')
        # cv2.imwrite(feat_name, feat[i,])
        np.save(feat_name, feat[i, :])


def main(_):
  if not os.path.exists(feature_path):
      os.mkdir(feature_path)
    # if not tf.gfile.Exists(image):
  #   tf.logging.fatal('File does not exist %s', image)

  #image_data = tf.gfile.FastGFile(image, 'rb').read()

  # Creates graph from saved GraphDef.
  create_graph()
  #merged = tf.merge_all_summaries()

  for class_index in os.listdir(image_path):
    dirname = os.path.join(image_path, str(class_index))
    save_path = os.path.join(feature_path, str(class_index))
    if not os.path.exists(save_path):
      os.mkdir(save_path)
    for sample in os.listdir(dirname):
      frame_image_path = os.path.join(dirname, str(sample))
      feature = run_inference_on_image(frame_image_path)
      feat_name = os.path.join(save_path, 'feat_' + str(sample) + '.npy')
      np.save(feat_name, feature)

  print ('done')


if __name__ == '__main__':
  tf.app.run()
