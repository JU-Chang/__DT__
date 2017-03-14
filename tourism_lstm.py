import tensorflow as tf
import convert_to_tfrecords as ctt
import numpy as np


tf.app.flags.DEFINE_integer('batch_size', 30,
                            'Batch size.')
tf.app.flags.DEFINE_integer('num_epochs', 20, 'Number of epochs to run trainer.')
FLAGS = tf.app.flags.FLAGS


num_class = 10
num_units = 100
num_layers = 1
learning_rate = 0.001
output_keep_prob = 1.0
input_keep_prob = 1.0
test_size = 1


# one-hot encoding,return one_hot_label of size[batch_size*max_time*num_class]
def one_hot(label_list, sequence_len, data_type='train'):
    one_hot_label = np.zeros([FLAGS.batch_size, max(sequence_len), num_class], dtype='float32')
    instances_num = FLAGS.batch_size
    if not data_type == 'train':
        instances_num = 1
    for index_of_example in range(instances_num):
        one_hot_label[index_of_example, 0:sequence_len[index_of_example], label_list[index_of_example]] = 1.0
    return one_hot_label


def slot_shuffle(feature_data, labels_data, sequence_len):
    perm = range(FLAGS.batch_size)
    np.random.shuffle(perm)
    feature_data = feature_data[perm]
    labels_data = labels_data[perm]
    sequence_len = sequence_len[perm]
    return feature_data, labels_data, sequence_len


def lstm(inputs_placeholder, rows_placeholder, dropout):
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units,
                                             forget_bias=1.0,
                                             state_is_tuple=False)
    if dropout:
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,
                                                  input_keep_prob=input_keep_prob,
                                                  output_keep_prob=output_keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers, state_is_tuple=False)
    initial_state = cell.zero_state(FLAGS.batch_size, dtype=tf.float32)
    outputs, output_states = tf.nn.dynamic_rnn(
        cell=cell,
        inputs=inputs_placeholder,
        sequence_length=rows_placeholder,
        initial_state=initial_state,
        dtype=tf.float32,
        swap_memory=False,
        time_major=False)
    return outputs


# def inference_of_one_example(outputs, example, softmax_w, softmax_b, labels_placeholder, rows_placeholder):
#     logits = tf.nn.softmax(tf.matmul(outputs[example, :, :], softmax_w) + softmax_b)
#     cross_entropy = -tf.reduce_sum(labels_placeholder[example, :, :] * tf.log(logits))
#     one_example_loss = tf.reduce_sum(cross_entropy, name='slot_loss') / rows_placeholder[example]
#     filter_mat = tf.expand_dims(tf.reduce_sum(labels_placeholder[example, :, :], 1), 1)
#     a = tf.argmax(tf.mul(logits, filter_mat), 1)
#     b = tf.argmax(labels_placeholder[example, :, :], 1)
#     correct = tf.equal(a, b)
#     eval_correct = tf.reduce_sum(tf.cast(correct, tf.int32))
#     return one_example_loss, eval_correct

def inference_of_one_example(outputs, example, softmax_w, softmax_b):
    logits = tf.nn.softmax(tf.matmul(outputs[example, :, :], softmax_w) + softmax_b)
    return logits


