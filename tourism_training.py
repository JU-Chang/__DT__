import time
import tensorflow as tf
import convert_to_single_tfrecords as ctt
import tourism_lstm as tl
import tourism_eval as te
import numpy as np


def eval_and_loss(softmax_w, softmax_b, outputs, labels_placeholder, rows_placeholder):

    batch_loss_sum = 0.0
    batch_correct = 0
    for example in range(tl.FLAGS.batch_size):
        one_example_loss, eval_correct = tl.inference_of_one_example(
            outputs, example, softmax_w, softmax_b, labels_placeholder, rows_placeholder)
        batch_loss_sum += one_example_loss
        batch_correct += eval_correct
    loss = batch_loss_sum / tl.FLAGS.batch_size
    return loss, batch_correct


def training(loss):
    optimizer = tf.train.AdamOptimizer(tl.learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def run_training():

    with tf.Graph().as_default():

        # Extract data from tfrecords
        data_set_type = ['train', 'validation', 'test']
        data_sets = []
        for i in range(3):
            images, labels, rows = ctt.inputs(data_set_type=data_set_type[i],
                                              batch_size=tl.FLAGS.batch_size,
                                              num_epochs=tl.FLAGS.num_epochs)
            data_sets.append([images, labels, rows])
        with tf.name_scope('training'):
            inputs_placeholder = tf.placeholder("float32", [tl.FLAGS.batch_size, None, ctt.FLAGS.feature_col])
            rows_placeholder = tf.placeholder("float32", [tl.FLAGS.batch_size])
            labels_placeholder = tf.placeholder("float32", [tl.FLAGS.batch_size, None, tl.num_class])
            # train_placeholder = tf.placeholder("bool")

        with tf.variable_scope("output_layer"):
            softmax_w = tf.get_variable(name='weights', shape=[tl.num_units, tl.num_class], dtype=tf.float32)
            softmax_b = tf.get_variable(name='biases', shape=[tl.num_class], initializer=tf.constant_initializer(0.0))
        # Add to the Graph operations that train the model.

        # LSTM
        outputs = tl.lstm(inputs_placeholder, rows_placeholder, False)

        # logits
        loss, eval_correct = eval_and_loss(softmax_w, softmax_b, outputs, labels_placeholder, rows_placeholder)

        train_op = training(loss)

        # summary = tf.merge_all_summaries()

        # The op for initializing the variables.
        init_op = tf.group(tf.initialize_all_variables(),
                           tf.initialize_local_variables())

        sess = tf.Session()

        # Initialize the variables (the trained variables and the
        # epoch counter).

        # Instantiate a SummaryWriter to output summaries and the Graph.
        # summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            step = 0
            while not coord.should_stop():

                start_time = time.time()

                feature_data, labels_data, sequence_len = sess.run(data_sets[0])
                # slot_shuffle
                feature_data, labels_data, sequence_len = tl.slot_shuffle(feature_data,
                                                                          labels_data,
                                                                          sequence_len)

                labels_data = tl.one_hot(labels_data, sequence_len)

                _, loss_value = sess.run([train_op, loss],
                                         feed_dict={inputs_placeholder: feature_data,
                                                    rows_placeholder: sequence_len,
                                                    labels_placeholder: labels_data})

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                duration = time.time() - start_time

                # Write the summaries and print an overview fairly often.
                if step % 10 == 0:
                    # Print status to stdout.
                    print('Step %d: loss = %.2f(%.3f sec)' % (step, loss_value, duration))
                    # summary_writer.add_summary(summary)
                    # summary_writer.flush()

                    # evaluation
                    te.evaluation(inputs_placeholder, rows_placeholder, labels_placeholder,
                                  tl.test_batch, 'test', data_sets, eval_correct, sess)

                step += 1

        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (tl.FLAGS.num_epochs, step))

        finally:
            coord.request_stop()

        coord.join(threads)
        sess.close()


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
