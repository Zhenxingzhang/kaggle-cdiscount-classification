"""
Helper functions for learning
"""

from tensorflow.python import pywrap_tensorflow
import tensorflow as tf


def restore_variables_from_checkpoint(sess, checkpoint_dir, exclude=None,
                                      load_scope=None, verbose=False):
    '''
    Resrote saved variables from file
    '''
    vars_to_restore = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                        scope=load_scope)

    # Don't restore exclude variables
    exclude = [] if exclude is None else exclude
    vars_to_restore = [v for v in vars_to_restore
                       if not any([excl in v.name for excl in exclude])]

    with tf.name_scope('load_variables'):
        saver = tf.train.Saver(vars_to_restore)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if verbose:
            _print_tensors(ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)


def initialize_remaining_variables(sess, verbose=False):
    '''
    Initialize all variables that have not been restored
    '''
    with tf.name_scope('random_initialiser'):
        # Warn users of vars that have not been initialised
        uninitialized_vars = sess.run(tf.report_uninitialized_variables())
        uninitialized_vars = uninitialized_vars.astype(str)

        # Get variable objects with (w/ :<device number> striped off the names
        needs_initialisation = [v for v in tf.global_variables() +
                                tf.local_variables()
                                if v.name.split(':')[0] in uninitialized_vars]

        if len(needs_initialisation) != 0:
            group_by_type = {
                'trainable': {v.name.split('/')[-1]: [] for v in
                              needs_initialisation if v in
                              tf.trainable_variables()},
                'not_trainable': {v.name.split('/')[-1]: [] for v in
                                  needs_initialisation if v not in
                                  tf.trainable_variables()}
            }

            for var in needs_initialisation:
                key = ('trainable' if var in tf.trainable_variables()
                       else 'not_trainable')
                group_by_type[key][var.name.split('/')[-1]].append(var.name)

            if verbose:
                print("TRAINABLE Variables Initialised Automatically:")
                for k, v in group_by_type['trainable'].items():
                    print(k)

                print("NOT_TRAINABLE Variables Initialised Automatically:")
                for k, v in group_by_type['not_trainable'].items():
                    print("{} (total: {})".format(k, len(v)))

        # initialise using default initializers when required
        sess.run(tf.variables_initializer(needs_initialisation))


def _print_tensors(ckpt_file):
    '''
    Print names and shapes of tensors from checkpoint file
    '''
    print("Tensors Restored From File: {}".format(ckpt_file))
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_file)
    print(reader.debug_string().decode("utf-8"))
