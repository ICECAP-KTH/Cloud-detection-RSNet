#!/usr/bin/env python
import time
import argparse
from src.params import get_params
from src.Unet import Unet
from src.evaluate_model import evaluate_test_set

# Don't allow tensorflow to reserve all memory available
# from keras.backend import set_session
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran) (nothing gets printed in Jupyter, only if you run it standalone)
# sess = tf.compat.v1.Session(config=config)  # set this TensorFlow session as the default session for Keras
# set_session(sess)

# ----------------------------------------------------------------------------------------------------------------------
# Define default pipeline
# ----------------------------------------------------------------------------------------------------------------------
# Create the parser. The formatter_class argument makes sure that default values are shown when --help is called.
parser = argparse.ArgumentParser(description='Pipeline for running the project',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Define which steps should be run automatically when this file is run. When using action='store_true', the argument
# has to be provided to run the step. When using action='store_false', the step will be run when this file is executed.
parser.add_argument('--test',
                    type=str,
                    default='KTHTest',
                    help='Predict cloud cover on these images')

if __name__ == '__main__':
    # Load the arguments
    args = parser.parse_args()

    # Store current time to calculate execution time later
    start_time = time.time()

    print("\n---------------------------------------")
    print("Script started")
    print("---------------------------------------\n")

    # Load hyperparameters into the params object containing name-value pairs
    params = get_params()

    if args.test:
        # Use the trained model
        model = Unet(params)
        evaluate_test_set(model, params, args.test)

    # Print execution time
    exec_time = str(time.time() - start_time)
    print("\n---------------------------------------")
    print("Script executed in: " + exec_time + "s")
    print("---------------------------------------")
