#!/usr/bin/env python
import time
import argparse
from src.params import get_params
from src.Unet import Unet
from src.evaluate_model import evaluate_test_set

# ----------------------------------------------------------------------------------------------------------------------
# Define default pipeline
# ----------------------------------------------------------------------------------------------------------------------
# Create the parser. The formatter_class argument makes sure that default values are shown when --help is called.
parser = argparse.ArgumentParser(description='Pipeline for running the project',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

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
