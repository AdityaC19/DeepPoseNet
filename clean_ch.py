# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import argparse
import os
import os.path as osp
import glob
import pickle

from tqdm import tqdm
import numpy as np


def clean_fn(fn, output_folder='output'):
    with open(fn, 'rb') as body_file:
        # Use 'latin1' encoding to handle non-ASCII characters
        body_data = pickle.load(body_file, encoding='latin1')

    output_dict = {}
    for key, data in body_data.items():  # Changed from .iteritems() to .items() for Python 3 compatibility
        if 'chumpy' in str(type(data)):
            output_dict[key] = np.array(data)
        else:
            output_dict[key] = data

    out_fn = osp.split(fn)[1]
    out_path = osp.join(output_folder, out_fn)
    with open(out_path, 'wb') as out_file:
        pickle.dump(output_dict, out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-models', dest='input_models',
                        required=True, type=str,
                        help='The path to the directory containing model files to process')
    parser.add_argument('--output-folder', dest='output_folder',
                        required=True, type=str,
                        help='The path to the output folder')

    args = parser.parse_args()

    input_models_dir = args.input_models
    output_folder = args.output_folder

    if not osp.exists(output_folder):
        print('Creating directory: {}'.format(output_folder))
        os.makedirs(output_folder)

    # Use glob to find all .pkl files in the input_models directory
    pkl_files = glob.glob(osp.join(input_models_dir, "*.pkl"))

    # Process each .pkl file found in the directory
    for input_model in tqdm(pkl_files, desc="Processing models"):
        clean_fn(input_model, output_folder=output_folder)
