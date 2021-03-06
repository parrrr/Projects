"""
This file contains code that will kick off training and testing processes
"""
import json
import os
import numpy as np
from data_prep.HippocampusDatasetLoader import LoadHippocampusData
from experiments.UNetExperiment import UNetExperiment
from sklearn.model_selection import train_test_split


class Config:
    """
    Holds configuration parameters
    """
    def __init__(self):
        self.name = "Basic_UNet"
        self.root_dir = "C:\\Users\\Par2\\Documents\\nd320-c3-3d-imaging-starter-master\\section2\\src\\data"
        self.n_epochs = 5
        self.learning_rate = 0.0001
        self.batch_size = 8
        self.patch_size = 64
        self.test_results_dir = "..\\result4"

if __name__ == "__main__":
    # Get configuration

    # Config class
    c = Config()

    # Load data
    print("Loading data...")
 
    data = LoadHippocampusData(c.root_dir, y_shape = c.patch_size, z_shape = c.patch_size)


    # Create test-train-val split

    keys = range(len(data))

    # Here, random permutation of keys array would be useful in case if we do something like 
    # a k-fold training and combining the results. 

    split = dict()

    # create three keys in the dictionary: "train", "val" and "test". In each key, store
    # the array with indices of training volumes to be used for training, validation 
    # and testing respectively.
    #split['train'n copied into loaders
    #del dataset], val_keys = train_test_split(keys, test_size = 0.2)
    #split['val'], split['test'] = train_test_split(val_keys, test_size = 0.5)
    shuffled_keys = np.random.RandomState(seed=1).permutation(keys)

    split['train'] = shuffled_keys[:int(0.7 * len(shuffled_keys))]
    split['val'] = shuffled_keys[int(0.7 * len(shuffled_keys)):int(0.9 * len(shuffled_keys))]
    split['test'] = shuffled_keys[int(0.9 * len(shuffled_keys)):]

    #     # Set up and run experiment

    exp = UNetExperiment(c, split, data)

    # run training
    exp.run()

    # prep and run testing

    # Test method
    results_json = exp.run_test()

    results_json["config"] = vars(c)

    with open(os.path.join(exp.out_dir, "results.json"), 'w') as out_file:
        json.dump(results_json, out_file, indent=2, separators=(',', ': '))

