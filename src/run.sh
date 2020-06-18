#!/bin/bash

python hist_euqal.py
python split_10.py
python data_test_augment.py
python cp_to_train.py
python main.py
python model_fusion.py
