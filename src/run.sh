#!/bin/bash
# python 00_profile_report.py
python 01_labelencoding.py
python 02_featuretools.py
python 03_powertransform.py
python 04_sequence.py
python 04_sequence.py --credit
python 05_onehot.py
python 06_onehot_seq.py
python 06_onehot_seq.py --credit
python 11_baseline.py
python 12_featuretools.py
python 13_mlp.py
python 13_mlp.py --onehot
python 14_lstm.py
python 14_lstm.py --onehot
python 15_cnn.py
python 15_cnn.py --onehot
