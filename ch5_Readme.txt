01_action_pattern_calibration

Input data:
HH_data_0-3000.csv: Human-vehicle (HV) following HV, 3000 vehicles, after filtered

Main script: action_pattern_calibration.ipynb

Segmentation: extract Action phases
Get data: all_action_phases.npy, all_action_labels.npy, total_states.npy, action_phases_list.npy, action_phases_labels_list.npy

Resample data, get: normalized_data.npy

After PCA, get: action_phase_components.npy

Clustering: Hierarchical clustering and X-means, get: agg_cluster01.pdf, xmeans_cluster01.pdf, Xmeans_cluster_metrics.csv, Agg_cluster_metrics.csv

Normalized data - KL, get: KL_check.pdf


02_action_pattern_classification

Input data: raw driving data - trainHH_data.csv

labelling.ipynb: after labelling, get labeled_data.csv, labeled_data.json, labels.txt

train_model.py: main file for model training
deeplearning_model.py: including LSTM, Bi-LSTM, ALSTM, and ABLST models