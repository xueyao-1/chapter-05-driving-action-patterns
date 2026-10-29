# Chapter 5: Driving action patterns

Code, small data files, and figures accompanying this chapter. The original folder structure is preserved.

## Software

The analysis uses Jupyter notebooks and Python. Dependencies and execution paths have not yet been validated. Consult the notebooks and scripts for imports and input paths.

## Data included and omitted

Data files of at most 10 MB (10,000,000 bytes) are included. Larger files are omitted; workflows requiring those files need the original data before they can run. No replacement or sampled data has been generated.

| Omitted file | Size (MB) |
| --- | ---: |
| `Data_source_and_script/01_action_pattern_calibration/HH_data_0-3000.csv` | 66.33 |
| `Data_source_and_script/01_action_pattern_calibration/action_phase_set_original.npy` | 39.07 |
| `Data_source_and_script/01_action_pattern_calibration/action_phases_list.npy` | 19.31 |
| `Data_source_and_script/01_action_pattern_calibration/all_action_phases.npy` | 21.04 |
| `Data_source_and_script/02_action_pattern_classification/labeled_data.json` | 45.44 |
| `Data_source_and_script/02_action_pattern_classification/trainHH_data.csv` | 792.79 |

## Original chapter instructions

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
