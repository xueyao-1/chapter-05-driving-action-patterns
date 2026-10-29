import matplotlib.pyplot as plt
from scipy import signal
import torch.nn.functional as F
import json
import csv
import wandb
import random
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from deeplearning_model import ImprovedLSTMModel, AttentionBasedModel, BidirectionalLSTMWithAttention, LSTMModel 
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Path to your labels file
file_path = 'labels.txt'
# load labeled data
with open('labeled_data.json', 'r') as json_file:
    data_loaded = json.load(json_file)

with open(file_path, 'r') as file:
    labels = [line.strip() for line in file]
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
assert len(set(encoded_labels)) <= 6, "There are more than 6 unique labels."

all_v, all_a, all_T, all_dv = [], [], [], []

for segment in data_loaded:
    all_v.extend(segment[0])  # All 'v' values across segments
    all_a.extend(segment[1])  # All 'a' values across segments
    all_T.extend(segment[2])  # All 'T' values across segments
    all_dv.extend(segment[3])  # All 'dv' values across segments

scaler = MinMaxScaler()

norm_v = scaler.fit_transform(np.array(all_v).reshape(-1, 1)).flatten()
norm_a = scaler.fit_transform(np.array(all_a).reshape(-1, 1)).flatten()
norm_T = scaler.fit_transform(np.array(all_T).reshape(-1, 1)).flatten()
norm_dv = scaler.fit_transform(np.array(all_dv).reshape(-1, 1)).flatten()

normalized_segments = []
index = 0

for segment in data_loaded:
    segment_length = len(segment[0])  # Length of each type of variable in the segment
    normalized_segment = [
        norm_v[index:index + segment_length],
        norm_a[index:index + segment_length],
        norm_T[index:index + segment_length],
        norm_dv[index:index + segment_length]
    ]
    index += segment_length  # Update index for the next segment
    normalized_segments.append(normalized_segment)

tensor_sequences = []
for segment in normalized_segments:
    # Convert the list of numpy arrays to a single numpy array
    segment_array = np.array(segment).T  # Transpose to shape [sequence_length, 4]

    # Convert the numpy array to a PyTorch tensor
    segment_tensor = torch.tensor(segment_array, dtype=torch.float32)
    tensor_sequences.append(segment_tensor)

# Pad sequences
padded_sequences = pad_sequence(tensor_sequences, batch_first=True, padding_value=0)

class CustomDataset(Dataset):
    def __init__(self, sequences, labels, lengths):
        self.sequences = sequences
        self.labels = labels
        self.lengths = lengths

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], self.lengths[idx]

lengths = [len(sequence) for sequence in padded_sequences]
if not isinstance(padded_sequences, torch.Tensor):
    padded_sequences = torch.tensor(padded_sequences, dtype=torch.float32)

if not isinstance(encoded_labels, torch.Tensor):
    encoded_labels = torch.tensor(encoded_labels, dtype=torch.long)

test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, encoded_labels, test_size=test_size, random_state=42
)
lengths_train, lengths_test = train_test_split(lengths, test_size=0.2, random_state=42)

assert isinstance(X_train, torch.Tensor) and isinstance(X_test, torch.Tensor)
assert isinstance(y_train, torch.Tensor) and isinstance(y_test, torch.Tensor)

# Create TensorDataset for training and testing
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_dataset = CustomDataset(X_train, y_train, lengths_train)
test_dataset = CustomDataset(X_test, y_test, lengths_test)

batch_size = 64  # You can adjust the batch size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
# , pin_memory=True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

input_dim = 4  # Number of features
hidden_dim = 64
output_dim = 6 # Total number of label categories
num_layers = 3
dropout_rate = 0.6
# LSTM
# model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers, dropout_rate)
# Bi-LSTM
model = ImprovedLSTMModel(input_dim, hidden_dim, output_dim, num_layers, dropout_rate, bidirectional=True)
# Attention-Bi-LSTM
# model = BidirectionalLSTMWithAttention(input_dim, hidden_dim, output_dim, num_layers, dropout_rate)
# Attention-based
# model = AttentionBasedModel(input_dim, hidden_dim, output_dim, num_layers, dropout_rate)
model.to(device)  # Transfer the model to the GPU

# model_path = '/home/xyao/DeepLearning/LSTM/model_weights.pth'
#
# checkpoint = torch.load(model_path)
# model.load_state_dict(checkpoint['state_dict'])

# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
#     break  # 
#
# state = model.load_state_dict(checkpoint['state_dict'], strict=False)
# print(state)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()

training_losses = []
training_accuracy = []
testing_losses = []
testing_accuracy = []
testing_precision = []
testing_recall = []
testing_f1 = []
# Training loop
num_epochs = 600 # Set the number of epochs
torch.cuda.empty_cache()
#init wandb project
wandb.init(
    # set the wandb project where this run will be logged
    project="test_project",name=model.__class__.__name__
)

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    train_loss = 0
    correct_train = 0
    total_train = 0

    for sequences, labels, lengths in train_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        # lengths = lengths.to(dtype=torch.int64, device=device)  # Ensuring lengths are tensors
        optimizer.zero_grad()   # Clear existing gradients
        output = model(sequences, lengths)
        # output = model(sequences)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted_train = torch.max(output.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()

    train_loss /= len(train_loader)
    train_accuracy = 100 * correct_train / total_train
    training_losses.append(train_loss)
    training_accuracy.append(train_accuracy)

    # model_path = '/home/xyao/DeepLearning/LSTM/model_weights.pth'  # .pth是PyTorch模型的常用文件扩展名
    # torch.save(model.state_dict(), model_path)
    # checkpoint = torch.load(model_path)
    # model.load_state_dict(checkpoint)
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    #     break  
    # try:
    #     state = model.load_state_dict(checkpoint, strict=False)
    #     print(state)
    # except RuntimeError as e:
    #     print(e)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for sequences, labels, lengths in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences, lengths)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    test_loss /= len(test_loader)
    # test_accuracy = 100 * correct / total
    # testing_losses.append(test_loss)
    # testing_accuracy.append(test_accuracy)
    test_accuracy = 100 * accuracy_score(all_labels, all_predictions)
    test_precision = 100 * precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    test_recall = 100 * recall_score(all_labels, all_predictions, average='macro', zero_division=0)
    test_f1 = 100 * f1_score(all_labels, all_predictions, average='macro', zero_division=0)

    testing_losses.append(test_loss)
    testing_accuracy.append(test_accuracy)
    testing_precision.append(test_precision)
    testing_recall.append(test_recall)
    testing_f1.append(test_f1)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    wandb.log({'train_loss':train_loss,'test_loss':test_loss})

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Test Loss: {test_loss}, "
          f"Train Accuracy: {train_accuracy:.2f}, Test Accuracy: {test_accuracy:.2f}, Test Precision: {test_precision:.2f}, "
          f"Test Recall: {test_recall:.2f}, Test F1: {test_f1:.2f}")

metrics_data = {
    'Epoch': range(1, num_epochs + 1),
    'Training Loss': training_losses,
    'Training Accuracy': training_accuracy,
    'Testing Loss': testing_losses,
    'Testing Accuracy': testing_accuracy,
    'Test Precision': testing_precision,
    'Test Recall': testing_recall,
    'Test F1': testing_f1
}

metrics_df = pd.DataFrame(metrics_data)

# Save the DataFrame to a CSV file
metrics_file_path = "Result_model_metrics.csv"
metrics_df.to_csv(metrics_file_path, index=False)
