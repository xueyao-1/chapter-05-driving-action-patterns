import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.init as init


# Bi-direction LSTM
class ImprovedLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate=0.5, bidirectional=True):
        super(ImprovedLSTMModel, self).__init__()
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=dropout_rate if num_layers > 1 else 0,
                            bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim * self.num_directions, output_dim)

    def forward(self, x, lengths):
        packed_input = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_input)
        if self.num_directions == 2:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        else:
            hidden = hidden[-1,:,:]
        out = self.fc(hidden)
        # out = F.softmax(out, dim=1)
        # print("out", out)
        return out

# # Attention-based Bi-direction LSTM
class Attention(nn.Module):
    def __init__(self, hidden_dim, output_dim, bidirectional=True):
        super(Attention, self).__init__()
        self.num_directions = 2 if bidirectional else 1
        self.attention_linear = nn.Linear(hidden_dim * self.num_directions, output_dim)

#     def forward(self, lstm_output):
#         batch_size, seq_len, _ = lstm_output.shape
#         lstm_output_flat = lstm_output.contiguous().view(-1, lstm_output.shape[2])
#         attention_scores = self.attention_linear(lstm_output_flat).view(batch_size, seq_len, -1)
#         attention_scores = F.softmax(attention_scores, dim=1).unsqueeze(-1)
#         weighted_output = lstm_output * attention_scores
#         attn_output_summed = weighted_output.sum(dim=1)
#         return attn_output_summed
# class Attention(nn.Module):
#     def __init__(self, hidden_dim, bidirectional=True):
#         super(Attention, self).__init__()
#         self.num_directions = 2 if bidirectional else 1
#         self.attention_linear = nn.Linear(hidden_dim * self.num_directions, 1)

    def forward(self, lstm_output):
        attention_scores = self.attention_linear(lstm_output).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(-1)
        weighted_output = lstm_output * attention_weights
        attn_output_summed = weighted_output.sum(dim=1)
        return attn_output_summed

class BidirectionalLSTMWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate=0.5, bidirectional=True):
        super(BidirectionalLSTMWithAttention, self).__init__()
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0, bidirectional=bidirectional)
        self.attention_layer = Attention(hidden_dim * self.num_directions)  # Pass the correct input dimension to Attention
        self.fc = nn.Linear(hidden_dim * self.num_directions, output_dim)

    def forward(self, x, lengths):
        packed_input = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = rnn_utils.pad_packed_sequence(packed_output, batch_first=True)
        attn_output = self.attention_layer(output)
        out = self.fc(attn_output)
        # out = F.softmax(out, dim=1)
        out = out.view(out.size(0), -1)
        return out


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate=0.0, bidirectional=False):
        super(LSTMModel, self).__init__()
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout_rate if num_layers > 1 else 0, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim * self.num_directions, output_dim)

        self.init_weights()

    def init_weights(self):
        pass

        # Initialize weights as before

    def forward(self, x, lengths):
        packed_input = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_input)

        if self.num_directions == 2:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]

        out = self.fc(hidden)
        return out


# # Simple LSTM
# class LSTMModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate=0.0):
#         super(LSTMModel, self).__init__()
#         self.bn = nn.BatchNorm1d(input_dim)  # BatchNorm for the feature dimension
#
#         self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
#                             dropout=dropout_rate if num_layers > 1 else 0)
#         self.dropout = nn.Dropout(dropout_rate)
#         self.fc = nn.Linear(hidden_dim, output_dim)
#
#         self.init_weights()
#
#     def init_weights(self):
#         # LSTM weights initialization
#         for name, param in self.lstm.named_parameters():
#             if 'weight_ih' in name:
#                 init.xavier_uniform_(param.data)
#             elif 'weight_hh' in name:
#                 init.orthogonal_(param.data)
#             elif 'bias' in name:
#                 param.data.fill_(0)
#
#         # Fully connected layer initialization
#         init.xavier_uniform_(self.fc.weight)
#         self.fc.bias.data.fill_(0)
#
#         # BatchNorm layer initialization
#         init.ones_(self.bn.weight)
#         init.zeros_(self.bn.bias)
#
#     def forward(self, x, lengths):
#         # Apply batch normalization
#         x = x.permute(0, 2, 1)  # Reshape to [batch_size, num_features, seq_length]
#         x = self.bn(x)
#         x = x.permute(0, 2, 1)  # Reshape back to [batch_size, seq_length, num_features]
#
#         packed_input = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
#         packed_output, (hidden, cell) = self.lstm(packed_input)
#
#         hidden = hidden[-1,:,:]
#         out = self.fc(hidden)
#         # print("out", out)
#         return out


        # packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        # packed_output, (hidden, cell) = self.lstm(packed_input)
        # output, _ = pad_packed_sequence(packed_output, batch_first=True)
        #
        # out = hidden[-1]  # Use the last hidden state
        # out = self.dropout(out)
        # print("pre_output", out)
        # out = self.fc(out)
        # output = F.softmax(out, dim=1)
        # print("output", output)
        # return output


# class LSTMModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate):
#         super(LSTMModel, self).__init__()
#         if num_layers > 1:
#             self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
#                                 batch_first=True, dropout=dropout_rate)
#         else:
#             # No dropout if there's only one layer
#             self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
#                                 batch_first=True)
#         self.fc = nn.Linear(hidden_dim, output_dim)
#
#     def forward(self, x, lengths):
#         packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
#         packed_output, (hidden, cell) = self.lstm(packed_input)
#         output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
#         out = self.fc(hidden[-1])
#         return out


# Attention-based LSTM
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        attention_scores = self.attention(lstm_output)  # [batch_size, seq_length, 1]
        attention_scores = F.softmax(attention_scores, dim=1)
        weighted_output = lstm_output * attention_scores
        context_vector = torch.sum(weighted_output, dim=1)
        return context_vector

class AttentionBasedModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate=0.0):
        super(AttentionBasedModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.attention_layer = AttentionLayer(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lengths):
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        unpacked_output, _ = pad_packed_sequence(packed_output, batch_first=True)
        attn_out = self.attention_layer(unpacked_output)
        attn_out = self.dropout(attn_out)
        output = self.fc(attn_out)
        # output = F.softmax(output, dim=1)
        return output
