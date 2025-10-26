import torch
import torch.nn as nn


class GLSTM(nn.Module):

    def __init__(self, in_features=None, out_features=None, hidden_size=896, groups=2):
        super().__init__()

        hidden_size_t = hidden_size // groups

        self.lstm_list1 = nn.ModuleList([nn.LSTM(hidden_size_t, hidden_size_t, 1, batch_first=True) for i in range(groups)])
        self.lstm_list2 = nn.ModuleList([nn.LSTM(hidden_size_t, hidden_size_t, 1, batch_first=True) for i in range(groups)])

        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        self.groups = groups

    def forward(self, x):
        out = x
        out = out.transpose(1, 2).contiguous()
        out = out.view(out.size(0), out.size(1), -1).contiguous()
        out = torch.chunk(out, self.groups, dim=-1)

        out = torch.stack([self.lstm_list1[i](out[i])[0] for i in range(self.groups)], dim=-1)
        out = torch.flatten(out, start_dim=-2, end_dim=-1)
        out = self.ln1(out)

        out = torch.chunk(out, self.groups, dim=-1)
        out = torch.cat([self.lstm_list2[i](out[i])[0] for i in range(self.groups)], dim=-1)
        out = self.ln2(out)

        out = out.view(out.size(0), out.size(1), x.size(1)).contiguous()

        out = out.transpose(1, 2).contiguous()

        return out