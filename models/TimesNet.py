import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class Model(nn.Module):
    """
    TimesNet modified for classification tasks with adaptive input handling.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = 'classification'
        self.seq_len = configs.seq_len
        self.num_class = 3  # Three classes for classification
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)

        # Classification layers
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(configs.d_model * configs.seq_len, self.num_class)

    def classification(self, x_enc, x_mark_enc):
        # Determine input dimensions and adjust if necessary
        if len(x_enc.shape) == 2:
            # If input is a 2D tensor, assume shape is [B, T], where T is seq_len * enc_in
            batch_size = x_enc.shape[0]
            x_enc = x_enc.view(batch_size, self.seq_len, -1)

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, T, C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # Flatten the sequence
        output = self.projection(output)  # Linear layer for classification
        return F.log_softmax(output, dim=1)  # Use log_softmax for classification output

    def forward(self, x_enc, x_mark_enc):
        # Only classification task is used
        dec_out = self.classification(x_enc, x_mark_enc)
        return dec_out  # [B, num_class]


# Example configuration (configs) that can be used to initialize the model
class Configs:
    def __init__(self):
        self.seq_len = 800  # Length of the input sequence
        self.enc_in = 448  # Number of input features
        self.d_model = 64  # Dimension of the model
        self.embed = 'fixed'  # Embedding type
        self.freq = 'h'  # Frequency for embedding
        self.dropout = 0.1  # Dropout rate
        self.n_heads = 4  # Number of attention heads
        self.d_ff = 256  # Dimension of feed-forward network
        self.e_layers = 2  # Number of encoder layers
        self.factor = 5  # Factor for attention
        self.activation = 'gelu'  # Activation function
        self.top_k = 2  # Top k periods to consider
        self.num_kernels = 6  # Number of kernels for Inception blocks


# Initialize model
configs = Configs()
model = Model(configs)

# Example input
x_enc = torch.randn(32, configs.seq_len * configs.enc_in)  # Batch size of 32, flattened input of length 800 * 448
x_mark_enc = torch.randn(32, configs.seq_len)  # Additional time-related features if needed

# Forward pass
output = model(x_enc, x_mark_enc)
print(output.shape)  # Expected output: [32, 3], where 3 is the number of classes
