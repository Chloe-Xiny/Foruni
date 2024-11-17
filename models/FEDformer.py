import torch 
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding
from layers.AutoCorrelation import AutoCorrelationLayer
from layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from layers.Autoformer_EncDec import Encoder, EncoderLayer, my_Layernorm, series_decomp


class Model(nn.Module):
    """
    FEDformer modified for classification tasks
    """

    def __init__(self, configs, version='fourier', mode_select='random', modes=32):
        """
        version: str, for FEDformer, there are two versions to choose, options: [Fourier, Wavelets].
        mode_select: str, for FEDformer, there are two mode selection method, options: [random, low].
        modes: int, modes to be selected.
        """
        super(Model, self).__init__()
        self.task_name = 'classification'
        self.seq_len = configs.seq_len
        self.num_class = 3  # Three classes for classification

        self.version = version
        self.mode_select = mode_select
        self.modes = modes

        # Embedding layer
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        # Encoder
        if self.version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(ich=configs.d_model, L=1, base='legendre')
        else:
            encoder_self_att = FourierBlock(in_channels=configs.d_model,
                                            out_channels=configs.d_model,
                                            n_heads=configs.n_heads,
                                            seq_len=self.seq_len,
                                            modes=self.modes,
                                            mode_select_method=self.mode_select)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,  # instead of multi-head attention in transformer
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )

        # Classification layers
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(configs.d_model * configs.seq_len, self.num_class)

    def classification(self, x_enc, x_mark_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

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
        self.moving_avg = 25  # Moving average window size
        self.activation = 'gelu'  # Activation function


# Initialize model
configs = Configs()
model = Model(configs)

# Example input
x_enc = torch.randn(32, configs.seq_len, configs.enc_in)  # Batch size of 32, sequence length of 800, 448 features
x_mark_enc = torch.randn(32, configs.seq_len)  # Additional time-related features if needed

# Forward pass
output = model(x_enc, x_mark_enc)
print(output.shape)  # Expected output: [32, 3], where 3 is the number of classes
