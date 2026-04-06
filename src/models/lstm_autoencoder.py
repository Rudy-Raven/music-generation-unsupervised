import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size=128, hidden_size=256, latent_size=64):
        super(LSTMAutoencoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.fc_enc = nn.Linear(hidden_size, latent_size)

        self.fc_dec = nn.Linear(latent_size, hidden_size)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=input_size,
            batch_first=True
        )

    def encode(self, x):
        _, (hidden, _) = self.encoder_lstm(x)
        z = self.fc_enc(hidden[-1])
        return z

    def decode(self, z, seq_len):
        decoder_input = self.fc_dec(z).unsqueeze(1).repeat(1, seq_len, 1)
        output, _ = self.decoder_lstm(decoder_input)
        return output

    def forward(self, x):
        z = self.encode(x)
        output = self.decode(z, x.size(1))
        return output