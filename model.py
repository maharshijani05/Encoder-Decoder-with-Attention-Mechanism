import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: [src_len, batch_size]
        embedded = self.dropout(self.embedding(src))  # [src_len, batch_size, emb_dim]
        outputs, (hidden, cell) = self.rnn(embedded)  
        # outputs: [src_len, batch_size, hid_dim], hidden and cell: [n_layers, batch_size, hid_dim]
        return outputs, hidden, cell


class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        # We learn to align encoder outputs and decoder hidden states
        self.attn = nn.Linear(hid_dim * 2, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        """
        hidden: [batch_size, hid_dim] (decoder hidden state at current time step)
        encoder_outputs: [src_len, batch_size, hid_dim] (all encoder outputs)

        returns: attention weights [batch_size, src_len]
        """
        src_len = encoder_outputs.shape[0]
        batch_size = encoder_outputs.shape[1]

        # Repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # [batch_size, src_len, hid_dim]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # [batch_size, src_len, hid_dim]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [batch_size, src_len, hid_dim]
        attention = self.v(energy).squeeze(2)  # [batch_size, src_len]

        return torch.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim + hid_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        # input: [batch_size]
        input = input.unsqueeze(0)  # [1, batch_size]

        embedded = self.dropout(self.embedding(input))  # [1, batch_size, emb_dim]

        # hidden[-1] is the top layer hidden state of decoder [batch_size, hid_dim]
        hidden_top = hidden[-1]

        # Calculate attention weights
        a = self.attention(hidden_top, encoder_outputs)  # [batch_size, src_len]

        # Weighted sum of encoder outputs to get context vector
        a = a.unsqueeze(1)  # [batch_size, 1, src_len]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # [batch_size, src_len, hid_dim]
        weighted = torch.bmm(a, encoder_outputs)  # [batch_size, 1, hid_dim]
        weighted = weighted.permute(1, 0, 2)  # [1, batch_size, hid_dim]

        # Concatenate embedded input word and context vector
        rnn_input = torch.cat((embedded, weighted), dim=2)  # [1, batch_size, emb_dim + hid_dim]

        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        # output: [1, batch_size, hid_dim]
        # embedded: [1, batch_size, emb_dim]
        # weighted: [1, batch_size, hid_dim]

        output = output.squeeze(0)  # [batch_size, hid_dim]
        embedded = embedded.squeeze(0)  # [batch_size, emb_dim]
        weighted = weighted.squeeze(0)  # [batch_size, hid_dim]

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))  # [batch_size, output_dim]

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [src_len, batch_size], trg: [trg_len, batch_size]
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        encoder_outputs, hidden, cell = self.encoder(src)

        input = trg[0, :]  # <sos>

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[t] = output
            top1 = output.argmax(1)
            input = trg[t] if torch.rand(1).item() < teacher_forcing_ratio else top1

        return outputs

