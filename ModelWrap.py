from DataScale import DataScale
import torch
from torch import nn, cuda

NUM_NUMERICAL_COLS = 6
TWEET_EMBEDDING_DIM = 1536

# Define the hyperparameters

# N_HEAD = 4
# NUM_ENCODER_LAYERS = 4
# DROPOUT = 1e-1
# EMBEDDING_DIM = 7  # 6 + 2*21 = 48 % 16 = 0
# # EMBEDDING_DIM = 1  # 6 + 2*21 = 48 % 16 = 0

N_HEAD = 16
NUM_ENCODER_LAYERS = 12
DROPOUT = 1e-1

# N_HEAD = 32
# NUM_ENCODER_LAYERS = 24
# DROPOUT = 1e-1

# EMBEDDING_DIM = 125  # 6 + 2*21 = 48 % 16 = 0
EMBEDDING_DIM = 717  # 6 + 2*21 = 48 % 16 = 0
# EMBEDDING_DIM = 29  # 6 + 2*21 = 48 % 16 = 0


N_DIM_OUT = NUM_NUMERICAL_COLS


class ModelWrap(DataScale):

    def __init__(self):
        super().__init__()
        n_symbols = len(self.symbol_to_code)
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        print(f'device is: {self.device}')
        self.model = Model(N_HEAD, NUM_ENCODER_LAYERS, DROPOUT, n_symbols,
                           NUM_NUMERICAL_COLS, EMBEDDING_DIM, TWEET_EMBEDDING_DIM).to(self.device)


class Model(nn.Module):
    def __init__(self, n_head, num_encoder_layers, dropout, num_symbols,
                 num_numerical_cols, embedding_dim, tweet_embedding_dim):
        super(Model, self).__init__()
        self.symbol_embedding = nn.Embedding(num_symbols, embedding_dim)
        self.tweet_processor = nn.Sequential(
            nn.Linear(tweet_embedding_dim, embedding_dim),
            nn.Tanh()
            # nn.ReLU()
        )
        # Assuming the dimension after concatenating symbol embedding, numerical data, and tweet features
        # 2* - because both the symbol and tweet after non-linear projection have the same size
        d_model = num_numerical_cols + 2*embedding_dim

        self.transformer = nn.Transformer(d_model, n_head, num_encoder_layers, dropout=dropout)
        self.output_linear = nn.Linear(d_model, N_DIM_OUT)  # outputs to one single dim

    def forward(self, src_data, src_symbol, src_tweet, tgt_data, tgt_symbol, tgt_tweet):

        src_symbol = self.symbol_embedding(src_symbol)
        tgt_symbol = self.symbol_embedding(tgt_symbol)

        src_tweet = self.tweet_processor(src_tweet)
        tgt_tweet = self.tweet_processor(tgt_tweet)

        src = torch.cat([src_data, src_symbol, src_tweet], dim=-1)
        # src = src.permute(1, 0, 2)  # Adjust for transformer encoder
        tgt = torch.cat([tgt_data, tgt_symbol, tgt_tweet], dim=-1)
        # tgt = tgt.permute(1, 0, 2)  # Adjust for transformer encoder

        # rawdog goes here - good luck :)
        output = self.transformer(src, tgt)
        # predict just one value - which we'll target/compare to the Close column
        # output = output.permute(0, 1)
        output = self.output_linear(output)

        return output
