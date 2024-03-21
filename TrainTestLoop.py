import numpy as np
import pandas as pd
from ModelWrap import ModelWrap
from Data import COLUMN_SYMBOL_CODE, COLUMN_TEXT_EMBEDDINGS, COLUMN_SYMBOL, COLUMN_DATE
from torch import tensor, nn, optim, no_grad
import torch

# from torchmetrics import R2Score

PRINT_MODULO = 10
# MIN_TRAIN_SAMPLES = 150
# MIN_TRAIN_SAMPLES = 77
MIN_TRAIN_SAMPLES = 10


class TrainTestLoop(ModelWrap):

    # def __init__(self, batch_size: int = 16, input_seq_len: int = 10, output_seq_len: int = 5,
    def __init__(self, batch_size: int = 32, input_seq_len: int = 10, output_seq_len: int = 2,
                 epochs: int = 20, lr: float = 1e-3):
        super().__init__()
        self.batch_size = batch_size
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.epochs = epochs
        self.num_samples = len(self.df) - input_seq_len - output_seq_len + 1
        self.num_samples_test = len(self.df_test) - input_seq_len - output_seq_len + 1
        print(f'train samples count: {self.num_samples}')
        print(f'test samples count: {self.num_samples_test}')

        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        # self.r2score = R2Score().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.trained_symbols: set = {-1, -2}

        print(f'batch: {batch_size}')

    def get_n_samples(self, df: pd.DataFrame) -> int:
        n_samples = len(df) - self.input_seq_len - self.output_seq_len + 1
        return n_samples

    @staticmethod
    def shuffle(df: pd.DataFrame) -> pd.DataFrame:
        print(f'shuffling {df.shape[0]} rows')
        # Group by 'Symbol' while preserving the order within each group
        grouped = [group for _, group in df.groupby(COLUMN_SYMBOL)]

        # Shuffle the list of groups
        np.random.shuffle(grouped)

        # Concatenate the shuffled groups
        shuffled_df = pd.concat(grouped).reset_index(drop=True)

        print(f'shuffled {shuffled_df.shape[0]} rows')
        return shuffled_df

    @staticmethod
    def calc_r2(mse: float, y: tensor) -> float:
        # print(f'mse: {mse}, yvar: {y.var()}')
        return 1. - mse / y.cpu().var()

    def generate_batch(self, df: pd.DataFrame, num_samples: int, start_index: int) -> tuple:
        """
        Generate a batch of data for training.
        """
        # df = self.df if not is_test else self.df_test
        # num_samples = self.num_samples if not is_test else self.num_samples_test
        input_seq_len, output_seq_len = self.input_seq_len, self.output_seq_len

        end_index = start_index + self.batch_size
        src_data, tgt_data = [], []
        src_symbol, tgt_symbol = [], []
        src_tweet, tgt_tweet = [], []

        for i in range(start_index, min(end_index, num_samples)):
            s_end = i + input_seq_len
            t_end = i + input_seq_len + output_seq_len
            src_data.append(df[self.columns_to_normalize_t].iloc[i:s_end].values)
            src_symbol.append(df[COLUMN_SYMBOL_CODE].iloc[i:s_end].values)
            tgt_data.append(df[self.columns_to_normalize_t].iloc[s_end:t_end].values)
            tgt_symbol.append(df[COLUMN_SYMBOL_CODE].iloc[s_end:t_end].values)

            src_tweet.append(
                np.array([np.array(x).astype(float) for x in df[COLUMN_TEXT_EMBEDDINGS].iloc[i:s_end]], dtype=float))
            tgt_tweet.append(
                np.array([np.array(x).astype(float) for x in df[COLUMN_TEXT_EMBEDDINGS].iloc[s_end:t_end]],
                         dtype=float))

        return (
            tensor(np.array(src_data)).float().to(self.device),
            tensor(np.array(tgt_data)).float().to(self.device),
            tensor(np.array(src_symbol)).int().to(self.device),
            tensor(np.array(tgt_symbol)).int().to(self.device),
            tensor(np.array(src_tweet)).float().to(self.device),
            tensor(np.array(tgt_tweet)).float().to(self.device)
        )

    def train(self):

        for epoch in range(self.epochs):
            self.model.train()
            self.df = self.shuffle(self.df)
            symbols: list = self.df[COLUMN_SYMBOL].unique().tolist()
            loss, train_losses, r2s, accuracies = None, [], [], []
            for k, symbol in enumerate(symbols):
                df = self.df[self.df[COLUMN_SYMBOL] == symbol].reset_index(drop=True)
                assert df[COLUMN_DATE].is_monotonic_increasing
                num_samples = self.get_n_samples(df)
                if num_samples < MIN_TRAIN_SAMPLES:
                    continue
                self.trained_symbols.add(symbol)
                for i in range(0, num_samples, self.batch_size):
                    # def generate_batch(self, df: pd.DataFrame, num_samples: int, start_index: int) -> tuple:
                    src_data, tgt_data, src_symbol, tgt_symbol, src_tweet, tgt_tweet = \
                        self.generate_batch(df, num_samples, i)
                    src_data, tgt_data, src_symbol, tgt_symbol, src_tweet, tgt_tweet = (
                        src_data.transpose(0, 1), tgt_data.transpose(0, 1),
                        src_symbol.transpose(0, 1), tgt_symbol.transpose(0, 1),
                        src_tweet.transpose(0, 1), tgt_tweet.transpose(0, 1)
                    )

                    # we have symbols, but no tweets
                    tgt_tweet.fill_(0.)  # no tweets for future
                    assert not tgt_tweet.sum()

                    output = self.model(src_data, src_symbol, src_tweet, tgt_data[:-1], tgt_symbol[:-1], tgt_tweet[:-1])
                    # loss = self.criterion(output, tgt_data[1:, :, 3:4])
                    loss = self.criterion(output, tgt_data[1:])
                    train_losses.append(loss.cpu().item())
                    r2 = self.calc_r2(loss.item(), tgt_data[1:])
                    r2s.append(r2)

                    accuracy = self.calculate_accuracy(tgt_data[:1], tgt_data[1:], output)
                    accuracies.append(accuracy)

                    if i and not ((i // self.batch_size) % PRINT_MODULO):
                        # r2 = self.r2score(output.flatten(), tgt_data[1:].flatten())
                        print(f'Symbol: {symbol}, {k}/{len(symbols)} :: Loss: {loss.item()}, R2: {r2}, '
                              f'Acc: {round(accuracy, 2)}%, '
                              f'n_samples: {num_samples}')
                    self.optimizer.zero_grad(), loss.backward(), self.optimizer.step()

                    # # TEMP
                    # val_loss = self.validation_loss()
                    # print(val_loss)

            print(f'\ncalculating validation loss...\n')
            self.validation_loss()
            print(f"Epoch {epoch + 1}/{self.epochs} \n"
                  f"Mean training loss: {np.mean(train_losses).round(5)}, "
                  f"Median training loss: {np.median(train_losses).round(5)} \n"
                  f"Mean training r2: {np.mean(r2s)}, Median training r2: {np.median(r2s).round(5)} \n"
                  f"Mean training accuracy: {np.mean(accuracies).round(2)}%, "
                  f"Median training accuracy: {np.median(accuracies).round(2)}%"
                  )
            print(f'=============\n\n')

    def validation_loss(self) -> np.array:
        """
        Perform autoregressive inference for validation.
        """
        self.model.eval()
        with no_grad():
            symbols: list = self.df[COLUMN_SYMBOL].unique().tolist()
            losses, r2s, accuracies = [], [], []
            self.df_test = self.shuffle(self.df_test)
            for k, symbol in enumerate(symbols):
                df = self.df_test[self.df_test[COLUMN_SYMBOL] == symbol].reset_index(drop=True)
                assert df[COLUMN_DATE].is_monotonic_increasing
                num_samples = self.get_n_samples(df)
                # if num_samples < 1:
                # if num_samples < 50:
                #     continue
                if symbol not in self.trained_symbols:
                    continue
                for i in range(0, num_samples, self.batch_size):
                    src_data, tgt_data, src_symbol, tgt_symbol, src_tweet, tgt_tweet = \
                        self.generate_batch(df, num_samples, i)
                    src_data, tgt_data, src_symbol, tgt_symbol, src_tweet, tgt_tweet = (
                        src_data.transpose(0, 1), tgt_data.transpose(0, 1),
                        src_symbol.transpose(0, 1), tgt_symbol.transpose(0, 1),
                        src_tweet.transpose(0, 1), tgt_tweet.transpose(0, 1)
                    )

                    tgt_data_orig = tgt_data.clone()
                    tgt_data.fill_(0)
                    # tgt_symbol.fill_(self.unk_code)  # we know which symbols we want to predict
                    tgt_tweet.fill_(0)  # but we don't have tweets for the future
                    assert tgt_data_orig.sum()
                    assert not tgt_data.sum()
                    assert not tgt_tweet.sum()

                    for j in range(self.output_seq_len):
                        output = self.model(src_data, src_symbol, src_tweet,
                                            tgt_data[:j + 1], tgt_symbol[:j + 1], tgt_tweet[:j + 1])
                        tgt_data[j] = output[-1]
                    loss = self.criterion(tgt_data, tgt_data_orig)  # !
                    losses.append(loss.cpu().item())
                    r2 = self.calc_r2(loss.item(), tgt_data_orig)  # !
                    r2s.append(r2)

                    # accuracy = self.calculate_accuracy(tgt_data_orig[:1], tgt_data_orig[1:], tgt_data[1:])
                    accuracy = self.calculate_accuracy(src_data[-1:], tgt_data_orig[:1], tgt_data[:1])
                    accuracies.append(accuracy)

                    if i and not ((i // self.batch_size) % 3):
                        print(f'Validation :: Symbol: {symbol}, {k}/{len(symbols)} :: '
                              f'Loss: {loss.item()}, R2 {r2}, Acc: {round(accuracy, 2)}%, '
                              f'n_samples: {num_samples}')

        print(f'\nValidation mean loss: {np.mean(losses).round(5)}, '
              f'Validation median loss: {np.median(losses).round(5)} \n'
              f'Validation mean r2: {np.mean(r2s).round(5)}, Validation median r2: {np.median(r2s)} \n'
              f'Validation mean accuracy: {np.mean(accuracies).round(2)}%, '
              f'Validation median accuracy: {np.median(accuracies).round(2)}% '
              )

        return np.mean(losses)

    @staticmethod
    def calculate_accuracy(src1, src2, predicted) -> float:
        diff_source_target = src1[..., 3] - src2[..., 3]
        diff_source_predicted = src1[..., 3] - predicted[..., 3]
        sign_source_target = torch.sign(diff_source_target)
        sign_source_predicted = torch.sign(diff_source_predicted)
        sign_source_predicted.mul_(-1)
        # matching_signs = (sign_source_target == sign_source_predicted) & (sign_source_target != 0)
        matching_signs = sign_source_target == sign_source_predicted

        accuracy = matching_signs.float().mean().cpu().item()
        return accuracy * 100
