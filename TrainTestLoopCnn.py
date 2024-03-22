import numpy as np
import pandas as pd
# from ModelCnn import ModelCnn
from ModelCnnNoTweets import ModelCnnNoTweets as ModelCnn
from Data import COLUMN_SYMBOL_CODE, COLUMN_TEXT_EMBEDDINGS, COLUMN_SYMBOL, COLUMN_DATE, COLUMN_CLOSE
from torch import tensor, nn, optim, no_grad
import torch
from sklearn.metrics import cohen_kappa_score, f1_score


PRINT_MODULO = 10
# MIN_TRAIN_SAMPLES = 150
# MIN_TRAIN_SAMPLES = 77
MIN_TRAIN_SAMPLES = 10
J_DEV_BOOST = .81
MIN_EPOCH_DEV = 4


class TrainTestLoopCnn(ModelCnn):

    # def __init__(self, batch_size: int = 16, input_seq_len: int = 10, output_seq_len: int = 5,
    def __init__(self, batch_size: int = 32, input_seq_len: int = 10, output_seq_len: int = 1,
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
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.trained_symbols: set = {-1, -2}
        self.dev_accuracies: dict = {}

        self.losses_train: list[float] = []
        self.losses_dev: list[float] = []
        self.accuracies_train: list[float] = []
        self.accuracies_dev: list[float] = []
        self.cks_train: list[float] = []
        self.cks_dev: list[float] = []
        self.f1s_train: list[float] = []
        self.f1s_dev: list[float] = []

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
    def shuffle_batch(t1: torch.tensor, t2: torch.tensor, t3: torch.tensor) -> tuple:
        return t1, t2, t3

        # # Generate a random permutation of indices for the first dimension
        # indices = torch.randperm(t1.size(0))
        #
        # # Apply this permutation to both tensors to shuffle them along the same dimension
        # t1 = t1[indices]
        # t2 = t2[indices]
        # t3 = t3[indices]
        #
        # return t1, t2, t3

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
            tgt_symbol.append(df[COLUMN_SYMBOL_CODE].iloc[s_end:t_end].values)

            tgt_data.append(df[self.columns_to_normalize_t].iloc[s_end:t_end][COLUMN_CLOSE+'_t'].values >
                            df[self.columns_to_normalize_t].iloc[s_end-1:s_end][COLUMN_CLOSE+'_t'].values)

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
            loss, train_losses, r2s, accuracies, cks, f1s = None, [], [], [], [], []
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

                    src_data, src_symbol, src_tweet = self.shuffle_batch(src_data, src_symbol, src_tweet)
                    output = self.model(src_data, src_symbol, src_tweet)
                    loss = self.criterion(output, tgt_data)
                    train_losses.append(loss.cpu().item())

                    output = output > .5
                    accuracy = ((output == tgt_data).squeeze().sum() / len(output)).cpu().item()
                    accuracies.append(accuracy)

                    ck = cohen_kappa_score(tgt_data.cpu().numpy(), output.cpu().numpy().astype(float))
                    ck = 0. if np.isnan(ck) else ck
                    cks.append(ck)
                    f1 = f1_score(tgt_data.cpu().numpy(), output.cpu().numpy().astype(float))
                    f1s.append(f1)

                    if i and not ((i // self.batch_size) % PRINT_MODULO):
                        print(f'Symbol: {symbol}, {k}/{len(symbols)} :: Loss: {round(loss.item(), 5)}, '
                              f'Acc: {round(accuracy, 3)}%, '
                              f"Cohen's Kappa: {ck}, "
                              f"F1 score: {f1}, "
                              f'n_samples: {num_samples}')
                    self.optimizer.zero_grad(), loss.backward(), self.optimizer.step()

                    # # TEMP
                    # val_loss = self.validation_loss(epoch+1)
                    # print(val_loss)

            print(f'\ncalculating validation loss...\n')
            self.validation_loss(epoch+1)
            self.losses_train.append(np.mean(train_losses))
            self.accuracies_train.append(np.mean(accuracies))
            self.cks_train.append(np.mean(cks))
            self.f1s_train.append(np.mean(f1s))
            print(f"Epoch {epoch + 1}/{self.epochs} \n"
                  f"Mean training loss: {np.mean(train_losses).round(5)}, "
                  f"Median training loss: {np.median(train_losses).round(5)} \n"
                  f"Mean training accuracy: {np.mean(accuracies).round(3)}%, "
                  f"Median training accuracy: {np.median(accuracies).round(3)}%\n"
                  f"Mean training Cohen's Kappa: {np.mean(cks).round(3)}\n"
                  f"Mean training F1 score: {np.mean(f1s).round(3)}"
                  )
            print(f'=============\n\n')

    def validation_loss(self, i_epoch: int) -> np.array:
        self.model.eval()
        with no_grad():
            symbols: list = self.df[COLUMN_SYMBOL].unique().tolist()
            losses, r2s, accuracies, cks, f1s = [], [], [], [], []
            self.df_test = self.shuffle(self.df_test)
            for k, symbol in enumerate(symbols):
                acc_symbol, losses_symbol, cks_symbol, f1s_symbol = [], [], [], []
                df = self.df_test[self.df_test[COLUMN_SYMBOL] == symbol].reset_index(drop=True)
                assert df[COLUMN_DATE].is_monotonic_increasing
                num_samples = self.get_n_samples(df)
                if (symbol not in self.trained_symbols) or (num_samples < self.batch_size):
                    continue

                for i in range(0, num_samples, self.batch_size):
                    src_data, tgt_data, src_symbol, tgt_symbol, src_tweet, tgt_tweet = \
                        self.generate_batch(df, num_samples, i)

                    src_data, src_symbol, src_tweet = self.shuffle_batch(src_data, src_symbol, src_tweet)
                    output = self.model(src_data, src_symbol, src_tweet)
                    loss = self.criterion(output, tgt_data)
                    # losses.append(loss.cpu().item())
                    losses_symbol.append(loss.cpu().item())

                    output = output > .5
                    accuracy = ((output == tgt_data).squeeze().sum() / len(output)).cpu().item()
                    # accuracies.append(accuracy)
                    acc_symbol.append(accuracy)

                    ck = cohen_kappa_score(tgt_data.cpu().numpy(), output.cpu().numpy().astype(float))
                    ck = 0. if np.isnan(ck) else ck
                    cks_symbol.append(ck)
                    f1 = f1_score(tgt_data.cpu().numpy(), output.cpu().numpy().astype(float))
                    f1s_symbol.append(f1)

                    if i and not ((i // self.batch_size) % 3):
                        print(f'Validation :: Symbol: {symbol}, {k}/{len(symbols)} :: '
                              f'Loss: {round(loss.item(), 5)}, Acc: {round(accuracy, 3)}%, '
                              f"Cohen's Kappa: {ck}, "
                              f"F1 score: {f1}, "
                              f'n_samples: {num_samples}')

                self.dev_accuracies[symbol] = np.mean(acc_symbol)
                if self.is_symbol_in_top_j(i_epoch, symbol):
                    accuracies.extend(acc_symbol)
                    losses.extend(losses_symbol)
                    cks.extend(cks_symbol)
                    f1s.extend(f1s_symbol)

        self.losses_dev.append(np.mean(losses))
        self.accuracies_dev.append(np.mean(accuracies))
        self.cks_dev.append(np.mean(cks))
        self.f1s_dev.append(np.mean(f1s))
        print(f'\nValidation mean loss: {np.mean(losses).round(5)}, '
              f'Validation median loss: {np.median(losses).round(5)} \n'
              f'Validation mean accuracy: {np.mean(accuracies).round(3)}%, '
              f'Validation median accuracy: {np.median(accuracies).round(3)}%\n'
              f"Validation mean Cohen's Kappa: {np.mean(cks).round(3)}\n"
              f"Validation mean F1 score: {np.mean(f1s).round(3)}"
              )

        return np.mean(losses)

    def is_symbol_in_top_j(self, epoch: int, symbol: str, min_epoch: int = MIN_EPOCH_DEV):
        """
        Check if the symbol is in the top j symbols based on their accuracy.

        Parameters:
        - symbols_dict: Dict[str, float] - A dictionary of symbols and their accuracies.
        - num_epochs: int - Total number of epochs.
        - epoch: int - Current epoch.
        - symbol: str - The symbol to check.
        - min_epoch: int - The minimum epoch before adjusting j.

        Returns:
        - bool: True if the symbol is in the top j symbols, False otherwise.
        """
        num_epochs = self.epochs
        symbols_dict = self.dev_accuracies
        # Ensure the function behaves correctly when the epoch is less than or equal to min_epoch
        if epoch <= min_epoch:
            j = len(symbols_dict)  # Use the full length of the dictionary
        else:
            # Calculate the decreasing factor based on the progress through the epochs
            # Ensure that j does not decrease to less than half the length of the dictionary
            decreasing_factor = 1 - (num_epochs - epoch) / num_epochs
            max_decrease = len(symbols_dict) / 2
            j = int((len(symbols_dict) - decreasing_factor*max_decrease) * J_DEV_BOOST)
            # print(f'total validation symbols: {len(symbols_dict)}, to be used: {j}')

        # Order the dictionary by accuracy in descending order
        ordered_symbols = sorted(symbols_dict, key=symbols_dict.get, reverse=True)

        # Check if the symbol is in the top j ordered symbols
        return symbol in ordered_symbols[:j]
