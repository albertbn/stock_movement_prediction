import os
import traceback
import pandas as pd
import numpy as np
import json

FILE_PATH = 'yumoxu_text_embeddings.feather'
DATA_URL = f"https://getfiles.adcore.com/img/{FILE_PATH}"
COLUMN_TEXT, COLUMN_SYMBOL, COLUMN_DATE, COLUMN_SYMBOL_CODE = 'Text', 'Symbol', 'Date', 'Symbol_code'
COLUMN_OPEN, COLUMN_HIGH, COLUMN_LOW, COLUMN_CLOSE, COLUMN_ADJ_CLOSE, COLUMN_VOLUME, COLUMN_TEXT_EMBEDDINGS = \
    ('Open', 'High', 'Low', 'Close',  'Adj Close', 'Volume', 'text_embeddings')
FIRST, UNK = 'first', 'UNK'
FILE_SYMBOL_TO_CODE = 'symbol_to_code.json'


class Data:
    def __init__(self):
        super().__init__()
        self.df: pd.DataFrame = pd.DataFrame()
        self.df_test: pd.DataFrame = pd.DataFrame()
        self.symbol_to_code: dict = {}
        self.unk_code: int = -1

    def load_data(self):
        if not os.path.exists(FILE_PATH):
            print(f'downloading 800 MB feather from {DATA_URL}')
            command = f'wget {DATA_URL}'
            os.system(command)
        df = pd.read_feather(FILE_PATH)
        del df[COLUMN_TEXT]
        df[COLUMN_DATE] = pd.to_datetime(df[COLUMN_DATE], format='%Y-%m-%d').dt.date
        self.df = df

    def aggregate_data(self):
        gb = self.df.groupby([COLUMN_SYMBOL, COLUMN_DATE])
        aggregation_functions = {
            COLUMN_SYMBOL: FIRST,
            COLUMN_DATE: FIRST,
            COLUMN_OPEN: FIRST,
            COLUMN_HIGH: FIRST,
            COLUMN_LOW: FIRST,
            COLUMN_CLOSE: FIRST,
            COLUMN_ADJ_CLOSE: FIRST,
            COLUMN_VOLUME: FIRST,
            # Vertically stack the text_embeddings arrays
            # COLUMN_TEXT_EMBEDDINGS: lambda x: np.vstack(x)
            # COLUMN_TEXT_EMBEDDINGS: lambda x: np.mean(np.vstack(x), axis=0)  # aggregate/mean the variable days
            COLUMN_TEXT_EMBEDDINGS: lambda x: np.mean(x, axis=0)  # aggregate/mean the variable days
        }
        # Apply custom aggregation
        # Reset index if you want the GroupBy keys as columns in the resulting DataFrame
        self.df = gb.agg(aggregation_functions).reset_index(drop=True)
        print(self.df.shape)

    def symbol_categories(self):
        df = self.df
        # convert symbol to categorical
        df[COLUMN_SYMBOL] = df[COLUMN_SYMBOL].astype('category')
        # Convert the categorical 'Symbol' to integer codes
        df[COLUMN_SYMBOL_CODE] = df[COLUMN_SYMBOL].cat.codes

        # Convert 'Symbol' to categorical and create a mapping dictionary
        self.symbol_to_code = {symbol: code for code, symbol in enumerate(df[COLUMN_SYMBOL].cat.categories)}
        # Add an 'UNK' (unknown) symbol with a unique code (e.g., max existing code + 1)
        self.unk_code = max(self.symbol_to_code.values()) + 1
        self.symbol_to_code[UNK] = self.unk_code

        # Save the mapping for later use (e.g., during inference)
        with open(FILE_SYMBOL_TO_CODE, 'w') as f:
            json.dump(self.symbol_to_code, f)

    # region boiler
    def __del__(self):
        try:
            del self.df
        except:
            # print(traceback.format_exc())
            traceback.format_exc()
            pass

    def __enter__(self):
        return self

    def __exit__(self, type_, value, traceback_):
        self.__del__()

    # endregion boiler
