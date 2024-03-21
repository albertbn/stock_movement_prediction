from Data import (Data, COLUMN_OPEN, COLUMN_HIGH, COLUMN_LOW, COLUMN_CLOSE, COLUMN_ADJ_CLOSE, COLUMN_VOLUME,
                  COLUMN_DATE, pd, COLUMN_TEXT_EMBEDDINGS)
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA


class DataScale(Data):

    def __init__(self):
        super().__init__()
        self.min_vals, self.min_vals_test = None, None
        self.range_vals, self.range_vals_test = None, None

        # Select columns to normalize (excluding 'Symbol', 'Date')
        self.columns_to_normalize = [COLUMN_OPEN, COLUMN_HIGH, COLUMN_LOW,
                                     COLUMN_CLOSE, COLUMN_ADJ_CLOSE, COLUMN_VOLUME]
        self.columns_to_normalize_t = [f'{o}_t' for o in self.columns_to_normalize]

        # train: 17051/21990 78%, test: 4939/21990 22%
        # train samples count: 17037
        # test samples count: 4925
        # batch: 128
        self.date_boundary = datetime.strptime('2015-10-06', "%Y-%m-%d").date()

        self.init_data()

    def init_data(self):
        self.load_data()
        self.aggregate_data()
        self.symbol_categories()
        # DONE - ensure Date is a date object. Split to train and dev set - dev where date >= some date
        #  that splits the data 80/20 % or 85/15 %
        self.data_split()
        self.scale()

        self.reduce_embedding_dim()

    def reduce_embedding_dim(self):
        embeddings_matrix = pd.DataFrame(self.df[COLUMN_TEXT_EMBEDDINGS].tolist())
        # It's often a good practice to standardize the features before applying PCA
        scaler = StandardScaler()
        scaler = scaler.fit(embeddings_matrix)
        embeddings_standardized = scaler.transform(embeddings_matrix)
        pca = PCA(n_components=16)
        pca = pca.fit(embeddings_standardized)
        reduced_embeddings = pca.transform(embeddings_standardized)
        self.df[COLUMN_TEXT_EMBEDDINGS] = list(reduced_embeddings)

        embeddings_matrix_test = pd.DataFrame(self.df_test[COLUMN_TEXT_EMBEDDINGS].tolist())
        embeddings_standardized_test = scaler.transform(embeddings_matrix_test)
        reduced_embeddings_test = pca.transform(embeddings_standardized_test)
        self.df_test[COLUMN_TEXT_EMBEDDINGS] = list(reduced_embeddings_test)

    def data_split(self):
        where = self.df[COLUMN_DATE] <= self.date_boundary
        df_train: pd.DataFrame = self.df[where].copy().reset_index(drop=True)
        self.df_test: pd.DataFrame = self.df[~where].copy().reset_index(drop=True)
        self.df = df_train
        train_count, test_count = self.df.shape[0], self.df_test.shape[0]
        total = train_count + test_count
        # 78%, 22%
        percent_train, percent_test = round(train_count/total*100), round(test_count/total*100)
        print(f'train: {train_count}/{total} {percent_train}%, test: {test_count}/{total} {percent_test}%')
        print()

    def scale(self):
        # scaler = MinMaxScaler()
        scaler = StandardScaler()
        scaler = scaler.fit(self.df[self.columns_to_normalize])

        self.df[self.columns_to_normalize_t] = scaler.transform(self.df[self.columns_to_normalize])
        self.df_test[self.columns_to_normalize_t] = scaler.transform(self.df_test[self.columns_to_normalize])

        df, df_test = self.df, self.df_test
        # self.df[self.columns_to_normalize_t] = scaler.fit_transform(self.df[self.columns_to_normalize])
        # Store the min and range (max - min) for later use in reversing the scaling
        # self.min_vals = scaler.data_min_
        # self.range_vals = scaler.data_max_ - scaler.data_min_

        # scaler_test = MinMaxScaler()
        # # self.df_test[self.columns_to_normalize_t] = scaler_test.fit_transform(self.df_test[self.columns_to_normalize])
        # self.df_test[self.columns_to_normalize_t] = scaler.fit_transform(self.df_test[self.columns_to_normalize])
        # # Store the min and range (max - min) for later use in reversing the scaling
        # self.min_vals_test = scaler_test.data_min_
        # self.range_vals_test = scaler_test.data_max_ - scaler_test.data_min_

    def reverse_normalize(self, scaled_value, column_index, is_test=1):
        """
        Reverses the normalization for a single value in a specified column.
        :param scaled_value: The normalized value to reverse.
        :param column_index: The index of the column in `columns_to_normalize`.
        :param is_test: If it's the test set
        :return: The original scale value.
        """
        range_vals = self.range_vals if not is_test else self.range_vals_test
        min_vals = self.min_vals if not is_test else self.min_vals_test
        original_value = (scaled_value * range_vals[column_index]) + min_vals[column_index]
        return original_value

    @staticmethod
    def check_epsilon_deviation(original_value, predicted_value, epsilon=1e-5):
        """
        Checks if the original and predicted values match within a specified epsilon deviation.
        :param original_value: The original value before normalization.
        :param predicted_value: The predicted value to compare, after reversing normalization.
        :param epsilon: The allowed deviation between the original and predicted values.
        :return: True if the deviation is within epsilon, False otherwise.
        """
        return abs(original_value - predicted_value) <= epsilon
