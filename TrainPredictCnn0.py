from TrainTestLoopCnn import TrainTestLoopCnn


class TrainPredictCnn(TrainTestLoopCnn):
    # def __init__(self, batch_size: int = 128, input_seq_len: int = 10, output_seq_len: int = 5,
    # lr 1e-3
    def __init__(self, batch_size: int = 32, input_seq_len: int = 10, output_seq_len: int = 1,
                 epochs: int = 20, lr: float = 1e-3):
        super().__init__(batch_size, input_seq_len, output_seq_len, epochs, lr)

    def run_train(self):
        self.train()


if __name__ == '__main__':
    with TrainPredictCnn() as obj:
        obj.run_train()
