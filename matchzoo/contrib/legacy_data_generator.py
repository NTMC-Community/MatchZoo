import numpy as np

import matchzoo as mz


def print_deprecation_warning(instance):
    name = instance.__class__.__name__
    print(f"WARNING: {name} will be deprecated in MatchZoo v2.2. "
          "Use `DataGenerator` with callbacks instead.")


class HistogramDataGenerator(mz.DataGenerator):
    def __init__(
        self,
        data_pack: mz.DataPack,
        embedding_matrix: np.ndarray,
        bin_size: int = 30,
        hist_mode: str = 'CH',
        batch_size: int = 32,
        shuffle: bool = True
    ):
        super().__init__(
            data_pack=data_pack,
            batch_size=batch_size,
            shuffle=shuffle,
            callbacks=[
                mz.data_generator.callbacks.Histogram(
                    embedding_matrix=embedding_matrix,
                    bin_size=bin_size,
                    hist_mode=hist_mode
                )
            ]
        )
        print_deprecation_warning(self)


class HistogramPairDataGenerator(mz.DataGenerator):
    def __init__(
        self,
        data_pack: mz.DataPack,
        embedding_matrix: np.ndarray,
        bin_size: int = 30,
        hist_mode: str = 'CH',
        num_dup: int = 1,
        num_neg: int = 1,
        batch_size: int = 32,
        shuffle: bool = True
    ):
        super().__init__(
            data_pack=data_pack,
            mode='pair',
            num_dup=num_dup,
            num_neg=num_neg,
            batch_size=batch_size,
            shuffle=shuffle,
            callbacks=[
                mz.data_generator.callbacks.Histogram(
                    embedding_matrix=embedding_matrix,
                    bin_size=bin_size,
                    hist_mode=hist_mode
                )
            ]
        )
        print_deprecation_warning(self)


class DPoolDataGenerator(mz.DataGenerator):
    def __init__(
        self,
        data_pack: mz.DataPack,
        fixed_length_left: int,
        fixed_length_right: int,
        compress_ratio_left: float = 1,
        compress_ratio_right: float = 1,
        batch_size: int = 32,
        shuffle: bool = True
    ):
        super().__init__(
            data_pack=data_pack,
            shuffle=shuffle,
            batch_size=batch_size,
            callbacks=[
                mz.data_generator.callbacks.DynamicPooling(
                    fixed_length_left=fixed_length_left,
                    fixed_length_right=fixed_length_right,
                    compress_ratio_left=compress_ratio_left,
                    compress_ratio_right=compress_ratio_right
                )
            ]
        )
        print_deprecation_warning(self)


class DPoolPairDataGenerator(mz.DataGenerator):
    def __init__(
        self,
        data_pack: mz.DataPack,
        fixed_length_left: int,
        fixed_length_right: int,
        compress_ratio_left: float = 1,
        compress_ratio_right: float = 1,
        num_dup: int = 1,
        num_neg: int = 1,
        batch_size: int = 32,
        shuffle: bool = True
    ):
        super().__init__(
            data_pack=data_pack,
            mode='pair',
            num_dup=num_dup,
            num_neg=num_neg,
            batch_size=batch_size,
            shuffle=shuffle,
            callbacks=[
                mz.data_generator.callbacks.DynamicPooling(
                    fixed_length_left=fixed_length_left,
                    fixed_length_right=fixed_length_right,
                    compress_ratio_left=compress_ratio_left,
                    compress_ratio_right=compress_ratio_right
                )
            ]
        )
        print_deprecation_warning(self)


class PairDataGenerator(mz.DataGenerator):
    def __init__(
        self,
        data_pack: mz.DataPack,
        num_dup: int = 1,
        num_neg: int = 1,
        batch_size: int = 32,
        shuffle: bool = True
    ):
        super().__init__(
            data_pack=data_pack,
            mode='pair',
            num_dup=num_dup,
            num_neg=num_neg,
            batch_size=batch_size,
            shuffle=shuffle,

        )
        print_deprecation_warning(self)


class DynamicDataGenerator(mz.DataGenerator):
    def __init__(self, func, *args, **kwargs):
        super().__init__(*args, **kwargs)
        callback = mz.data_generator.callbacks.LambdaCallback(
            on_batch_data_pack=func)
        self.callbacks.append(callback)
        print_deprecation_warning(self)
