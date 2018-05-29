import attrdict

import matchzoo


class ModelParams(attrdict.AttrDict):
    def __init__(self):
        super().__init__(
                name='default_model_name',
                text_1_max_len=30,
                text_2_max_len=30,
                trainable=False,
                task=matchzoo.engine.task.Classification,
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['acc']
        )