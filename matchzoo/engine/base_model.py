"""Contains the base Model class, from which all models inherit.
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import abc
from keras.models import load_model


class BaseModel(object):
    """Abstract base model class.

    # Properties:
        name: str, name of the model.
        task_type: str, model task type, ranking or classification.
        trainable: boolean, indicates whether the model is allowed to train. 
        fixed_hyper_parameters: dict, fixed hyper parameters with values.
        default_hyper_parameters: dict, universal hyper parameters.
        model_specific_hyper_parameters: dict, hyper parameters w.r.t models.
        user_given_parameters: dict, hyper parameters given by users.
        num_hidden_layers: int, default number hidden layer in paper.
        load(model_dir, custom_loss=None)

    # Methods:
        compile(**kwargs)
        train(text_1, text_2, labels)
        load()

    # Internal Methods:
        _aggregate_hyper_parameters()
        _build()
    """

    def __init__(self, **kwargs):
        """Initialization"""
        self._name = 'BaseModel'
        self._task_type = None
        self._trainable = True
        #  This list serve as a "BLACK LIST", not allowed to changed.
        self._list_fixed_hyper_parameters = []
        self._fixed_hyper_parameters = {}
        # Universal model parameters, can be overwrite if not fixed.
        self._default_hyper_parameters = {
            'learning_rate': 0.01,
            'decay': 1e-6,
            'batch_size': 256,
            'num_epochs': 20,
            'train_test_split': 0.2,
            'verbose': 1,
            'shuffle': True
        }
        # Model specific hyper parameters, can be overwritten by
        # _user_given_parameters if not fixed.
        self._model_specific_hyper_parameters = {}
        # User given hyper parameters.
        # THE Intersection between default_hyper_parameters and
        # _model_specific_hyper_parameters will be overrite by
        # _user_given_parameters if it's not fixed
        self._user_given_parameters = {}
        for key, value in kwargs.items():
            if key not in self._list_fixed_hyper_parameters:
                self._user_given_parameters[key] = value
        # Number of default hidden layers.
        # E.g. DSSM is a 3 layer network, user can custom a 5 layer
        #   DSSM with _num_hidden_layers.
        self._num_hidden_layers = None

    @property
    def name(self):
        """Get model name."""
        return self._name

    @property
    def task_type(self):
        """Model task_type, classification, ranking or both."""
        return self._task_type

    @task_type.setter
    def task_type(self, value):
        """Set model task_type."""
        if value not in ['ranking', 'classification']:
            raise ValueError('{} is not a valid model task_type'.format(
                value))
        self._task_type = value

    @property
    def trainable(self):
        """Indicate allow to train or not."""
        return self._trainable

    @property
    def fixed_hyper_parameters(self):
        """Get fixed hyper parameters."""
        return self._fixed_hyper_parameters

    @fixed_hyper_parameters.setter
    def fixed_hyper_parameters(self, config):
        """Set fixed hyper parameters"""
        for key, value in config.items():
            self._fixed_hyper_parameters[key] = value
            self._list_fixed_hyper_parameters.append(key)

    @property
    def default_hyper_parameters(self):
        """Universal parameters that can be use across varies models.
        """
        return self._default_hyper_parameters

    @default_hyper_parameters.setter
    def default_hyper_parameters(self, config):
        """Set default hyper parameters."""
        allowed_default_parameters = self._default_hyper_parameters.keys()
        for key, value in config.items():
            if key not in allowed_default_parameters:
                raise ValueError(
                    '{} not in allowed default parameters: {}.'.format(
                        key,
                        allowed_default_parameters))
            self._default_hyper_parameters[key] = value

    @property
    def model_specific_hyper_parameters(self):
        """Get model specific hyper parameters."""
        return self._model_specific_hyper_parameters

    @model_specific_hyper_parameters.setter
    def model_specific_hyper_parameters(self, config):
        """Set model specific hyper parameters."""
        self._model_specific_hyper_parameters[key] = value

    @property
    def user_given_parameters(self):
        """Get user given hyper parameters."""
        return self._user_given_parameters

    def _aggregate_hyper_parameters(self):
        """This method is used to merge all the parameters.

        This function is used internaly.`

        # Returns:
            conf: final model  configuration.
        """
        conf = self._default_hyper_parameters.copy()
        # Universal config will be overwrite by model_specific.
        # Then merge.
        conf.update(self._model_specific_hyper_parameters)
        # Merged universal config and model config will be.
        # overriten by user_given_parameters.
        conf.update(self._user_given_parameters)
        # Filter fixed-parameters from configuration.
        [conf.pop(key, None) for key in self._list_fixed_hyper_parameters]
        # Merge default fixed parameters.
        conf.update(self._fixed_hyper_parameters)
        return conf

    @property
    def num_hidden_layers(self):
        """Get number of hiddden layers."""
        return self._num_hidden_layers

    @num_hidden_layers.setter
    def num_hidden_layers(self, value):
        """Set number of hidden layers."""
        self._num_hidden_layers = value

    @abc.abstractmethod
    def _build(self):
        """Build model, each sub class need to impelemnt this method.

        This function is used internally with `self._build()`.
        """
        return

    @abc.abstractmethod
    def compile(self, **kwargs):
        """Compile model, each  sub class need to implement this method.

        This function is used internally with `self._compile()`.

        # Arguments:
            **kwargs: Trainable parameters.
        """
        return

    @abc.abstractmethod
    def train(self, text_1, text_2, labels):
        """Train MatchZoo model, each sub class need to implement this method.

        # Arguments:
            text_1: list of text to be trained, left node.
            text_2: list of text to be trained, right node.
            labels: ground truth.
        """
        return

    def load(self, model_dir, custom_loss=None):
        """Load keras model by dir.

        If Keras model was trained with custom loss function use custom_loss.

        # Arguments:
            model_dir: Model directory.
            custom_loss: custom loss function, if required, expect dict as input,
                         where dict key is loss name, value is custom loss function.

        # Returns:
            model: Keras model instance.
        """
        if not custom_loss:
            return load_model(model_dir)
        else:
            if not isinstance(custom_loss, dict):
                raise TypeError(
                    "Custom loss should be a dict, get {}".format(
                        type(custom_loss)))
            return load_model(model_dir,
                              custom_objects=custom_loss)
