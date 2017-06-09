# -*- coding=utf8 -*-
import os
import sys
import tensorflow as tf
from .utils import generic_utils

def _to_snake_case(name):
    intermediate = re.sub('(.)([A-Z][a-z0-9]+)', r'\1_\2', name)
    insecure = re.sub('([a-z])([A-Z])', r'\1_\2', intermediate).lower()
    # If the class is private the name starts with "_" which is not secure
    # for creating scopes. We prefix the name with "private" in this case.
    if insecure[0] != '_':
        return insecure
    return 'private' + insecure
def _to_list(x):
    """Normalizes a list/tensor into a list.

    If a tensor is passed, we return
    a list of size 1 containing the tensor.

    # Arguments
        x: target object to be normalized.

    # Returns
        A list.
    """
    if isinstance(x, list):
        return x
    return [x]

def _object_list_uid(object_list):
    object_list = _to_list(object_list)
    return ', '.join([str(abs(id(x))) for x in object_list])

class Node(object):
    """A `Node` describes the connectivity between two layers.

    Each time a layer is connected to some new input,
    a node is added to `layer.inbound_nodes`.
    Each time the output of a layer is used by another layer,
    a node is added to `layer.outbound_nodes`.

    # Arguments
        outbound_layer: the layer that takes
            `input_tensors` and turns them into `output_tensors`
            (the node gets created when the `call`
            method of the layer was called).
        inbound_layers: a list of layers, the same length as `input_tensors`,
            the layers from where `input_tensors` originate.
        node_indices: a list of integers, the same length as `inbound_layers`.
            `node_indices[i]` is the origin node of `input_tensors[i]`
            (necessary since each inbound layer might have several nodes,
            e.g. if the layer is being shared with a different data stream).
        tensor_indices: a list of integers,
            the same length as `inbound_layers`.
            `tensor_indices[i]` is the index of `input_tensors[i]` within the
            output of the inbound layer
            (necessary since each inbound layer might
            have multiple tensor outputs, with each one being
            independently manipulable).
        input_tensors: list of input tensors.
        output_tensors: list of output tensors.
        input_masks: list of input masks (a mask can be a tensor, or None).
        output_masks: list of output masks (a mask can be a tensor, or None).
        input_shapes: list of input shape tuples.
        output_shapes: list of output shape tuples.
        arguments: dictionary of keyword arguments that were passed to the
            `call` method of the layer at the call that created the node.

    `node_indices` and `tensor_indices` are basically fine-grained coordinates
    describing the origin of the `input_tensors`, verifying the following:

    `input_tensors[i] == inbound_layers[i].inbound_nodes[node_indices[i]].output_tensors[tensor_indices[i]]`

    A node from layer A to layer B is added to:
        A.outbound_nodes
        B.inbound_nodes
    """

    def __init__(self, outbound_layer,
                 inbound_layers, node_indices, tensor_indices,
                 input_tensors, output_tensors,
                 input_masks, output_masks,
                 input_shapes, output_shapes,
                 arguments=None):
        # Layer instance (NOT a list).
        # this is the layer that takes a list of input tensors
        # and turns them into a list of output tensors.
        # the current node will be added to
        # the inbound_nodes of outbound_layer.
        self.outbound_layer = outbound_layer

        # The following 3 properties describe where
        # the input tensors come from: which layers,
        # and for each layer, which node and which
        # tensor output of each node.

        # List of layer instances.
        self.inbound_layers = inbound_layers
        # List of integers, 1:1 mapping with inbound_layers.
        self.node_indices = node_indices
        # List of integers, 1:1 mapping with inbound_layers.
        self.tensor_indices = tensor_indices

        # Following 2 properties:
        # tensor inputs and outputs of outbound_layer.

        # List of tensors. 1:1 mapping with inbound_layers.
        self.input_tensors = input_tensors
        # List of tensors, created by outbound_layer.call().
        self.output_tensors = output_tensors

        # Following 2 properties: input and output masks.
        # List of tensors, 1:1 mapping with input_tensor.
        self.input_masks = input_masks
        # List of tensors, created by outbound_layer.compute_mask().
        self.output_masks = output_masks

        # Following 2 properties: input and output shapes.

        # List of shape tuples, shapes of input_tensors.
        self.input_shapes = input_shapes
        # List of shape tuples, shapes of output_tensors.
        self.output_shapes = output_shapes

        # Optional keyword arguments to layer's `call`.
        self.arguments = arguments

        # Add nodes to all layers involved.
        for layer in inbound_layers:
            if layer is not None:
                layer.outbound_nodes.append(self)
        outbound_layer.inbound_nodes.append(self)

    def get_config(self):
        inbound_names = []
        for layer in self.inbound_layers:
            if layer:
                inbound_names.append(layer.name)
            else:
                inbound_names.append(None)
        return {'outbound_layer': self.outbound_layer.name if self.outbound_layer else None,
                'inbound_layers': inbound_names,
                'node_indices': self.node_indices,
                'tensor_indices': self.tensor_indices}

class Layer(object):
    """Abstract base layer class
    # Properties
        name: String, must be unique within a model.
        input_spec: List of InputSpec class instances
            each entry describes one required input:
                - ndim
                - dtype
            A layer with `n` input tensors must have
            an `input_spec` of length `n`.
        trainable: Boolean, whether the layer weights
            will be updated during training.
        uses_learning_phase: Whether any operation
            of the layer uses `K.in_training_phase()`
            or `K.in_test_phase()`.
        input_shape: Shape tuple. Provided for convenience,
            but note that there may be cases in which this
            attribute is ill-defined (e.g. a shared layer
            with multiple input shapes), in which case
            requesting `input_shape` will raise an Exception.
            Prefer using `layer.get_input_shape_for(input_shape)`,
            or `layer.get_input_shape_at(node_index)`.
        output_shape: Shape tuple. See above.
        inbound_nodes: List of nodes.
        outbound_nodes: List of nodes.
        input, output: Input/output tensor(s). Note that if the layer is used
            more than once (shared layer), this is ill-defined
            and will raise an exception. In such cases, use
            `layer.get_input_at(node_index)`.
        input_mask, output_mask: Same as above, for masks.
        trainable_weights: List of variables.
        non_trainable_weights: List of variables.
        weights: The concatenation of the lists trainable_weights and
            non_trainable_weights (in this order).
        constraints: Dict mapping weights to constraints.

    # Methods
        call(x, mask=None): Where the layer's logic lives.
        __call__(x, mask=None): Wrapper around the layer logic (`call`).
            If x is a Keras tensor:
                - Connect current layer with last layer from tensor:
                    `self._add_inbound_node(last_layer)`
                - Add layer to tensor history
            If layer is not built:
                - Build from x._keras_shape
        get_weights()
        set_weights(weights)
        get_config()
        count_params()
        compute_output_shape(input_shape)
        compute_mask(x, mask)
        get_input_at(node_index)
        get_output_at(node_index)
        get_input_shape_at(node_index)
        get_output_shape_at(node_index)
        get_input_mask_at(node_index)
        get_output_mask_at(node_index)

    # Class Methods
        from_config(config)

    # Internal methods:
        build(input_shape)
        _add_inbound_node(layer, index=0)
        assert_input_compatibility()
    """
    def __init__(self, **kwargs):
        self.input_spec = None
        self.support_masking = False

        #These properties will be set upon call of self.build()
        self._trainable_weights = []
        self._none_trainable_weights = []
        self._constraints = {} # dict { tensor: constraint instance}
        self._losses = []
        self._updates = []
        self._per_input_losses = {}
        self._per_input_updates = {}
        self._build = False

        # These lists will be filled via successive calls
        # to self._add_inbound_node().
        self.inbound_nodes = []
        self.outbound_nodes = []

        # These properties should be set by the user via keyword arguments.
        # note that 'dtype', 'input_shape' and 'batch_input_shape'
        # are only applicable to input layers: do not pass these keywords
        # to non-input layers.
        allowed_kwargs = {'input_shape',
                'batch_input_shape',
                'batch_size',
                'dtype',
                'name',
                'trainable',
                'weights',
                'input_dtype', #legacy
                }
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood:', kwarg)
        name = kwargs.get('name')
        if not name:
            prefix = self.__class__.__name__
            name = _to_snake_case(prefix) + '_' + str(generic_utils.get_uid(prefix))
        self.name = name

        self.trainable = kwargs.get('trainable', True)
        if 'input_shape' in kwargs or 'batch_input_shape' in kwargs:
            # In this case we will later create an input layer
            # to insert before the current layer
            if 'batch_input_shape' in kwargs:
                batch_input_shape = tuple(kwargs['batch_input_shape'])
            elif 'input_shape' in kwargs:
                if 'batch_size' in kwargs:
                    batch_size = kwargs['batch_size']
                else:
                    batch_size = None
                batch_input_shape = (batch_size,) + tuple(kwargs['input_shape'])
            self.batch_input_shape = batch_input_shape

            # Set dtype.
            dtype = kwargs.get('dtype')
            if dtype is None:
                dtype = kwargs.get('input_dtype')
            if dtype is None:
                dtype = tf.float32
            self.dtype = dtype
        if 'weights' in kwargs:
            self._initial_weights = kwargs['weights']
        else:
            self._initial_weights = None

    @property
    def losses(self):
        return self._losses

    @property
    def updates(self):
        return self._updates

    @property
    def built(self):
        return self._built

    @build.setter
    def built(self, value):
        self._built = value

    @property
    def constraints(self):
        return self.__constraints

    @constraints.setter
    def constraints(self, value):
        self._constraints = value

    @property
    def trainable_weights(self):
        trainable = getattr(self, 'trainable', True)
        if trainable:
            return self._trainable_weights
        else:
            return []

    @trainable_weights.setter
    def trainable_weights(self, weights):
        self._trainable_weights = weights

    @property
    def non_trainable_weights(self):
        trainable = getattr(self, 'trainable', True)
        if not non_trainable:
            return self._trainable_weights + self._non_trainable_weights
        else:
            return self._non_trainable_weights

    @non_trainable_weights.setter
    def trainable_weights(self, weights):
        self._non_trainable_weights = weights

    @interfaces.legacy_add_weight_support
    def add_weight(self, name, shape,
            dtype = None,
            initializer = None,
            regularizer = None,
            trainable = True,
            constraint = None):
        """ Adds a weight variable to the layer.
        # Arguments
            name: String, the name for the weight variable.
            shape: The shape tuple of the weight.
            dtype: The dtype of the weight.
            initializer: An Initializer instance (callable).
            regularizer: An optional Regularizer instance.
            trainable: A boolean, whether the weight should
                be trained via backprop or not (assuming
                that the layer itself is also trainable).
            constraint: An optional Constraint instance.

        # Returns
            The created weight variable.
        """
        initializer =  initializers.get(initializer)
        if dtype is None:
            dtype = tf.float32
        weight = tf.Variable(initializer(shape), dtype = dtype, name = name)
        if Regularizer is not None:
            self.add_loss(regularizer(weight))
        if constraint is not None:
            self.constraints[weight] = constraint
        if trainable:
            self._trainable_weights.append(weight)
        else:
            self._non_trainable_weights.append(weight)
        return weight

    def assert_input_compatibility(self, inputs):
        """Checks compatibility between the layer and provided inputs.

        This checks that the tensor(s) `input`
        verify the input assumptions of the layer
        (if any). If not, exceptions are raised.

        # Arguments
            inputs: input tensor or list of input tensors.

        # Raises
            ValueError: in case of mismatch between
                the provided inputs and the expectations of the layer.
        """
        inputs = _to_list(inputs)
        for x in inputs:
            try:
                generic_utils.is_tensor(x)
            except ValueError:
                raise ValueError('Layer ' + self.name + ' was called with '
                                 'an input that isn\'t a symbolic tensor. '
                                 'Received type: ' +
                                 str(type(x)) + '. Full input: ' +
                                 str(inputs) + '. All inputs to the layer '
                                 'should be tensors.')

        if not self.input_spec:
            return
        if not isinstance(self.input_spec, (list, tuple)):
            input_spec = _to_list(self.input_spec)
        else:
            input_spec = self.input_spec
        if len(inputs) != len(input_spec):
            raise ValueError('Layer ' + self.name + ' expects ' +
                             str(len(input_spec)) + ' inputs, '
                             'but it received ' + str(len(inputs)) +
                             ' input tensors. Input received: ' +
                             str(inputs))
        for input_index, (x, spec) in enumerate(zip(inputs, input_spec)):
            if spec is None:
                continue

            # Check ndim.
            if spec.ndim is not None:
                if ndim(x) != spec.ndim:
                    raise ValueError('Input ' + str(input_index) +
                                     ' is incompatible with layer ' +
                                     self.name + ': expected ndim=' +
                                     str(spec.ndim) + ', found ndim=' +
                                     str(ndim(x)))
            if spec.max_ndim is not None:
                ndim = ndim(x)
                if ndim is not None and ndim > spec.max_ndim:
                    raise ValueError('Input ' + str(input_index) +
                                     ' is incompatible with layer ' +
                                     self.name + ': expected max_ndim=' +
                                     str(spec.max_ndim) + ', found ndim=' +
                                     str(ndim(x)))
            if spec.min_ndim is not None:
                ndim = ndim(x)
                if ndim is not None and ndim < spec.min_ndim:
                    raise ValueError('Input ' + str(input_index) +
                                     ' is incompatible with layer ' +
                                     self.name + ': expected min_ndim=' +
                                     str(spec.min_ndim) + ', found ndim=' +
                                     str(ndim(x)))
            # Check dtype.
            dtype = tf.DType(x).name
            if spec.dtype is not None:
                if dtype != spec.dtype:
                    raise ValueError('Input ' + str(input_index) +
                                     ' is incompatible with layer ' +
                                     self.name + ': expected dtype=' +
                                     str(spec.dtype) + ', found dtype=' +
                                     str(dtype))
            # Check specific shape axes.
            if spec.axes:
                x_shape = tf.get_shape().as_list()
                if x_shape is not None:
                    for axis, value in spec.axes.items():
                        if value is not None and x_shape[int(axis)] not in {value, None}:
                            raise ValueError('Input ' + str(input_index) +
                                             ' is incompatible with layer ' +
                                             self.name + ': expected axis ' +
                                             str(axis) + ' of input shape to have '
                                             'value ' + str(value) +
                                             ' but got shape ' + str(x_shape))
            # Check shape.
            if spec.shape is not None:
                x_shape = tf.get_shape().as_list()
                if x_shape is not None:
                    for spec_dim, dim in zip(spec.shape, x_shape):
                        if spec_dim is not None and dim is not None:
                            if spec_dim != dim:
                                raise ValueError(
                                    'Input ' + str(input_index) +
                                    ' is incompatible with layer ' +
                                    self.name + ': expected shape=' +
                                    str(spec.shape) + ', found shape=' +
                                    str(x_shape))

    def call(self, inputs, **kwargs):
        """This is where the layer's logic lives.

        # Arguments
            inputs: Input tensor, or list/tuple of input tensors.
            **kwargs: Additional keyword arguments.

        # Returns
            A tensor or list/tuple of tensors.
        """
        return inputs

    def __call__(self, inputs, **kwargs):
        """Wrapper around self.call(), for handling internal references.

        If a Keras tensor is passed:
            - We call self._add_inbound_node().
            - If necessary, we `build` the layer to match
                the _keras_shape of the input(s).
            - We update the _keras_shape of every input tensor with
                its new shape (obtained via self.compute_output_shape).
                This is done as part of _add_inbound_node().
            - We update the _keras_history of the output tensor(s)
                with the current layer.
                This is done as part of _add_inbound_node().

        # Arguments
            inputs: Can be a tensor or list/tuple of tensors.
            **kwargs: Additional keyword arguments to be passed to `call()`.

        # Returns
            Output of the layer's `call` method.

        # Raises
            ValueError: in case the layer is missing shape information
                for its `build` call.
        """
        if isinstance(inputs, list):
            inputs = inputs[:]
        with tf.name_scope(self.name):
            # Handle laying building (weight creating, input spec locking).
            if not self.built:
                # Raise exceptions in case the input is not compatible
                # with the input_spec specified in the layer constructor.
                self.assert_input_compatibility(inputs)

                # Collect input shapes to build layer.
                input_shapes = []
                for x_elem in _to_list(inputs):
                    input_shapes.append(x_elem.get_shape().as_list())
                if len(input_shapes) == 1:
                    self.build(input_shapes[0])
                else:
                    self.build(input_shapes)
                self.built = True

                # Load weights that were specified at layer instantiation.
                if self._initial_weights is not None:
                    self.set_weights(self._initial_weights)

            # Raise exceptions in case the input is not compatible
            # with the input_spec set at build time.
            self.assert_input_compatibility(inputs)

            '''
            # Handle mask propagation.
            previous_mask = _collect_previous_mask(inputs)
            user_kwargs = copy.copy(kwargs)
            if not _is_all_none(previous_mask):
                # The previous layer generated a mask.
                if 'mask' in inspect.getargspec(self.call).args:
                    if 'mask' not in kwargs:
                        # If mask is explicitly passed to __call__,
                        # we should override the default mask.
                        kwargs['mask'] = previous_mask
            '''
            # Actually call the layer, collecting output(s), mask(s), and shape(s).
            output = self.call(inputs, **kwargs)
            '''
            output_mask = self.compute_mask(inputs, previous_mask)
            '''

            # If the layer returns tensors from its inputs, unmodified,
            # we copy them to avoid loss of tensor metadata.
            output_ls = _to_list(output)
            inputs_ls = _to_list(inputs)
            output_ls_copy = []
            for x in output_ls:
                if x in inputs_ls:
                    x = tf.identity(x)
                output_ls_copy.append(x)
            if len(output_ls_copy) == 1:
                output = output_ls_copy[0]
            else:
                output = output_ls_copy

            # Infering the output shape is only relevant for Theano.
            if all([s is not None for s in _to_list(input_shape)]):
                output_shape = self.compute_output_shape(input_shape)
            else:
                if isinstance(input_shape, list):
                    output_shape = [None for _ in input_shape]
                else:
                    output_shape = None

            # Add an inbound node to the layer, so that it keeps track
            # of the call and of all new variables created during the call.
            # This also updates the layer history of the output tensor(s).
            # If the input tensor(s) had not previous Keras history,
            # this does nothing.
            self._add_inbound_node(input_tensors=inputs, output_tensors=output,
                                   input_masks=previous_mask, output_masks=output_mask,
                                   input_shapes=input_shape, output_shapes=output_shape,
                                   arguments=user_kwargs)

            # Apply activity regularizer if any:
            if hasattr(self, 'activity_regularizer') and self.activity_regularizer is not None:
                regularization_losses = [self.activity_regularizer(x) for x in _to_list(output)]
                self.add_loss(regularization_losses, _to_list(inputs))
        return output

    def _add_inbound_node(self, input_tensors, output_tensors,
                          input_masks, output_masks,
                          input_shapes, output_shapes, arguments=None):
        """Internal method to create an inbound node for the layer.

        # Arguments
            input_tensors: list of input tensors.
            output_tensors: list of output tensors.
            input_masks: list of input masks (a mask can be a tensor, or None).
            output_masks: list of output masks (a mask can be a tensor, or None).
            input_shapes: list of input shape tuples.
            output_shapes: list of output shape tuples.
            arguments: dictionary of keyword arguments that were passed to the
                `call` method of the layer at the call that created the node.
        """
        input_tensors = _to_list(input_tensors)
        output_tensors = _to_list(output_tensors)
        input_masks = _to_list(input_masks)
        output_masks = _to_list(output_masks)
        input_shapes = _to_list(input_shapes)
        output_shapes = _to_list(output_shapes)

        # Collect input tensor(s) coordinates.
        inbound_layers = []
        node_indices = []
        tensor_indices = []
        for x in input_tensors:
            if hasattr(x, '_keras_history'):
                inbound_layer, node_index, tensor_index = x._keras_history
                inbound_layers.append(inbound_layer)
                node_indices.append(node_index)
                tensor_indices.append(tensor_index)
            else:
                inbound_layers.append(None)
                node_indices.append(None)
                tensor_indices.append(None)

        # Create node, add it to inbound nodes.
        Node(
            self,
            inbound_layers=inbound_layers,
            node_indices=node_indices,
            tensor_indices=tensor_indices,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            input_masks=input_masks,
            output_masks=output_masks,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            arguments=arguments
        )

        # Update tensor history, _keras_shape and _uses_learning_phase.
        for i in range(len(output_tensors)):
            output_tensors[i]._keras_shape = output_shapes[i]
            uses_lp = any([getattr(x, '_uses_learning_phase', False) for x in input_tensors])
            uses_lp = getattr(self, 'uses_learning_phase', False) or uses_lp
            output_tensors[i]._uses_learning_phase = getattr(output_tensors[i], '_uses_learning_phase', False) or uses_lp
            output_tensors[i]._keras_history = (self,
                                                len(self.inbound_nodes) - 1,
                                                i)

    def compute_output_shape(self, input_shape):
        """Computes the output shape of the layer.

        Assumes that the layer will be built
        to match that input shape provided.

        # Arguments
            input_shape: Shape tuple (tuple of integers)
                or list of shape tuples (one per output tensor of the layer).
                Shape tuples can include None for free dimensions,
                instead of an integer.

        # Returns
            An input shape tuple.
        """
        if hasattr(self, 'get_output_shape_for'):
            msg = "Class `{}.{}` defines `get_output_shape_for` but does not override `compute_output_shape`. " + \
                  "If this is a Keras 1 layer, please implement `compute_output_shape` to support Keras 2."
            warnings.warn(msg.format(type(self).__module__, type(self).__name__), stacklevel=2)
        return input_shape

    def build(self, input_shape):
        """Creates the layer weights.

        Must be implemented on all layers that have weights.

        # Arguments
            input_shape: Keras tensor (future input to layer)
                or list/tuple of Keras tensors to reference
                for weight shape computations.
        """
        self.built = True

    def add_loss(self, losses, inputs=None):
        """Add losses to the layer.

        The loss may potentially be conditional on some inputs tensors,
        for instance activity losses are conditional on the layer's inputs.

        # Arguments
            losses: loss tensor or list of loss tensors
                to add to the layer.
            inputs: input tensor or list of inputs tensors to mark
                the losses as conditional on these inputs.
                If None is passed, the loss is assumed unconditional
                (e.g. L2 weight regularization, which only depends
                on the layer's weights variables, not on any inputs tensors).
        """
        if losses is None or losses == []:
            return
        # Update self.losses
        losses = _to_list(losses)
        if hasattr(self, '_losses'):
            self._losses += losses
        # Update self._per_input_updates
        if isinstance(input, list) and inputs == []:
            inputs = None
        if inputs is not None:
            inputs_hash = _object_list_uid(inputs)
        else:
            # Updates indexed by None are unconditional
            # rather than input-dependent
            inputs_hash = None
        if inputs_hash not in self._per_input_losses:
            self._per_input_losses[inputs_hash] = []
        self._per_input_losses[inputs_hash] += losses
    
    def add_update(self, updates, inputs=None):
        """Add updates to the layer.

        The updates may potentially be conditional on some inputs tensors,
        for instance batch norm updates are conditional on the layer's inputs.

        # Arguments
            updates: update op or list of update ops
                to add to the layer.
            inputs: input tensor or list of inputs tensors to mark
                the updates as conditional on these inputs.
                If None is passed, the updates are assumed unconditional.
        """
        if updates is None or updates == []:
            return
        # Update self.updates
        updates = _to_list(updates)
        if hasattr(self, '_updates'):
            self._updates += updates
        # Update self._per_input_updates
        if isinstance(inputs, list) and inputs == []:
            inputs = None
        if inputs is not None:
            inputs_hash = _object_list_uid(inputs)
        else:
            # Updates indexed by None are unconditional
            # rather than input-dependent
            inputs_hash = None
        if inputs_hash not in self._per_input_updates:
            self._per_input_updates[inputs_hash] = []
        self._per_input_updates[inputs_hash] += updates

    def get_updates_for(self, inputs):
        if inputs is not None:
            inputs_hash = _object_list_uid(inputs)
        else:
            inputs_hash = None
        if inputs_hash in self._per_input_updates:
            return self._per_input_updates[inputs_hash]
        return []

    def get_losses_for(self, inputs):
        if inputs is not None:
            inputs_hash = _object_list_uid(inputs)
        else:
            inputs_hash = None
        if inputs_hash in self._per_input_losses:
            return self._per_input_losses[inputs_hash]
        return []

    @property
    def weights(self):
        return self.trainable_weights + self.non_trainable_weights
    
    def set_weights(self, weights):
        """Sets the weights of the layer, from Numpy arrays.

        # Arguments
            weights: a list of Numpy arrays. The number
                of arrays and their shape must match
                number of the dimensions of the weights
                of the layer (i.e. it should match the
                output of `get_weights`).

        # Raises
            ValueError: If the provided weights list does not match the
                layer's specifications.
        """
        params = self.weights
        if len(params) != len(weights):
            raise ValueError('You called `set_weights(weights)` on layer "' +
                             self.name +
                             '" with a  weight list of length ' +
                             str(len(weights)) +
                             ', but the layer was expecting ' +
                             str(len(params)) +
                             ' weights. Provided weights: ' +
                             str(weights)[:50] + '...')
        if not params:
            return
        weight_value_tuples = []
        param_values = generic_utils.batch_get_value(params)
        for pv, p, w in zip(param_values, params, weights):
            if pv.shape != w.shape:
                raise ValueError('Layer weight shape ' +
                                 str(pv.shape) +
                                 ' not compatible with '
                                 'provided weight shape ' + str(w.shape))
            weight_value_tuples.append((p, w))
        generic_utils.batch_set_value(weight_value_tuples)

    def get_weights(self):
        """Returns the current weights of the layer.

        # Returns
            Weights values as a list of numpy arrays.
        """
        params = self.weights
        return generic_utils.batch_get_value(params)

    def get_config(self):
        """Returns the config of the layer.

        A layer config is a Python dictionary (serializable)
        containing the configuration of a layer.
        The same layer can be reinstantiated later
        (without its trained weights) from this configuration.

        The config of a layer does not include connectivity
        information, nor the layer class name. These are handled
        by `Container` (one layer of abstraction above).

        # Returns
            Python dictionary.
        """
        config = {'name': self.name,
                  'trainable': self.trainable}
        if hasattr(self, 'batch_input_shape'):
            config['batch_input_shape'] = self.batch_input_shape
        if hasattr(self, 'dtype'):
            config['dtype'] = self.dtype
        return config

    @classmethod
    def from_config(cls, config):
        """Creates a layer from its config.

        This method is the reverse of `get_config`,
        capable of instantiating the same layer from the config
        dictionary. It does not handle layer connectivity
        (handled by Container), nor weights (handled by `set_weights`).

        # Arguments
            config: A Python dictionary, typically the
                output of get_config.

        # Returns
            A layer instance.
        """
        return cls(**config)

    def count_params(self):
        """Count the total number of scalars composing the weights.

        # Returns
            An integer count.

        # Raises
            RuntimeError: if the layer isn't yet built
                (in which case its weights aren't yet defined).
        """
        if not self.built:
            if self.__class__.__name__ == 'Sequential':
                self.build()
            else:
                raise RuntimeError('You tried to call `count_params` on ' +
                                   self.name + ', but the layer isn\'t built. '
                                   'You can build it manually via: `' +
                                   self.name + '.build(batch_input_shape)`.')
        return sum([generic_utils.count_params(p) for p in self.weights])

