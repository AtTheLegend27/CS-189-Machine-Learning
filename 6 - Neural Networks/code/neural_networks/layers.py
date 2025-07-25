"""
Author: Sophia Sanborn, Sagnik Bhattacharya
Institution: UC Berkeley
Date: Spring 2020
Course: CS189/289A
Website: github.com/sophiaas, github.com/sagnibak
Modified by: Bill Zheng, Tejas Prabhune, Spring 2025
Website: github.com/WJ2003B, github.com/tejasprabhune
"""

import numpy as np
from abc import ABC, abstractmethod

from neural_networks.activations import initialize_activation
from neural_networks.weights import initialize_weights
from collections import OrderedDict

from typing import Callable, List, Literal, Tuple, Union


class Layer(ABC):
    """Abstract class defining the `Layer` interface."""

    def __init__(self):
        self.activation = None

        self.n_in = None
        self.n_out = None

        self.parameters = {}
        self.cache = {}
        self.gradients = {}

        super().__init__()

    @abstractmethod
    def forward(self, z: np.ndarray) -> np.ndarray:
        pass

    def clear_gradients(self) -> None:
        self.cache = OrderedDict({a: [] for a, b in self.cache.items()})
        self.gradients = OrderedDict(
            {a: np.zeros_like(b) for a, b in self.gradients.items()}
        )

    def forward_with_param(
        self, param_name: str, X: np.ndarray,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Call the `forward` method but with `param_name` as the variable with
        value `param_val`, and keep `X` fixed.
        """

        def inner_forward(param_val: np.ndarray) -> np.ndarray:
            self.parameters[param_name] = param_val
            return self.forward(X)

        return inner_forward

    def _get_parameters(self) -> List[np.ndarray]:
        return [b for a, b in self.parameters.items()]

    def _get_cache(self) -> List[np.ndarray]:
        return [b for a, b in self.cache.items()]

    def _get_gradients(self) -> List[np.ndarray]:
        return [b for a, b in self.gradients.items()]


def initialize_layer(
    name: str,
    activation: str = None,
    weight_init: str = None,
    n_out: int = None,
    kernel_shape: Tuple[int, int] = None,
    stride: int = None,
    pad: int = None,
    mode: str = None,
    eps: float = 1e-8,
    momentum: float = 0.95,
    keep_dim: str = "first",
) -> Layer:
    """Factory function for layers."""
    if name == "fully_connected":
        return FullyConnected(
            n_out=n_out, activation=activation, weight_init=weight_init,
        )

    elif name == "conv2d":
        return Conv2D(
            n_out=n_out,
            activation=activation,
            kernel_shape=kernel_shape,
            stride=stride,
            pad=pad,
            weight_init=weight_init,
        )
    
    elif name == "batchnorm1d":
        return BatchNorm1D(eps=eps, momentum=momentum,)

    elif name == "pool2d":
        return Pool2D(kernel_shape=kernel_shape, mode=mode, stride=stride, pad=pad)

    elif name == "flatten":
        return Flatten(keep_dim=keep_dim)

    else:
        raise NotImplementedError("Layer type {} is not implemented".format(name))


class FullyConnected(Layer):
    """A fully-connected layer multiplies its input by a weight matrix, adds
    a bias, and then applies an activation function.
    """

    def __init__(
        self, n_out: int, activation: str, weight_init="xavier_uniform"
    ) -> None:

        super().__init__()
        self.n_in = None
        self.n_out = n_out
        self.activation = initialize_activation(activation)

        # instantiate the weight initializer
        self.init_weights = initialize_weights(weight_init, activation=activation)

    def _init_parameters(self, X_shape: Tuple[int, int]) -> None:
        """Initialize all layer parameters (weights, biases)."""
        self.n_in = X_shape[1]

        ### BEGIN YOUR CODE ###

        W = self.init_weights((self.n_in, self.n_out))
        b = np.zeros((1,self.n_out))

        self.parameters = OrderedDict({"W": W, "b": b}) # DO NOT CHANGE THE KEYS
        self.cache = OrderedDict()   # cache for backprop
        self.gradients = OrderedDict({"W": np.zeros_like(W), "b": np.zeros_like(b)})  # parameter gradients initialized to zero
                                           # MUST HAVE THE SAME KEYS AS `self.parameters`

        ### END YOUR CODE ###

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: multiply by a weight matrix, add a bias, apply activation.
        Also, store all necessary intermediate results in the `cache` dictionary
        to be able to compute the backward pass.

        Parameters
        ----------
        X  input matrix of shape (batch_size, input_dim)

        Returns
        -------
        a matrix of shape (batch_size, output_dim)
        """
        # initialize layer parameters if they have not been initialized
        if self.n_in is None:
            self._init_parameters(X.shape)

        ### BEGIN YOUR CODE ###
        W = self.parameters["W"]
        b = self.parameters["b"]
        Z = X@W + b
        out = self.activation(Z)
        
        self.cache["Z"] = Z
        self.cache["X"] = X

        ### END YOUR CODE ###

        return out

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for fully connected layer.
        Compute the gradients of the loss with respect to:
            1. the weights of this layer (mutate the `gradients` dictionary)
            2. the bias of this layer (mutate the `gradients` dictionary)
            3. the input of this layer (return this)

        Parameters
        ----------
        dLdY  gradient of the loss with respect to the output of this layer
              shape (batch_size, output_dim)

        Returns
        -------
        gradient of the loss with respect to the input of this layer
        shape (batch_size, input_dim)
        """
        ### BEGIN YOUR CODE ###
        X = self.cache["X"]
        Z = self.cache["Z"]
        W = self.parameters["W"]
        
        dZ = self.activation.backward(Z, dLdY)
        dX = dZ @ W.T
        dW = X.T @ dZ
        dB = np.sum(dZ, axis=0, keepdims = True)

        # store the gradients in `self.gradients`
        # the gradient for self.parameters["W"] should be stored in
        # self.gradients["W"], etc.
        
        self.gradients["W"] = dW
        self.gradients["b"] = dB

        ### END YOUR CODE ###

        return dX


class BatchNorm1D(Layer):
    def __init__(
        self, 
        weight_init: str = "xavier_uniform",
        eps: float = 1e-8,
        momentum: float = 0.9,
    ) -> None:
        super().__init__()
        
        # instantiate the weight initializer
        self.init_weights = initialize_weights(weight_init,)

        self.eps = eps
        self.momentum = momentum

    def _init_parameters(self, X_shape: Tuple[int, int]) -> None:
        """Initialize all layer parameters (weights, biases)."""
        self.n_in = X_shape[1]

        ### BEGIN YOUR CODE ###

        gamma = self.init_weights((1, self.n_in))
        beta = np.zeros((1, self.n_in))

        self.parameters = OrderedDict({"gamma": gamma, "beta": beta}) # DO NOT CHANGE THE KEYS
        self.cache = OrderedDict({"X": None, "X_hat": None, 
                                  "mu": None, "var": None, 
                                  "running_mu": np.zeros((1, self.n_in)), "running_var": np.zeros((1, self.n_in))})  
        # cache for backprop
        self.gradients = OrderedDict({
        "gamma": np.zeros_like(gamma),
        "beta": np.zeros_like(beta),
    })  
        # parameter gradients initialized to zero
        # MUST HAVE THE SAME KEYS AS `self.parameters`

        ### END YOUR CODE ###

    def forward(self, X: np.ndarray, mode: str = "train") -> np.ndarray:
        """ Forward pass for 1D batch normalization layer.
        Allows taking in an array of shape (B, C) and performs batch normalization over it. 

        We use Exponential Moving Average to update the running mean and variance. with alpha value being equal to self.gamma

        You should set the running mean and running variance to the mean and variance of the first batch after initializing it.
        You should also make separate cases for training mode and testing mode.
        """
        ### BEGIN YOUR CODE ###
        
        gamma = self.parameters["gamma"]
        beta = self.parameters["beta"]
        
        #train mode
        if mode == "train":
            mu = np.mean(X, axis = 0, keepdims=True)
            var = np.var(X, axis = 0, keepdims=True)
            X_hat = (X - mu) / np.sqrt(var + self.eps)
            out = gamma * X_hat + beta
            
            if self.cache["running_mu"] is None:
                self.cache["running_mu"] = mu
                self.cache["running_var"] = var
            else:
                self.cache["running_mu"] = (
                    self.momentum * self.cache["running_mu"] + (1 - self.momentum) * mu
                )
                self.cache["running_var"] = (
                    self.momentum * self.cache["running_var"] + (1 - self.momentum) * var
                )
            
            #cache back
            self.cache["X"]  = X
            self.cache["X_hat"]  = X_hat
            self.cache["mu"] = mu
            self.cache["var"] = var
            
        #test mode
        else:
            mu = self.cache["running_mu"]
            var = self.cache["running_var"]
            X_hat = (X - mu) / np.sqrt(var + self.eps)
            out = gamma * X_hat + beta

        ### END YOUR CODE ###
        return out

    def backward(self, dY: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward method for batch normalization layer. You don't need to implement this to get full credit, although it is
        fun to do so if you have the time.
        """

        ### BEGIN YOUR CODE ###

        # implement backward pass for batchnorm.

        ### END YOUR CODE ###
        

        return dX

class Conv2D(Layer):
    """Convolutional layer for inputs with 2 spatial dimensions."""

    def __init__(
        self,
        n_out: int,
        kernel_shape: Tuple[int, int],
        activation: str,
        stride: int = 1,
        pad: str = "same",
        weight_init: str = "xavier_uniform",
    ) -> None:

        super().__init__()
        self.n_in = None
        self.n_out = n_out
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.pad = pad

        self.activation = initialize_activation(activation)
        self.init_weights = initialize_weights(weight_init, activation=activation)

    def _init_parameters(self, X_shape: Tuple[int, int, int, int]) -> None:
        """Initialize all layer parameters and determine padding."""
        self.n_in = X_shape[3]

        W_shape = self.kernel_shape + (self.n_in,) + (self.n_out,)
        W = self.init_weights(W_shape)
        b = np.zeros((1, self.n_out))

        self.parameters = OrderedDict({"W": W, "b": b}) # DO NOT CHANGE THE KEYS
        self.cache = OrderedDict({"Z": [], "X": []}) # cache for backprop
        self.gradients = OrderedDict({"W": np.zeros_like(W), "b": np.zeros_like(b)}) # parameter gradients initialized to zero
                                                                                     # MUST HAVE THE SAME KEYS AS `self.parameters`

        if self.pad == "same":
            self.pad = ((W_shape[0] - 1) // 2, (W_shape[1] - 1) // 2)
        elif self.pad == "valid":
            self.pad = (0, 0)
        elif isinstance(self.pad, int):
            self.pad = (self.pad, self.pad)
        else:
            raise ValueError("Invalid Pad mode found in self.pad.")

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass for convolutional layer. This layer convolves the input
        `X` with a filter of weights, adds a bias term, and applies an activation
        function to compute the output. This layer also supports padding and
        integer strides. Intermediates necessary for the backward pass are stored
        in the cache.

        Parameters
        ----------
        X  input with shape (batch_size, in_rows, in_cols, in_channels)

        Returns
        -------
        output feature maps with shape (batch_size, out_rows, out_cols, out_channels)
        """
        if self.n_in is None:
            self._init_parameters(X.shape)

        W = self.parameters["W"]
        b = self.parameters["b"]

        kernel_height, kernel_width, in_channels, out_channels = W.shape
        n_examples, in_rows, in_cols, in_channels = X.shape
        kernel_shape = (kernel_height, kernel_width)
        stride_height = stride_width = self.stride
        pad_height, pad_width = self.pad
        
        ### BEGIN YOUR CODE ###
        X_padded = np.pad(X, ((0, 0), (pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant')
        
        out_r = (in_rows + 2 * pad_height - kernel_height) // stride_height + 1
        out_c = (in_cols + 2 * pad_width - kernel_width) // stride_width + 1

        out_pre = np.zeros((n_examples, out_r, out_c, out_channels))
        
        for r in range(n_examples):
            for i in range(out_r):
                for j in range(out_c):
                    for f in range(out_channels):
                        r_strt = i * stride_height
                        c_strt = j * stride_width
                        
                        r_nd = r_strt + kernel_height
                        c_nd = c_strt + kernel_width
                        
                        curr = X_padded[r, r_strt:r_nd, c_strt:c_nd, :]
                        kernel = W[:, :, :, f]
                        
                        out_pre[r, i, j, f] = np.sum(curr * kernel) + b[0, f]
        
        out = self.activation(out_pre)
                        
        # cache any values required for backprop
        self.cache["Z"] = out_pre
        self.cache["X"] = X
        ### END YOUR CODE ###
        
        return out

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for conv layer. Computes the gradients of the output
        with respect to the input feature maps as well as the filter weights and
        biases.

        Parameters
        ----------
        dLdY  gradient of loss with respect to output of this layer
              shape (batch_size, out_rows, out_cols, out_channels)

        Returns
        -------
        gradient of the loss with respect to the input of this layer
        shape (batch_size, in_rows, in_cols, in_channels)
        """
        ### BEGIN YOUR CODE ###

        # perform a backward pass

        ### END YOUR CODE ###

        return dX

class Pool2D(Layer):
    """Pooling layer, implements max and average pooling."""

    def __init__(
        self,
        kernel_shape: Tuple[int, int],
        mode: str = "max",
        stride: int = 1,
        pad: Union[int, Literal["same"], Literal["valid"]] = 0,
    ) -> None:

        if type(kernel_shape) == int:
            kernel_shape = (kernel_shape, kernel_shape)

        self.kernel_shape = kernel_shape
        self.stride = stride

        if pad == "same":
            self.pad = ((kernel_shape[0] - 1) // 2, (kernel_shape[1] - 1) // 2)
        elif pad == "valid":
            self.pad = (0, 0)
        elif isinstance(pad, int):
            self.pad = (pad, pad)
        else:
            raise ValueError("Invalid Pad mode found in self.pad.")

        self.mode = mode

        if mode == "max":
            self.pool_fn = np.max
            self.arg_pool_fn = np.argmax
        elif mode == "average":
            self.pool_fn = np.mean

        self.cache = {
            "out_rows": [],
            "out_cols": [],
            "X_pad": [],
            "p": [],
            "pool_shape": [],
        }
        self.parameters = {}
        self.gradients = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: use the pooling function to aggregate local information
        in the input. This layer typically reduces the spatial dimensionality of
        the input while keeping the number of feature maps the same.

        As with all other layers, please make sure to cache the appropriate
        information for the backward pass.

        Parameters
        ----------
        X  input array of shape (batch_size, in_rows, in_cols, channels)

        Returns
        -------
        pooled array of shape (batch_size, out_rows, out_cols, channels)
        """
        ### BEGIN YOUR CODE ###
        
        kernel_height, kernel_width = self.kernel_shape
        n_examples, in_rows, in_cols, in_channels = X.shape
        stride_height = stride_width = self.stride
        pad_height, pad_width = self.pad
        
        X_padded = np.pad(X, ((0, 0), (pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant')
        
        out_r = (in_rows + 2 * pad_height - kernel_height) // stride_height + 1
        out_c = (in_cols + 2 * pad_width - kernel_width) // stride_width + 1
        
        if self.mode == "max":
            pool_function = np.max
        else:
            pool_function = np.mean

        X_pool = np.zeros((n_examples, out_r, out_c, in_channels))        
        
        for i in range(out_r):
            for j in range(out_c):
                r_strt = i * stride_height
                c_strt = j * stride_width
                
                r_nd = r_strt + kernel_height
                c_nd = c_strt + kernel_width
                
                X_pool[:, i, j, :] = pool_function(X_padded[:, r_strt:r_nd, c_strt:c_nd, :], axis=(1, 2))
                
                
        # cache any values required for backprop
        self.cache["X_pad"] = X_padded
        self.cache["pool_shape"] = (kernel_height, kernel_width)
        self.cache["out_rows"] = out_r
        self.cache["out_cols"] = out_c
        self.cache["p"] = (pad_height, pad_width)
        ### END YOUR CODE ###

        return X_pool

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for pooling layer.

        Parameters
        ----------
        dLdY  gradient of loss with respect to the output of this layer
              shape (batch_size, out_rows, out_cols, channels)

        Returns
        -------
        gradient of loss with respect to the input of this layer
        shape (batch_size, in_rows, in_cols, channels)
        """
        ### BEGIN YOUR CODE ###

        X_pad = self.cache["X_pad"]
        kernel_height, kernel_width = self.cache["pool_shape"]
        out_r, out_c = self.cache["out_rows"], self.cache["out_cols"]
        stride_height = stride_width = self.stride
        pad_height, pad_width = self.cache["p"]
        n_examples, in_rows_pad, in_cols_pad, in_channels = X_pad.shape
        
        dX_pad = np.zeros_like(X_pad)
        
        for n in range(n_examples):
            for i in range(out_r):
                for j in range(out_c):
                    r_strt = i * stride_height
                    c_strt = j * stride_width
                    
                    r_nd = r_strt + kernel_height
                    c_nd = c_strt + kernel_width
                    
                    if self.mode == "average":
                        gradient = dLdY[n, i, j, :] / (kernel_height * kernel_width)
                        dX_pad[n, r_strt:r_nd, c_strt:c_nd, :] += gradient[np.newaxis, np.newaxis, :]
                    else:
                        patch = X_pad[n, r_strt:r_nd, c_strt:c_nd, :]  
                        max_val = np.max(patch, axis=(0, 1), keepdims=True)
                        mask = (patch == max_val)
                        gradient = dLdY[n, i, j, :][np.newaxis, np.newaxis, :]
                        dX_pad[n, r_strt:r_nd, c_strt:c_nd, :] += mask * gradient
                        
        ### END YOUR CODE ###
        gradX = dX_pad[:, pad_height:-pad_height or None, pad_width:-pad_width or None, :]
        
        return gradX

class Flatten(Layer):
    """Flatten the input array."""

    def __init__(self, keep_dim: str = "first") -> None:
        super().__init__()

        self.keep_dim = keep_dim
        self._init_params()

    def _init_params(self):
        self.X = []
        self.gradients = {}
        self.parameters = {}
        self.cache = {"in_dims": []}

    def forward(self, X: np.ndarray, retain_derived: bool = True) -> np.ndarray:
        self.cache["in_dims"] = X.shape

        if self.keep_dim == -1:
            return X.flatten().reshape(1, -1)

        rs = (X.shape[0], -1) if self.keep_dim == "first" else (-1, X.shape[-1])
        return X.reshape(*rs)

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        in_dims = self.cache["in_dims"]
        gradX = dLdY.reshape(in_dims)
        return gradX
