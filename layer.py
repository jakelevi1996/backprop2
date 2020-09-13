import numpy as np

class NeuralLayer():
    def __init__(self, num_units, num_inputs, act_func, weight_std, bias_std):
        """
        Initialise the constants and parameters for a neural network layer. This
        method is called during NeuralNetwork.__init__()
        """
        # Randomly initialise parameters
        self.weights = np.random.normal(0, weight_std, [num_units, num_inputs])
        self.bias = np.random.normal(0, bias_std, [num_units, 1])
        
        # Set layer constants
        self.input_dim      = num_inputs
        self.output_dim     = num_units
        self.act_func       = act_func
        self.num_weights    = self.weights.size
        self.num_bias       = self.bias.size
        self.diag_indices   = np.diag_indices(num_units)

        x, y = np.indices([num_units, num_inputs])
        self.eps_inds       = x.ravel()
        self.z_inds         = y.ravel()
    
    def activate(self, layer_input):
        """
        Calculate the pre-activation and output of this layer as a function of
        the input, and store the input for subsequent gradient calculations.

        Inputs:
        -   layer_input: input to the layer. Should be a numpy array with shape
            (input_dim, N_D)

        Outputs:
        -   self.output: output from the layer, in a numpy array with shape
            (output_dim, N_D)
        -   self.pre_activation (not returned): result of linear transformation
            being applied to the input, before nonlinearity is applied to give
            the output, in a numpy array with shape (output_dim, N_D)
        """
        self.input = layer_input
        self.pre_activation = np.matmul(self.weights, layer_input) + self.bias
        self.output = self.act_func(self.pre_activation)
        return self.output
    
    def backprop(self, next_layer):
        """
        Calculate delta (gradient of the error function with respect to the
        pre-activations) for the current layer of the network, using the delta
        and weights of the next layer in the network.

        Inputs:
        -   next_layer: the next layer in the network, as an instance of
            NeuralLayer. Must have pre-calculated delta for that layer, and its
            delta and weights must be the correct shape
        """
        self.delta = np.einsum(
            "jk,ji,ik->ik",
            next_layer.delta,
            next_layer.weights,
            self.act_func.dydx(self.pre_activation)
        )
    
    def backprop2(self, next_layer):
        """
        Calculate epsilon (2nd derivative of the error function with respect to
        the pre-activations) for the current layer of the network, using the
        delta and weights of the next layer in the network.

        Inputs:
        -   next_layer: the next layer in the network, as an instance of
            NeuralLayer. Must have pre-calculated epsilon and delta for that
            layer, and its epsilon, delta and weights must be the correct shape.
        """
        self.epsilon = np.einsum(
            "kmd,ki,id,mj,jd->ijd",
            next_layer.epsilon,
            next_layer.weights,
            self.act_func.dydx(self.pre_activation),
            next_layer.weights,
            self.act_func.dydx(self.pre_activation)
        )
        self.epsilon[self.diag_indices] += np.einsum(
            "jk,ji,ik->ik",
            next_layer.delta,
            next_layer.weights,
            self.act_func.d2ydx2(self.pre_activation)
        )
    
    def calc_gradients(self):
        """
        Calculate the gradients of the error function with respect to the bias
        and weights in this layer, using the delta for this layer, which must
        have already been calculated using self.backprop(next_layer).

        TODO: test whether optimising the path gives a speed-up, and if so then
        calculate path in advance using np.einsum_path during initialisation.

        NB if the deltas and inputs are transposed, then the einsum subscripts
        should be "ij,ik->ijk"
        """
        self.b_grad = self.delta
        self.w_grad = np.einsum("ik,jk->ijk", self.delta, self.input)
    
    def calc_weight_gradients2(self, block_inds, N_D):
        """
        ...
        """
        block_size = len(block_inds)
        hessian_block = np.multiply(
            [
                self.epsilon[
                    self.eps_inds[block_inds].reshape(block_size, 1),
                    self.eps_inds[block_inds].reshape(1, block_size),
                    :
                ],
                self.input[
                    self.z_inds[block_inds],
                    :
                ].reshape(block_size, 1, N_D),
                self.input[
                    self.z_inds[block_inds],
                    :
                ].reshape(1, block_size, N_D)
            ],
            axis=0
        )
        return hessian_block