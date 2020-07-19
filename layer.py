import numpy as np

class NeuralLayer():
    def __init__(self, num_units, num_inputs, act_func, weight_std, bias_std):
        """
        __init__: initialise the constants and parameters for a neural network
        layer. This method is called during NeuralNetwork.__init__()
        """
        # Set layer constants
        self.input_dim = num_inputs
        self.output_dim = num_units
        self.act_func = act_func
        
        # Randomly initialise parameters
        self.weights = np.random.normal(0, weight_std, [num_units, num_inputs])
        self.bias = np.random.normal(0, bias_std, [num_units, 1])
    
    def activate(self, input):
        """
        activate: calculate the pre-activation and output of this layer as a
        function of the input, and store the input for subsequent gradient
        calculations.

        Inputs:
        -   input: input to the layer. Should be a numpy array with shape
            (input_dim, N_D)

        Outputs:
        -   self.output: output from the layer, in a numpy array with shape
            (output_dim, N_D)
        -   self.pre_activation (not returned): result of linear transformation
            being applied to the input, before nonlinearity is applied to give
            the output, in a numpy array with shape (output_dim, N_D)
        """
        self.input = input
        self.pre_activation = np.matmul(self.weights, input) + self.bias
        self.output = self.act_func(self.pre_activation)
        return self.output
    
    def backprop(self, next_layer):
        """
        backprop: calculate delta (gradient of the error function with respect
        to the pre-activations) for the current layer of the network, using the
        delta and weights of the next layer in the network.

        Inputs:
        -   next_layer: the next layer in the network, as an instance of
            NeuralLayer. Must have pre-calculated delta for that layer, and its
            delta and weights must be the correct shape
        """
        self.delta = np.einsum(
            "jk,ji,ik->ik", next_layer.delta, next_layer.weights,
            self.act_func.dydx(self.pre_activation)
        )
    
    def calc_gradients(self):
        """
        calc_gradients: calculate the gradients of the error function with
        respect to the bias and weights in this layer, using the delta for this
        layer, which must have already been calculated using
        self.backprop(next_layer).

        TODO: test whether optimising the path gives a speed-up, and if so then
        calculate path in advance using np.einsum_path during initialisation.

        NB if the deltas and inputs are transposed, then the einsum subscripts
        should be "ij,ik->ijk"
        """
        self.b_grad = self.delta
        self.w_grad = np.einsum("ik,jk->ijk", self.delta, self.input)
