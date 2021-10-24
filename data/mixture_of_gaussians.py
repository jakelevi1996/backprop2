from data.classification import Classification, BinaryClassification
import numpy as np

def _mixture_of_gaussians(
    input_dim,
    output_dim,
    n_points,
    n_mixture_components,
    scale_matrix,
    mean,
    mixture_to_class,
):
    """ Generate inputs, labels and one-hot outputs for a mixture-of-Gaussians
    classification dataset. This function avoids duplicate code for generating
    the training and test sets within the MixtureOfGaussians initialiser
    method.

    Inputs:
    -   input_dim: number of input dimensions, as an integer
    -   output_dim: number of output dimensions, as an integer
    -   n_points: number of data points to generate, as an integer
    -   n_mixture_components: number of mixture components, as an integer
    -   scale_matrix: array of matrices used to scale the inputs in each
        mixture component, as a numpy array with shape (n_mixture_components,
        input_dim, input_dim)
    -   mean: array of vectors used to translate the inputs in each mixture
        component, as a numpy array with shape (n_mixture_components,
        input_dim)
    -   mixture_to_class: array of integers mapping mixture components to
        classes, as a 1D numpy array with n_mixture_components elements, each
        of which is an integer in the half-open interval [0, output_dim)

    Outputs:
    -   x: input coordindates for each data point, as a numpy array with shape
        (input_dim, n_points)
    -   labels: labels for each data point, as a 1D numpy array with n_points
        elements, each of which is an integer in the half-open interval [0,
        output_dim)
    -   y: binary matrix containing one-hot outputs for each data point, as a
        numpy array with shape (output_dim, n_points), in which each column has
        one value equal to 1 (corresponding to the class that the data point
        belongs to) and all other values equal to zero
    """
    # Generate all of the input points
    x = np.random.normal(size=[input_dim, n_points])
    # Generate assignments of inputs to mixture components
    z = np.random.randint(n_mixture_components, size=n_points)
    # Transform each input point according to its mixture component
    for i in range(n_mixture_components):
        x[:, z == i] = (
            (scale_matrix[i] @ x[:, z == i])
            + mean[i].reshape(-1, 1)
        )
    # Set labels and one-hot output data using mixture components
    labels = mixture_to_class[z]
    y = np.zeros([output_dim, n_points])
    y[labels, np.arange(n_points)] = 1
    return x, labels, y


class MixtureOfGaussians(Classification):
    """ Class for a mixture-of-Gaussians classification dataset. The means of
    the mixture components are generated from a normal distribution, and the
    variance matrices are generated randomly and implicitly. There can be more
    mixture components than classes, although by default there will be there
    same number of mixture components and classes.  """
    def __init__(
        self,
        input_dim=2,
        output_dim=3,
        n_train=None,
        n_test=None,
        n_mixture_components=None,
        scale=0.2,
    ):
        """ Initialise a mixture-of-Gaussians classification data set.

        Inputs:
        -   input_dim: dimensionality of inputs for this data set. Should be a
            positive integer
        -   output_dim: dimensionality of outputs for this data set. Should be
            a positive integer
        -   n_train: number of points in the training set for this data set.
            Should be None or a positive integer. If n_train is None, then it
            is chosen as 50 to the power of the input dimension
        -   n_test: number of points in the test set for this data set. Should
            be None or a positive integer. If n_test is None, then it is chosen
            to be equal to the number of points in the training set
        -   n_mixture_components: number of mixture components in the data set.
            Should be None, or a positive integer. Can be more than the output
            dimension, in which case some classes will have multiple mixture
            components, or less than the output dimension, in which case some
            classes will have no mixture components (and therefore no data
            points)
        -   scale: positive float, which determines the expected variance of
            the mixture components relative to the distance between them. A
            larger scale will mean that the variances of mixture components are
            large, and therefore the mixture components are more likely to
            overlap and become more difficult to distinguish, making the
            classification task "harder"
        """
        Classification.__init__(self)
        # Set shape constants and number of mixture components
        self.set_shape_constants(input_dim, output_dim, n_train, n_test)
        if n_mixture_components is None:
            n_mixture_components = self.output_dim

        # Generate mean and scale for each mixture component
        mean = np.random.normal(size=[n_mixture_components, input_dim])
        scale_matrix = scale * np.random.normal(
            size=[n_mixture_components, input_dim, input_dim],
        )

        # Initialise mixture_to_class which maps mixture components to classes
        if n_mixture_components > output_dim:
            mixture_to_class = np.full(n_mixture_components, np.nan)
            initial_class_assignment_inds = np.random.choice(
                n_mixture_components,
                size=output_dim,
                replace=False,
            )
            mixture_to_class[initial_class_assignment_inds] = np.arange(
                output_dim
            )
            mixture_to_class[np.isnan(mixture_to_class)] = np.random.randint(
                output_dim,
                size=n_mixture_components-output_dim,
            )
            mixture_to_class = mixture_to_class.astype(int)
        elif n_mixture_components < output_dim:
            mixture_to_class = np.random.choice(
                output_dim,
                size=n_mixture_components,
                replace=False,
            )
        else:
            mixture_to_class = np.arange(output_dim)

        # Generate inputs, outputs and labels
        self.train.x, self.train_labels, self.train.y = _mixture_of_gaussians(
            self.input_dim,
            self.output_dim,
            self.train.n,
            n_mixture_components,
            scale_matrix,
            mean,
            mixture_to_class,
        )
        self.test.x, self.test_labels, self.test.y = _mixture_of_gaussians(
            self.input_dim,
            self.output_dim,
            self.test.n,
            n_mixture_components,
            scale_matrix,
            mean,
            mixture_to_class,
        )


def _binary_mixture_of_gaussians(
    input_dim,
    n_points,
    n_mixture_components,
    scale_matrix,
    mean,
    mixture_to_class,
):
    """ Generate data for a binary mixture-of-Gaussians classification dataset.
    This function avoids duplicate code for generating the training and test
    sets within the BinaryMixtureOfGaussians initialiser method.

    Inputs:
    -   input_dim: number of input dimensions, as an integer
    -   n_points: number of data points to generate, as an integer
    -   n_mixture_components: number of mixture components, as an integer
    -   scale_matrix: array of matrices used to scale the inputs in each
        mixture component, as a numpy array with shape (n_mixture_components,
        input_dim, input_dim)
    -   mean: array of vectors used to translate the inputs in each mixture
        component, as a numpy array with shape (n_mixture_components,
        input_dim)
    -   mixture_to_class: binary array of integers mapping mixture components
        to classes, as a 1D numpy array with n_mixture_components elements,
        each of which is an integer in {0, 1}

    Outputs:
    -   x: input coordindates for each data point, as a numpy array with shape
        (input_dim, n_points)
    -   y: binary vector containing class labels for each data point, as a
        numpy array with shape (1, n_points), in which each element is in {0,
        1}
    """
    # Generate all of the input points
    x = np.random.normal(size=[input_dim, n_points])
    # Generate assignments of inputs to mixture components
    z = np.random.randint(n_mixture_components, size=n_points)
    # Transform each input point according to its mixture component
    for i in range(n_mixture_components):
        x[:, z == i] = (
            (scale_matrix[i] @ x[:, z == i])
            + mean[i].reshape(-1, 1)
        )
    # Set output labels using mixture components
    y = np.array(mixture_to_class[z]).reshape(1, -1)
    return x, y


class BinaryMixtureOfGaussians(BinaryClassification):
    """ Class for a binary mixture-of-Gaussians classification dataset. The
    means of the mixture components are generated from a normal distribution,
    and the variance matrices are generated randomly and implicitly. """
    def __init__(
        self,
        input_dim=2,
        n_train=None,
        n_test=None,
        n_mixture_components=2,
        scale=0.2,
    ):
        """ Initialise a binary mixture-of-Gaussians classification data set.

        Inputs:
        -   input_dim: dimensionality of inputs for this data set. Should be a
            positive integer
        -   n_train: number of points in the training set for this data set.
            Should be None or a positive integer. If n_train is None, then it
            is chosen as 50 to the power of the input dimension
        -   n_test: number of points in the test set for this data set. Should
            be None or a positive integer. If n_test is None, then it is chosen
            to be equal to the number of points in the training set
        -   n_mixture_components: number of mixture components in the data set.
            Should be a positive integer >= 2. If > 2, then there will be
            multiple mixture components per class
        -   scale: positive float, which determines the expected variance of
            the mixture components relative to the distance between them. A
            larger scale will mean that the variances of mixture components are
            large, and therefore the mixture components are more likely to
            overlap and become more difficult to distinguish, making the
            classification task "harder"
        """
        BinaryClassification.__init__(self)
        # Set shape constants and number of mixture components
        self.set_shape_constants(input_dim, 1, n_train, n_test)
        if n_mixture_components is None:
            n_mixture_components = self.output_dim

        # Generate mean and scale for each mixture component
        mean = np.random.normal(size=[n_mixture_components, input_dim])
        scale_matrix = scale * np.random.normal(
            size=[n_mixture_components, input_dim, input_dim]
        )

        # Initialise mixture_to_class which maps mixture components to classes
        if n_mixture_components > 2:
            mixture_to_class = np.full(n_mixture_components, np.nan)
            initial_class_assignment_inds = np.random.choice(
                n_mixture_components,
                size=2,
                replace=False,
            )
            mixture_to_class[initial_class_assignment_inds] = np.arange(2)
            mixture_to_class[np.isnan(mixture_to_class)] = np.random.randint(
                2,
                size=n_mixture_components-2,
            )
            mixture_to_class = mixture_to_class.astype(int)
        else:
            mixture_to_class = np.array([0, 1])

        # Generate inputs, outputs and labels
        self.train.x, self.train.y = _binary_mixture_of_gaussians(
            self.input_dim,
            self.train.n,
            n_mixture_components,
            scale_matrix,
            mean,
            mixture_to_class,
        )
        self.test.x, self.test.y = _binary_mixture_of_gaussians(
            self.input_dim,
            self.test.n,
            n_mixture_components,
            scale_matrix,
            mean,
            mixture_to_class,
        )
