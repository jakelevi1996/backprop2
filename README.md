# backprop2

![image](https://github.com/jakelevi1996/backprop2/raw/master/Results/Images%20for%20Readme/Hidden%20activations%20meta-learning%203%20-%20collage.png "Meta-learning model adapted to 3 different examples of an image of a 3")

## Contents

- [backprop2](#backprop2)
  - [Contents](#contents)
  - [Project description in brief](#project-description-in-brief)
  - [Introduction](#introduction)
  - [Meta-learning](#meta-learning)
    - [What is meta-learning, and how is it different to ordinary supervised machine learning?](#what-is-meta-learning-and-how-is-it-different-to-ordinary-supervised-machine-learning)
    - [My contributions to meta-learning](#my-contributions-to-meta-learning)
  - [Improving data efficiency for supervised learning](#improving-data-efficiency-for-supervised-learning)
    - [Recognition as reconstruction](#recognition-as-reconstruction)
    - [MNIST classification using meta-learning and adaptive memories](#mnist-classification-using-meta-learning-and-adaptive-memories)
    - [So how well did this work in practise?](#so-how-well-did-this-work-in-practise)
    - [What next for this concept?](#what-next-for-this-concept)
  - [Endnote: other topics I researched using this repository](#endnote-other-topics-i-researched-using-this-repository)

## Project description in brief

- The ability to learn invariance (for example, scale, translation and rotational invariance in vision) is important in human intelligence and has the potential to have a big impact on machine learning (ML)
- A common work-around in the absence of invariance learning in ML models is data-augmentation, however an ML algorithm (EG a vision-based reinforcement learning algorithm) could not be considered truly intelligent unless it is able to learn new image categories online without requiring data-augmentation, which is something humans demonstrate every day
- If ML models were able to learn invariance, they would arguably be able to generalise more successfully in general, and therefore also be more data-efficient
- In this personal research project, I developed new approaches to invariance learning, based on new approaches to meta-learning which I also developed within this project
- The idea underlying these new approaches to invariance learning is as follows: imagine a random sample from the MNIST data-set of handwritten digits (for example, an image of the digit "3"), and imagine a function ![simple equation](https://latex.codecogs.com/png.image?\dpi{110}&space;\bg_white&space;f:\mathbb{R}^2\rightarrow[0,1] "simple equation") which returns the brightness of a given pixel (EG as a scalar in the range [0, 1]) as a function of the co-ordinates of that pixel in 2D space
- Now, the image which this function corresponds to can be thought of as a set 784 samples from this function, in which the input coordinates of these samples are arranged in a 28 x 28 grid
- Importantly, we could incorporate scale, translation and rotational invariance into this image representation simply by applying appropriate affine transformations to the 2D input points to the function ![simple equation](https://latex.codecogs.com/png.image?\dpi{110}&space;\bg_white&space;f "simple equation")
- If this function was represented by a multi-layer neural network, we could incorporate more general types of invariance in the image (EG line thickness) by modifying the parameters in all layers of the network
- This is where I incorporated meta-learning: in order to learn to classify images from MNIST, I used 10 meta-learning models (1 per class), and each meta-learning model is trained to reconstruct images (IE perform regression on brightness values as a function of the 2D coordinates of each pixel) from a particular class (EG one model reconstructs images of the digit "0", one model reconstructs images of the digit "1", etc)
- Each image is considered to be a data-set for regression, and each meta-learning model is trained on multiple images from its corresponding class, such that it can generalise to reconstruct unseen images from the class it was trained on
- The objective is for each meta-learning model to reconstruct images from the class it was trained on successfully, and reconstruct images from all other classes badly
- Classification of an unseen image can then be performed by applying a softmax function across the reconstruction losses of all 10 meta-learning models on the unseen image
- As part of this research project I also developed new approaches to meta-learning, which importantly learn a sense of parameter scale in addition to mean/initialisation parameter values, which can be used to define a probability distribution over task-specific parameters
- Some of the new meta-learning approaches I developed were based on using regularisation error functions, and I experimented with different regularisation error functions, including quadratic and multi-model quartic regularisation functions
- I also developed new approaches to meta-learning based on learning adaptive learning rates for each parameter in the model, for which the learning rates decay as those parameters get far from their mean initialisation values relative to the learned scale for each model parameter (this is instead of using regularisation error functions)
- My overall goal for the project was to use these approaches to perform MNIST classification with high accuracy, but using a very small fraction of the entire MNIST training set, IE to improve data-efficiency for learning MNIST

## Introduction

[MNIST](https://en.wikipedia.org/wiki/MNIST_database) is considered a solved problem by just about every researcher in ML. To me this is problematic because the training set for MNIST contains 60,000 training examples. If a human being required 60,000 examples of handwritten digits just to be able to recognise images of the numbers 0-9, wouldn’t you be concerned about that human being?

I find this problem especially interesting because one of the big drawbacks with state-of-the-art ML models is that they need an extreme amount of data in order to learn anything useful, and they don’t generalise the information that they do learn to new scenarios as well as humans, which is to say that these models have poor **data-efficiency**. They also typically aren’t capable of **learning invariance** in the same way as humans. In the long term I’d like to address these problems of data-efficiency and invariance-learning of ML models, and in the short term I decided that a simple task like MNIST would be a good place to start.

To this end, I developed this repository over the last 2 years as a basis for 2 main research objectives:
1. Develop new methods for meta-learning which model parameter scale in addition to initialisation values
1. Apply these new meta-learning methods using a novel approach to try to improve data-efficiency on the MNIST dataset of handwritten digits

(I also used this repository to work on 2 other minor research topics, discussed briefly in the [end-note](#endnote-other-topics-i-researched-using-this-repository) in this read-me).

## Meta-learning

### What is meta-learning, and how is it different to ordinary supervised machine learning?

Ordinary supervised machine learning involves training a model on a data-set so that the model can generalise to unseen data-points from the same distribution of data that it was trained on. However in meta-learning, instead of just being given one data-set (which from now on I will call a task), a meta-learning model is given a set of similar tasks, and the objective is to learn from each these tasks such that when the meta-learning model is given a new task which it hasn’t seen before (but is similar to the tasks that it trained on), it can adapt to that task more quickly and accurately than an ordinary machine learning algorithm learning that task from scratch, using the information it has learned from the tasks that it trained on.

The parameters that a meta-learning algorithm learns when adapting to a given task are called task-specific parameters. The parameters that the meta-learning algorithm learns from all the tasks that it trains on in order to help it adapt quickly to new tasks are called meta-parameters.

Meta-learning was a topic I researched during my 4th year research project with [Richard E Turner](http://www.eng.cam.ac.uk/profiles/ret26) in 2018-2019. During this project, 2 of the major papers/models I looked at were [MAML](https://arxiv.org/abs/1703.03400) and [Reptile](https://arxiv.org/abs/1803.02999). The common feature of both of these models is that the only meta-parameters they learn are optimal initial values for each parameter in the model, which are used as the initial parameters when the model tries to adapt to a new task. This set of meta-parameters can be thought of as the mean across tasks of the task-specific parameters from the tasks that the model used for training.

### My contributions to meta-learning

In my opinion, what’s missing from both of these models is a sense of **parameter scale**. Here are 2 examples of when this could be a problem:
- For some task-distributions, a large number of parameters might be required to learn each task, but only a small subset of parameters might need to vary in order to adapt to different tasks within the distribution
- You could have 2 task distributions which require the same mean parameter initialisations, but fitting samples from one task distribution requires the parameters to deviate much further from the mean parameters compared to the other task-distribution

In both cases, a knowledge of parameter scale is required to learn the true task-distribution and to adapt effectively to unseen tasks within that distribution, and seeing as both the MAML and Reptile algorithms only learn initialisation values for parameters and neither learns a sense of parameter scale, they will be incapable of demonstrating optimal performance on such task distributions.

I addressed this problem by developing meta-learning models which learn a scale for each parameter in addition to a mean initialisation value. There were 2 approaches I took to learning parameter scale:
- Using regularisation: a regularisation term was added to the objective function, which penalises the task-specific parameters from deviating too far from their mean value, based on a scale parameter which is learned for each model-parameter, according to how far that model-parameter usually deviates from its mean value across training tasks. There were 2 types of regularisation functions I tried to use:
  - Quadratic regularisation function: this consisted of a multivariable diagonal quadratic regularisation function, centred at the mean meta-parameters, with a scale which is learned for each model-parameter. While this approach is simple to understand and implement, it has a problem, which is that the gradient of the regularisation function always points towards the initialisation parameters, which can cause instability if during every iteration, each parameter is encouraged to converge closer and closer to the mean meta-parameters, and the scale parameter gets smaller and smaller, causing a positive feedback loop and ending up with the scale parameters tending towards zero and the task-specific parameters being exactly equal to the mean initialisation meta-parameters
  - Quartic regularisation function: this was similar to the quadratic regularisation function, but used a multi-modal quartic function of the form `(a*(x - m)^2 -1)^2`, where x is a task specific parameter, m is a mean/initialisation parameter, and a is an inverse scale parameter (see graph below for a comparison of the quartic regularisation function with the quadratic regularisation function in 1D). This ensured that parameters were encouraged to deviate from their initialisation values (unlike with the quadratic regularisation function), and that whichever direction a task specific parameter travels away from its initialisation value, there will be a local minimum in the regularisation function at a fixed distance away from the initialisation value, which is chosen according to how much that parameter normally deviates away from its mean when the meta-learning model is adapting to other tasks from the task distribution
- Using variable learning rates for each parameter: in this approach I did not modify the objective function for the tasks by adding a regularisation function. Instead, I used the scale parameter learned for each model-parameter to modify the learning rate of each model-parameter, such that model-parameters which have a high variance across tasks would have a relatively high learning rate when adapting to a new task, and model-parameters with a low variance across tasks would have a relatively low learning rate. I also experimented with decaying the learning rate of any model-parameter when it reaches a certain number of standard-deviations away from its mean initialisation value (the reason for this will become clear in the next section). I called this approach the Eve algorithm because it encourages high learning rates for model-parameters which vary significantly across tasks, which is in contrast to the [Adam algorithm](https://arxiv.org/abs/1412.6980), which dampens the learning rate of parameters which have large gradients over multiple time steps

![image](https://raw.githubusercontent.com/jakelevi1996/backprop2/master/Results/Images%20for%20Readme/Comparison%20of%201D%20regularisation%20functions.png "Comparison of 1D regularisation functions")

After designing and implementing these algorithms, I implemented some very simple toy meta-learning problems on which I tested my new algorithms in order to verify that they worked, and then moved on the next stage of my research. Ideally at this stage I would have run thorough experiments to demonstrate the efficacy of each of these meta-learning algorithms and compared them with MAML and Reptile before moving onto the next stage of research, but I didn’t, because I was running out of time (it’s hard trying to sustain a part-time research project on top of a full-time job), and I was excited to get to the next stage of my research project!

## Improving data efficiency for supervised learning

### Recognition as reconstruction

In this section I will explain my concept for using meta-learning to improve data-efficiency for supervised learning. Given an example image from MNIST, the usual approach to ML for supervised image processing is to represent the pixel intensities in the image as a vector (with one element in the vector for each pixel in the image), and pass that vector through a CNN, which has as many inputs as there are pixels in the image (28*28 = 784 in the case of MNIST).

My approach here is to treat an image from MNIST as a set of 784 data points, with 2D inputs which are the coordinates of each pixel, and 1D outputs which are the brightness of the pixel at the given input coordinate. Now, given such an image (which we treat as a dataset), we can perform regression to learn the brightness of each pixel as a function of each coordinate in the image (see images below for examples), using reconstruction loss as an objective function. Once we have the parameters of a model (EG a simple feedforward neural network) that has learned to reconstruct the given image, the parameters of this model effectively encode that image.

Furthermore, we can model invariance simply by changing the appropriate parameters in the model: translation by changing the input layer biases, rotation and scaling by appropriately changing the input layer weights, brightness by changing the output layer weights, and background brightness by changing the output layer biases.

![image](https://github.com/jakelevi1996/backprop2/raw/master/Results/Images%20for%20Readme/Hidden%20activations%20meta-learning%203%20-%20collage.png "Meta-learning model adapted to 3 different examples of an image of a 3")

### MNIST classification using meta-learning and adaptive memories

But how do we use this approach to perform classification on images from MNIST? **This is where meta-learning comes in**. Say we train one meta-learning model to fit all images of a 3 from MNIST (the set of all images of a 3 from MNIST here becomes our task-set). Then the mean initialisation meta-parameters would form a starting point from which an unseen image of a 3 can be quickly reconstructed, and the scale meta-parameters would describe how much each parameter is likely to vary from its mean initialisation value in order to model this unseen image of a 3.

Importantly, we would hope that if we gave an image of a 2 to the meta-learning model which has learned to reconstruct images of the number 3, it wouldn’t be able to reconstruct the image of the 2 very well, because it would require the parameters from the meta-learning model to adapt substantially differently to how they normally reconstruct images of the number 3, and the scale parameters would prevent that from happening (either through regularisation loss, or through decaying learning rates in the case of adaptive learning rates).

So now say we train one meta-learning model for each different class of digit (so 10 meta-learning models in total in the case of MNIST), and say we are given a test image that the models haven’t seen before. In order to classify it, we let every meta-learning model try to adapt to that image, and if everything goes according to plan, only the meta-learning model corresponding to the correct image class will be able to adapt effectively to that image, and achieve a low reconstruction loss.

So to decide which class the test image came from, we just choose the meta-learning model with the lowest adapted reconstruction loss. If we wanted a probability distribution over classes, we simply put all 10 of the reconstruction losses from each meta-learning model through a softmax function.

An interesting interpretation is that we can think of each of these meta-learning models as forming an adaptive memory for each digit. The parameters of each digit-model are all distinct from those of the other models, and each model reconstructively encodes the class which it represents. The scale-parameters within a digit-model encode the observed variance within the associated class, and allow the model to adapt to differences between memory and observation in terms of location, rotation, scale, brightness, and more abstract forms of variance. Adaptively reconstructing a particular observation of an unseen digit from MNIST using the appropriate digit-model can be thought of as 'remembering' the class from which that image came.

### So how well did this work in practise?

Unfortunately I got stuck at the stage of trying to make the meta-learning algorithm adapt well to the class of images that it was trained on, and adapt badly to classes of images that it was not trained on. In the case of both approaches to meta-learning that I developed, there is a single hyper-parameter which determines how strong is the effect of the meta-learning model vs ordinary supervised learning of the pixel intensities as a function of the 2D coordinates of each pixel. In the case of regularisation loss, this is the relative scale of the regularisation loss compared to the reconstruction loss, and in the case of the adaptive learning rates, this is how quickly the learning rate decays as the task-specific parameters deviate from their mean initialisation values.

In both cases, I found that at one extreme, the meta-learning model for a given class would reconstruct all images really well (whether a test image is from the correct class or not), and at the other extreme, the meta-learning model for a given class would reconstruct all images really badly (even if the test image was from the correct class). I couldn’t find a sweet spot at which the meta-learning model for a given class would be able to discriminate between images from the correct class and images from the other classes, and unfortunately I’ve run out of time to keep trying for the time being.

### What next for this concept?

What does this mean for my idea of using meta-learning to improve data-efficiency on MNIST? It could be that the idea was flawed to begin with, and was never going to work. It could be that I was really close to getting this thing working, and just needed to tweak one or two hyper-parameters. Or it could be somewhere in between, that the idea would work eventually, but it would still take a decent amount of work. Personally, I still think that the idea is a good one in principle (despite unfortunately not seeing the results I wanted to see), but I’m not planning to continue developing it in the near future.

This is because my original motivation for researching this was wanting to match human level performance in terms of data-efficiency and invariance learning, and these are still very much goals that I want to achieve. But my intuition has shifted recently towards thinking that if we want to achieve human level performance in machine learning, then it makes sense to try and understand the proprties and dynamics of human learning more thoroughly from a computational perspective, and try and make our computational models more realistic in comparison to the proprties and dynamics of human learning.

For all we know, it may well be that some features of biological brains (feaures which we might already know from neuroscience) might actually be fundamental to their greater success compared to state-of-the-art ML models. This is why *biological plausibility in ML* is what I want to focus on in my research moving forwards.

## Endnote: other topics I researched using this repository

I also used the repository to investigate automatic convergence metrics for meta-learning inner-loops (instead of using a fixed number of inner-loop iterations, which may well be sub-optimal), and (somewhat unrelated to the other objectives) research 2nd order optimisation methods, using block-diagonal Hessian matrices of a fixed block-size (to preserve the complexity of the algorithm as the number of parameters increases), and using eigenvalue decompositions to regularise the Hessian matrices to avoid the problem of non-convexity (which I proved can be motivated in terms of escaping non-convex regions of the objective function in the direction of inflection points). I actually first used this repository to start researching 2nd-order optimisation algorithms, hence the name "backprop2", and never got round to changing the name or starting a new repository as I moved on to other research topics.
