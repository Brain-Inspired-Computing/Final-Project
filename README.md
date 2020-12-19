# Research Project - Brain Inspired Computing Fall 2020

Goal with this research project is to develop a spiking binary classifier for detecting between residential and industrial labeled satellite images.

Our model uses unsupervised learning, more specifically creates a decoder from the training data then uses the decoder to decode the testing data.

Link to the pitch talk which highlights our idea in more detail: https://docs.google.com/presentation/d/1TsljP_CYUNU8F0qEeb-vXzG6FSWaU5R1PI5Xx1kBa4s/edit?usp=sharing

Link to term paper: https://docs.google.com/document/d/1MOwNo2fCIBoCrqcHnFHrbyNMeSEMaRIxoSYYLZOqx7Y/edit?usp=sharing

## Satellite Dataset 

Link to the EuroSat Dataset that we used for training and testing: https://github.com/phelber/EuroSAT

Split up the satellite dataset into training, validating and testing.

~The split ratio is: 70/15/15 (training/validating/testing)~

The split ratio is: 80/20 (training/testing)

Images are 64 x 64 (3 Channel rgb) 64x larger than the digit images in assignment 2

Picture of Industrial Satellite Image

![](Data/train/Industrial/Industrial_1.jpg?raw=true)

Picture of Residential Satellite Image

![](Data/train/Residential/Residential_1.jpg?raw=true)

We normalize the pixel values between the values of 0 to 1, and fetch our data dividing them into the images (x_..) and their one hot encoded feature category.

**LIF Equation**

$ \frac{dv}{dt} = \frac{1}{\tau_{rc}}(J-V)$

Using Euler's Method

$V_{t+1} = V_{t} + dt * \frac{dv}{dt}$

## Training

N.B.:

**Encoding pixel to current**

$J_M(x) = \alpha x + J^{bias}$
- $J_M$ is all past current input since *x*
- $\alpha$ is both the gain and a unit conversion factor
- $J^{bias}$ accounts for the steady background input to the cell

$J_i(x) = \alpha_i \langle x,e_i\rangle + J^{bias}_i$
- $e_i$ - random values normalised to unit length (represents the neuron's encoding or "preferred direction vector"
- $\langle\,\cdot,\cdot\rangle$ denotes the dot product between two vectors

---

**Decoding (w.o. Noise)**



$\hat{x} = Da$

$D^T \approx (A A^T)^{-1}AX^T$

```
"""
Decoder Python Code
d : Dimensionality of the value represented by a neuron population
n : Number of neurons in a neuron population
N : Number of samples
"""
A = np.array(...) # n x N array
X = np.array(...) # d x N array
D = np.linal.lstsq(A.T, X.T)[0].T
```
More Info: [Moore-Penrose Inverse](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse)

**Decoding (w. Noise)**
> Biological neural systems are subject to significant amounts of noise from various sources. Any analysis of such systems must take the effects of noise into account.

The idea to introduce noise into our model is by applying regularisation factors to the decoder.

$D^T \approx (A A^T + N \sigma^2 I)^{-1} A X^T$, where $I$ is the $n$ $x$ $n$ identity matrix

```
"""
Decoder Python Code
"""
A = np.array(...) # n x N array
X = np.array(...) # d x N array
D = np.linal.lstsq(A @ A.T + 0.5 * N * np.square(sigma) * np.eye(n), A @ X.T)[0].T
```

---

**Computer Firing Rate from Current**


- $a(x)$ is the firing rate at time x
- $\tau^{ref}$ refractory period time constant
- $\tau^{RC} $ Membrane time constant (aka the characteristic time or relaxation time)
- $J_{th}$ threshold current

- $a(x) = \frac{1}{\tau^{ref} - \tau^{RC}ln(1-\frac{J_{th}}{J_M(x)})}$
- assume $J_{th}$ = 1

$$
G(J) = \begin{cases} 
      a(x) = \frac{1}{\tau^{ref} - \tau^{RC}ln(1-\frac{1}{J_M(x)})} & \text{if }{J_M(x)} > 1 \\
      0 & \text{otherwise}
\end{cases}
$$


Resources: 
1. http://compneuro.uwaterloo.ca/courses/syde-750.html

**Post-synaptic Current Filter**

use case is to filter the spike train from the LIF neuron

$
h(t) = \begin{cases} 
      c^{-1} t^n e^{-\frac{t}{\tau}} & \text{if }{t} \ge 0 \\
      0 & \text{otherwise}
\end{cases}
$
*where n is a non-negative integer, and c normalizes the filter to area one to preserve energy*

$c = \int_{0}^\infty t^n e^{-\frac{t}{\tau}} dt$

## Results

![](results.png?raw=true)

<br>

## References

- Ali Safa. (2020, July 19). Digit Classification Using a One-Hidden-Layer Spiking Neural Network (Version V1.3). Zenodo. http://doi.org/10.5281/zenodo.3951585
- Eliasmith, Chris, and Charles H. Anderson. Neural Engineering: Computation, Representation, and Dynamics in Neurobiological Systems. Mitt Press, 2003. 
- Eric Hunsberger, Chris Eliasmith: “Spiking Deep Networks with LIF Neurons”, 2015; <a href='http://arxiv.org/abs/1510.08829'>arXiv:1510.08829</a>.
- Runchun Wang, Chetan Singh Thakur, Tara Julia Hamilton, Jonathan Tapson, Andre van Schaik: “A neuromorphic hardware architecture using the Neural Engineering Framework for pattern recognition”, 2015; <a href='http://arxiv.org/abs/1507.05695'>arXiv:1507.05695</a>.
- J. Dethier, V. Gilja, P. Nuyujukian, S. A. Elassaad, K. V. Shenoy, & K. Boahen (2011). Spiking neural network decoder for brain-machine interfaces. In 2011 5th International IEEE/EMBS Conference on Neural Engineering (pp. 396-399).
- Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. Patrick Helber, Benjamin Bischke, Andreas Dengel, Damian Borth. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2019.
- Introducing EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification. Patrick Helber, Benjamin Bischke, Andreas Dengel. 2018 IEEE International Geoscience and Remote Sensing Symposium, 2018.
