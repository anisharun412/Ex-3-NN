<H3>ENTER YOUR NAME : Arunsamy D</H3>
<H3>ENTER YOUR REGISTER NO : 212224240016</H3>
<H3>EX. NO.3</H3>
<H2 aligh = center> Implementation of MLP for a non-linearly separable data</H2>
<h3>Aim:</h3>
To implement a perceptron for classification using Python
<H3>Theory:</H3>
Exclusive or is a logical operation that outputs true when the inputs differ.For the XOR gate, the TRUTH table will be as follows:

XOR truth table
![Img1](https://user-images.githubusercontent.com/112920679/195774720-35c2ed9d-d484-4485-b608-d809931a28f5.gif)

XOR is a classification problem, as it renders binary distinct outputs. If we plot the INPUTS vs OUTPUTS for the XOR gate, as shown in figure below

![Img2](https://user-images.githubusercontent.com/112920679/195774898-b0c5886b-3d58-4377-b52f-73148a3fe54d.gif)

The graph plots the two inputs corresponding to their output. Visualizing this plot, we can see that it is impossible to separate the different outputs (1 and 0) using a linear equation.To separate the two outputs using linear equation(s), it is required to draw two separate lines as shown in figure below:
![Img 3](https://user-images.githubusercontent.com/112920679/195775012-74683270-561b-4a3a-ac62-cf5ddfcf49ca.gif)
For a problem resembling the outputs of XOR, it was impossible for the machine to set up an equation for good outputs. This is what led to the birth of the concept of hidden layers which are extensively used in Artificial Neural Networks. The solution to the XOR problem lies in multidimensional analysis. We plug in numerous inputs in various layers of interpretation and processing, to generate the optimum outputs.
The inner layers for deeper processing of the inputs are known as hidden layers. The hidden layers are not dependent on any other layers. This architecture is known as Multilayer Perceptron (MLP).
![Img 4](https://user-images.githubusercontent.com/112920679/195775183-1f64fe3d-a60e-4998-b4f5-abce9534689d.gif)
The number of layers in MLP is not fixed and thus can have any number of hidden layers for processing. In the case of MLP, the weights are defined for each hidden layer, which transfers the signal to the next proceeding layer.Using the MLP approach lets us dive into more than two dimensions, which in turn lets us separate the outputs of XOR using multidimensional equations.Each hidden unit invokes an activation function, to range down their output values to 0 or The MLP approach also lies in the class of feed-forward Artificial Neural Network, and thus can only communicate in one direction. MLP solves the XOR problem efficiently by visualizing the data points in multi-dimensions and thus constructing an n-variable equation to fit in the output values using back propagation algorithm

<h3>Algorithm :</H3>

Step 1 : Initialize the input patterns for XOR Gate<BR>
Step 2: Initialize the desired output of the XOR Gate<BR>
Step 3: Initialize the weights for the 2 layer MLP with 2 Hidden neuron  and 1 output neuron<BR>
Step 3: Repeat the  iteration  until the losses become constant and  minimum<BR>
    (i)  Compute the output using forward pass output<BR>
    (ii) Compute the error<BR>
	(iii) Compute the change in weight ‘dw’ by using backward progatation algorithm. <BR>
    (iv) Modify the weight as per delta rule.<BR>
    (v)  Append the losses in a list <BR>
Step 4 : Test for the XOR patterns.

<H3>Program:</H3>

### Import Libraries
```py
import numpy as np
import matplotlib.pyplot as plt
```

### Initialize the input vector and output vector for XOR
```py
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

Y = np.array([
    [0],
    [1],
    [1],
    [0]
])

print("Input:\n", X)
print("\nOutput:\n", Y)
     
Input:
 [[0 0]
 [0 1]
 [1 0]
 [1 1]]

Output:
 [[0]
 [1]
 [1]
 [0]]
```
### Initialize the structure of  MLP with input ,hidden  and output layer
```py
input_neurons = 2
hidden_neurons = 2
output_neurons = 1

learning_rate = 0.1
epochs = 10000

print("Input Neurons :", input_neurons)
print("Hidden Neurons:", hidden_neurons)
print("Output Neurons:", output_neurons)

     
Input Neurons : 2
Hidden Neurons: 2
Output Neurons: 1
```

### Weight matrix for hidden layer randomly
```py
np.random.seed(42)

W1 = np.random.randn(input_neurons, hidden_neurons)
B1 = np.zeros((1, hidden_neurons))

W2 = np.random.randn(hidden_neurons, output_neurons)
B2 = np.zeros((1, output_neurons))

print("W1:\n", W1)
print("\nW2:\n", W2)

     
W1:
 [[ 0.49671415 -0.1382643 ]
 [ 0.64768854  1.52302986]]

W2:
 [[-0.23415337]
 [-0.23413696]]
```

### Sigmoid Activation function
```py
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Sigmoid Derivative
def sigmoid_derivative(x):
    return x * (1 - x)
```
     
### Forward Propagation
```py
def forward_propagation(X, W1, B1, W2, B2):

    hidden_input = np.dot(X, W1) + B1

    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W2) + B2

    predicted_output = sigmoid(final_input)

    return hidden_output, predicted_output
``` 

### Backward Propagation
```py
def backward_propagation(
        X,
        Y,
        hidden_output,
        predicted_output,
        W2):

    d_output = (
        Y - predicted_output
    ) * sigmoid_derivative(predicted_output)

    hidden_error = np.dot(
        d_output,
        W2.T
    )

    d_hidden = (
        hidden_error
    ) * sigmoid_derivative(hidden_output)

    return d_output, d_hidden
```

### Update Wrights
```py
def update_weights(
        X,
        hidden_output,
        d_output,
        d_hidden,
        W1,
        B1,
        W2,
        B2,
        learning_rate):

    W2 += np.dot( hidden_output.T, d_output ) * learning_rate

    B2 += np.sum( d_output, axis=0, keepdims=True ) * learning_rate

    W1 += np.dot( X.T, d_hidden ) * learning_rate

    B1 += np.sum( d_hidden, axis=0, keepdims=True ) * learning_rate

    return W1, B1, W2, B2
     
```

### Train the MLP
```PY
def train_network():

    global W1, B1, W2, B2

    losses = []

    for epoch in range(epochs):

        hidden_output, predicted_output = forward_propagation(
            X,
            W1,
            B1,
            W2,
            B2
        )

        error = Y - predicted_output

        loss = np.mean(error ** 2)

        losses.append(loss)

        d_output, d_hidden = backward_propagation(
            X,
            Y,
            hidden_output,
            predicted_output,
            W2
        )

        W1, B1, W2, B2 = update_weights(
            X,
            hidden_output,
            d_output,
            d_hidden,
            W1,
            B1,
            W2,
            B2,
            learning_rate
        )

    return losses, predicted_output

losses, predicted_output = train_network()

print("Training Completed")
```

### plot losses to see how our network is doing
```PY
plt.plot(losses)

plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.title("Loss vs Epochs")

plt.grid(True)
```

### Test the XOR classification
```PY
print("Predicted Output")
print(predicted_output)

binary_output = (predicted_output > 0.5).astype(int)
print("\nBinary Output")
print(binary_output)
```

<H3>Output:</H3>

### Initialize the input vector and output vector for XOR
<img width="1740" height="229" alt="image" src="https://github.com/user-attachments/assets/909fab6f-8403-46f3-9c2c-978eb2187fcc" />

### Initialize the structure of MLP with input, hidden  and output layer
<img width="1111" height="73" alt="image" src="https://github.com/user-attachments/assets/328a9ea8-b58b-4f44-bedc-5e6f6ffce83e" />

###  Weight matrix for hidden layer randomly
<img width="1422" height="157" alt="image" src="https://github.com/user-attachments/assets/3b448425-5303-4aa4-abb0-660fd70193dc" />

### Training
<img width="1049" height="172" alt="image" src="https://github.com/user-attachments/assets/f122eabe-c902-4cc4-86ff-f2b6dd8a12c0" />


### plot losses to see how our network is doing
<img width="576" height="455" alt="download" src="https://github.com/user-attachments/assets/2e0f15a6-fe4d-46bd-9dd6-ad4b730a5a7e" />

### Test the XOR classification
<img width="1055" height="233" alt="image" src="https://github.com/user-attachments/assets/892d29ed-d4e3-4b28-b7de-d055b95072df" />

<H3> Result:</H3>
Thus, XOR classification problem can be solved using MLP in Python 
