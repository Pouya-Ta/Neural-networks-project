# Task 5
import numpy as np

from rsdl import Tensor
from rsdl.layers import Linear
from rsdl.optim import SGD
from rsdl.losses import loss_functions

X = Tensor(np.random.randn(100, 3))
coef = Tensor(np.array([-7, +3, -9]))
y = X @ coef + 5

# TODO: define a linear layer using Linear() class  
l = Linear(3, 1)  # Assuming 3 input features and 1 output feature

# TODO: define an optimizer using SGD() class
optimizer = SGD(l.parameters(), lr=0.01)  # Adjust learning rate as needed

# TODO: print weight and bias of linear layer
print("Initial weights:", l.weights.data)
print("Initial bias:", l.bias.data)

learning_rate = 0.01
batch_size = 10  # Adjust batch size as needed

for epoch in range(100):

    epoch_loss = 0.0

    for start in range(0, 100, batch_size):
        end = start + batch_size

        print(start, end)

        inputs = X[start:end]

        # TODO: predicted
        predicted = l(inputs)

        actual = y[start:end]
        actual.data = actual.data.reshape(batch_size, 1)
        # TODO: calculate MSE loss
        loss = loss_functions.mean_squared_error(predicted, actual)

        # TODO: backward
        loss.backward()

        # TODO: add loss to epoch_loss
        epoch_loss += loss.data

        # TODO: update w and b using optimizer.step()
        optimizer.step()

    # Print epoch loss after each epoch
    print(f"Epoch {epoch + 1}, Loss: {epoch_loss}")

# TODO: print weight and bias of linear layer
print("Final weights:", l.weights.data)
print("Final bias:", l.bias.data)
