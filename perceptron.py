import random
import torch
import torch.nn as nn
from torch import Tensor
import matplotlib.pyplot as plt
import numpy as np

class Perceptron():
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def predict(self, x: Tensor):
        activation = torch.dot(self.w, x) + self.b
        y_hat = 1 if activation.item() >= 0 else 0
        return y_hat
    
    def predict(self, x):
        with torch.no_grad():
            activation = torch.dot(self.w,x)
            y_hat = 1 if activation.item() >= 0 else 0
            return y_hat

#Perceptron Learning Algorithm
def plot_perceptron(neuron, max_x, min_x, x_coords_0, y_coords_0, x_coords_1, y_coords_1):
        # Create the plot
        plt.scatter(x_coords_0, y_coords_0, c='red', label='Class 0')
        plt.scatter(x_coords_1, y_coords_1, c='blue', label='Class 1')

        # Add titles and labels
        plt.title('Points')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.grid(True)  # Optional: add a grid

        # Plot Weight Vector
        plt.arrow(0,0,neuron.w[0], neuron.w[1], width=0.02, head_width=0.05)

        #Calculate the boundary line variables
        if neuron.w[0] != 0 and neuron.w[1] != 0:
            vector_slope = neuron.w[1]/neuron.w[0]
            bound_slope = -1/vector_slope
            x = np.linspace(min_x, max_x, num=1000)
            y = bound_slope * x

            #Plot the line
            plt.plot(x, y)
        elif neuron.w[0] != 0 and neuron.w[1] == 0:
            #Plot the vertical line
            plt.axvline(x=0)
        elif neuron.w[0] == 0 and neuron.w[1] != 0:
            #Plot the horizontal line
            plt.axhline(y=0)

        #Show the graph
        plt.show()
 

#DATA, REPLACE WHEN WE HAVE REAL DATA
#Data
data=torch.tensor([(-1, -1), (-1, -0.5), (-1, 0), (-1, 0.5), (1, 0.5), (1, -0.5), (1, 1), (1, -1)],dtype=float)
label=torch.tensor([1,1,1,1,0,0,0,0], dtype=float)

min_x = min(data[:, 0])
max_x = max(data[:, 0])

data_size=len(data)
data_sum=torch.sum(data,dim=0)
data_mean=data_sum/data_size

x_coords_0 = [point[0] for point, label in zip(data, label) if label == 0]
y_coords_0 = [point[1] for point, label in zip(data, label) if label == 0]
x_coords_1 = [point[0] for point, label in zip(data, label) if label == 1]
y_coords_1 = [point[1] for point, label in zip(data, label) if label == 1]

def Finding_weight():
        weight = torch.tensor([random.random(), random.random()], requires_grad=True, dtype=float)
        learning_rate = 0.5
        neuron = Perceptron(weight)

        if neuron != None:
            print('Initial Plot')
            with torch.no_grad():
                print(f"{neuron.w.tolist()} - {learning_rate}*{neuron.w.grad}")
                plot_perceptron(neuron, min_x, max_x, x_coords_0, y_coords_0, x_coords_1, y_coords_1)

        for epoch in range (5):
            old_w = neuron.w
            all_correct = True
            for i in range(len(data)):
                point = data[i]
                y = neuron.forward(point)

                # neuron update!
                loss_val = loss_fn(y, label[i])

                # Gradient calculation + weight update
                loss_val.backward()

                with torch.no_grad():
                    # store old w
                    old_w = neuron.w.detach().numpy().copy()

                    # update w
                    neuron.w -= lr * neuron.w.grad

                    # Only Print Equation when prediction is wrong
                    if label[i] != y_hat:
                        print(f"{neuron.w.tolist()} - {learning_rate}*{neuron.w.grad}")

                # Only Make a New Plot when the weights change
                if label[i] != y_hat:
                    with torch.no_grad():
                        plot_perceptron(neuron, max_x, min_x, x_coords_0, y_coords_0, x_coords_1, y_coords_1)

                # Break loop if perceptron predicts everything correct
                with torch.no_grad():
                    for i in range(len(data)):
                        point = data[i]
                        y_hat = neuron.predict(point)
                        if y_hat != label[i]:
                            all_correct = False
                    if all_correct:
                        break
            if all_correct:
                break    


            