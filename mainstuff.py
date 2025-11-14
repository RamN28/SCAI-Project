import torch
import torch.nn as nn #neural network section of torch

class myNN(nn.Module):

def __init__(self): 
	self.layer1 = nn.Linear(2, 3)  #lets say we have 2 hidden layers 👍
			 #takes in certain amt of inputs and outputs
	self.layer2 = nn.Linear(3, 2)  # 3 inputs, 2 outputs
	self.layer3 = nn.Linear(2, 1)  # 2 inputs, 1 output
	#too many layers may be too much computing or overfit data
	
	self.relu = nn.ReLU()  #activation! Defines RELU function, more common 
# can also use nn.sigmoind 
	
#now need to call all the methods

def forward(self, input):
	#getting an input, just need to return output of function
	result = self.layer1(input)
	result = self.relu(result)	#need to make sure result1 are nonlinear

	result = self.layer2(result)
	result = self.relu(result)

	result = self.layer3(result)
	# dont need a relu at the end, since output values won’t go into anything, doesn’t need to be nonlinear
    return result 




########################################

model = myNN()

out = model((7,8))

loss_fn = nn.MSE()  #MSE(mean square error) and CE(cross entropy)
# loss_fn = nn.CrossEntropyLoss()  #CE (cross entropy) - for classification 
optim = ADAM(0.01)
#too low - model gets something wrong, but barely changes weights
#too high - model gets something wrong, changes everything about model


#step 2: score
for i in range(100):
predict = model(train_input)
loss = loss_fn(pred, train_output) #good or bad our predictions are


#step 3: learn (don’t use this when testing)
optimizer.zero_grad()
loss.backward() #how much to adjust each weight
optimizer.step() 


#yay, fully trained model!


#Can use other layers besides linear layers 

with torch.no_grad
Test loop:
Same as train, but use test data and no learn step
Convolution (conv2d) for images instead of linear
self.layer1 = nn.conv2d(2, 3)

