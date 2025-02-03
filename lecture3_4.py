# Performing Regression task
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim

x = torch.linspace(-5,5,100).view(-1,1)
    # Batch formatting : Batch(first argument), Actual data input(second argument)
y_target = torch.sin(x)
loss_fn = nn.MSELoss()

#plt.plot(x, y_target)

class regression_net(nn.Module) :
    def __init__(self,input_size, output_size) :
        super(regression_net,self).__init__()
        self.fc1 = nn.Linear(input_size,32)
        self.fc2 = nn.Linear(32,32)
        self.fc3 = nn.Linear(32,output_size)
    
    def forward(self,x) :
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

regress_net = regression_net(input_size = 1, output_size = 1)
#print(regress_net)

#print(y_predict.shape)

"""
for _, para in regress_net.named_parameters() :
    print(para.shape)
"""
#print(regress_net.fc1.bias.grad)
#print(regress_net.fc1.weight.grad)
"""
for _ in range(100):
    y_predict = regress_net(x)
    loss = ((y_target-y_predict)**2).sum()
    loss.backward()
    for p in regress_net.parameters() :
        p.data.add_(-0.001*p.grad)
        p.grad.data.zero_()
"""
optimizer = optim.Adam(regress_net.parameters(), lr = 1e-3)
loss_fn = nn.MSELoss()

for _ in range(100) :
    y_predict = regress_net(x)
    loss = loss_fn(y_target, y_predict)

    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

plt.plot(x.detach().numpy(), y_predict.detach().numpy())
plt.plot(x.detach().numpy(), y_target.detach().numpy())
plt.xlim(-5,5)
plt.ylim(-1,1)
plt.grid()
plt.show()


print(torch.cuda.is_available())
    # False