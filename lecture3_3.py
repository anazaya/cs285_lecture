import torch
import numpy as np
import matplotlib.pyplot as plt

x_np = np.ones((2,3))
x = torch.ones(2,3)
y = torch.zeros(2,3)
#print(x,"\n",y)
#print(torch.sum(x, dim = 0))
x_like = torch.ones_like(x)
y_like = torch.zeros_like(y)
#print(x_like)
#print(y_like)

x_tensor = torch.from_numpy(x_np).to(torch.float32)
#print(x_tensor)
x_np[1,2] = 3
#print(x_np)
#print(x_tensor)
#   x_type = x_tensor.to(torch.float32)
#print(x_tensor.dtype)
#print(x_type.dtype)
x_numpy = x_like.numpy()
#print(type(x_like))
#print(type(x_numpy))

#random_array = np.random.rand(5)
#random_tensor = torch.from_numpy(random_array).to(torch.float32)
random_tensor = torch.linspace(-10,10,10).to(torch.float32)
relued_tensor = torch.relu(random_tensor)
sigmoid_tensor = torch.sigmoid(random_tensor)
tanh_tensor = torch.tanh(random_tensor)
softmax_tensor = torch.softmax(random_tensor, dim = 0)
#print(random_tensor)
#print(relued_tensor)
#print(sigmoid_tensor)
#print(tanh_tensor)

#plt.plot(random_tensor.numpy(),softmax_tensor.numpy())
#plt.show()

# -----------------------Automatic Differentiation-------------------------

shape = (3, )
x = torch.tensor([1,2,3], dtype = torch.float32, requires_grad=True)
y = torch.ones(shape)
y.requires_grad = True

#print(x.data, "&", x.grad, "\n", y.data, "&", y.grad)

intermediate = (2*x + y).sum()
loss = ((intermediate)**2).sum()
loss.backward()
#intermediate.backward()
#print(x.grad, "\n", y.grad)
#print(y.data_ptr())
y_detached = y.detach().numpy()
y_cloned = y.detach().clone()
print(y is y_detached)
print(y is y_cloned)
print(y_detached.__array_interface__["data"][0])
print(y_cloned.data_ptr())

"""
grad_fns = [(loss.grad_fn, 0)]
current_level = 0
lines = []
while grad_fns :
    previous_level = current_level
    fn, current_level = grad_fns.pop()
    if current_level != previous_level :
        print("====")
    print(fn.name())
    for next_fn, _ in fn.next_functions :
        if next_fn :
            grad_fns.append((next_fn, current_level + 1))
"""
