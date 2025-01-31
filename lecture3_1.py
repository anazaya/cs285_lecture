import numpy as np
x_1 = np.array([1,2,3])
x_2 = np.array([[4,5,6],[7,8,9]])
x_3 = np.random.rand(2,3,4)
print(f"{x_1}'s shape : {x_1.shape}")
print(f"{x_2}'s shape : {x_2.shape}")
print(f"{x_3}'s shape : {x_3.shape}")

print(f"x_2's 2nd row : {x_2[1]}")
print(f"x_3's 2nd dimension's 2nd to 3rd row : {x_3[1,1:3]}")