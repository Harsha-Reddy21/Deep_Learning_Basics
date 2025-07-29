import math 


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


x1,x2=map(float,input("Enter the values of x1 and x2: ").split())
w1,w2=map(float,input("Enter the values of w1,w2 ").split())

b=float(input("Enter bias: "))
              
              
z=x1*w1+x2*w2+b
y=sigmoid(z)
print(f"The output of the neuron is: {y}")
              
              
              
              
              
              