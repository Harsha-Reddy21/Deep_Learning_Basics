import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0,x)


def leaky_relu(x):
    return np.maximum(0.01*x,x)

activation_functions={
    "sigmoid":sigmoid,
    "tanh":tanh,
    "relu":relu,
    "leaky_relu":leaky_relu
}

n_inputs=np.random.randint(3,7)
n_hidden=np.random.randint(1,4)
layer_sizes=[np.random.randint(2,6) for _ in range(n_hidden)]
layer_sizes.append(1)

input_vector=np.round(np.random.uniform(-10,10,n_inputs),1)

weights=[]
biases=[]


prev_size=n_inputs

for size in layer_sizes:
    w=np.round(np.random.uniform(-1,1,(size,prev_size)),2)
    b=np.round(np.random.uniform(-1,1,size),2)
    weights.append(w)
    biases.append(b)
    prev_size=size


def forward_pass(input_vec, weights, biases, activation_fn):
    a=input_vec.reshape(-1,1)
    for i in range(len(weights)):
        z=np.dot(weights[i],a)+biases[i]
        a=activation_fn(z)

    return float(np.round(a[0][0],3))


outputs={}
for name, fn in activation_functions.items():
    y=forward_pass(input_vector,weights,biases,fn)
    outputs[name]=y


print("Random Seed: 42\n")
print("Generated Network:")
print(f"- Input Features: {n_inputs} → Values: {list(input_vector)}")
print(f"- Hidden Layers: {n_hidden}")
for i, size in enumerate(layer_sizes[:-1]):
    print(f"  • Layer {i+1}: {size} neurons")
print(f"- Output Layer: 1 neuron\n")

print("Final Outputs:")
for name in activation_functions:
    print(f"- {name}: [{outputs[name]}]")



plt.figure(figsize=(8, 5))
plt.bar(outputs.keys(), outputs.values(), color=["#e76f51", "#f4a261", "#2a9d8f", "#264653"])
plt.title("Final Output by Activation Function")
plt.ylabel("Output Value")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
