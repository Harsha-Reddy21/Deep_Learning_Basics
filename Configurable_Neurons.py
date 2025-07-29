import random
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def relu(x):
    return max(0,x)


n=int(input("Enter the number of inputs "))
h=int(input("Enter the number of hidden neurons "))


activation_name=input("Enter activation (sigmoid/relu): ").strip().lower()



if activation_name=="sigmoid":
    activation=sigmoid
elif activation_name=="relu":
    activation=relu
else:
    print("Invalid activation function")
    exit()



inputs=[round(random.uniform(-1,1),2) for _ in range(n)]

hidden_weights=[[round(random.uniform(-1,1),2) for _ in range(n)] for _ in range(h)]
hidden_bias=[round(random.uniform(-1,1),2) for _ in range(h)]

hidden_outputs=[]
for i in range(h):
    z=sum(x*w for x,w in zip(inputs,hidden_weights[i]))+hidden_bias[i]
    a=round(activation(z),2)
    hidden_outputs.append(a)

output_weights=[round(random.uniform(-1,1),2) for _ in range(h)]
output_bias=round(random.uniform(-1,1),2)


z_out=sum(h*w for h,w in zip(hidden_outputs,output_weights))+output_bias
final_output=round(activation(z_out),2)

print(f"Inputs: {inputs}")
print(f"Hidden Weights: {hidden_weights}")
print(f"Hidden Bias: {hidden_bias}")
print(f"Hidden Outputs: {hidden_outputs}")
print(f"Output Weights: {output_weights}")
print(f"Output Bias: {output_bias}")
print(f"Final Output: {final_output}")







