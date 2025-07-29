import random
import math 

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


inputs=[round(random.uniform(-1,1),2) for _ in range(3)]

hidden_weights=[[round(random.uniform(-1,1),2) for _ in range(3)] for _ in range(2)]
hidden_bias=[round(random.uniform(-1,1),2) for _ in range(2)]


# print(f"Inputs: {inputs}")
# print(f"Hidden Weights: {hidden_weights}")
# print(f"Hidden Bias: {hidden_bias}")

hidden_outputs=[]
for i in range(2):
    z=sum(x*w for x,w in zip(inputs,hidden_weights[i]))+hidden_bias[i]
    a=round(sigmoid(z),2)
    hidden_outputs.append(a)

# print(f"Hidden Outputs: {hidden_outputs}")

output_weights=[round(random.uniform(-1,1),2) for _ in range(2)]
output_bias=round(random.uniform(-1,1),2)


z_final=sum(h*w for h,w in zip(hidden_outputs,output_weights))+output_bias

final_output=round(sigmoid(z_final),3)

print(f"Final Output: {final_output}")