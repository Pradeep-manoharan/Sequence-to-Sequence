import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self,inputs_size,hidden_size,output):
        super().__init__()
        self.grn = nn.GRU(inputs_size,hidden_size,batch_first=True)
        self.fc1 = nn.Linear(hidden_size,output)

    def forward(self,x):

        gru_out, _  = self.grn(x)

        last_hidden_state = gru_out[:,-1,:]

        # Apply fully connected the neural network

        output = self.fc1(last_hidden_state)

        return output


inputs_size = 10
hidden_size = 20
output_size = 5

# Instantiate  the model

model = GRU(inputs_size,hidden_size,output_size)

# Hyper-parameter

batch_size= 32
sequence_lengh = 15

input_tensor = torch.rand((batch_size,sequence_lengh,inputs_size))

# Forward Pass

output_tensor = model(input_tensor)


# Print the model and model input and output

print(model)
print('Input Shape',input_tensor.shape)
print('Output Shape', output_tensor.shape)

