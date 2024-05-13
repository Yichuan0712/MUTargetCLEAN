import torch
import torch.nn as nn

def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}\n")
    


# Define the CNN layer
class SimpleCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(SimpleCNN, self).__init__()
        self.conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.linear = nn.Linear(out_channels, 1)
        self.relu = nn.ReLU()
        #Initialize weights from a uniform distribution within the specified range
        weight_range = torch.sqrt(torch.tensor(1.0 / in_channels))  # Range for weight initialization
        conv_layer.weight.data.uniform_(-weight_range, weight_range)
        # Initialize biases to zeros
        nn.init.zeros_(conv_layer.bias)
    
    def forward(self, x):
        x = self.conv_layer(x)
        #x = self.relu(x)
        #x, _ = torch.max(x, dim=1, keepdim=True)  # Max pooling across output channels
        x  = self.linear(x.permute(0,2,1)).squeeze(-1)
        return x

# Example usage
# Create an instance of the SimpleCNN class
cnn_layer = SimpleCNN(in_channels=1280, out_channels=1, kernel_size=1, stride=1, padding='same')
#cnn_layer = SimpleCNN(in_channels=1, out_channels=8, kernel_size=[14,1280], stride=1, padding='same') #change to nn.Conv2d
#cnn_layer=nn.Linear(1280, 1)
print_trainable_parameters(cnn_layer)
#Conv1d trainable params: 143368 || all params: 143368 || trainable%: 100.0
#Conv2d trainable params: 143368 || all params: 143368 || trainable%: 100.0
#linear trainable params: 1281 || all params: 1281 || trainable%: 100.0


# Create a random input tensor
input_tensor = torch.randn(10, 1280, 1022)  # Batch size 1, features size, sequence len

# Pass the input tensor through the CNN layer
output_tensor = cnn_layer(input_tensor)

print("Input shape:", input_tensor.shape)
print("Output shape:", output_tensor.shape)



class ParallelCNNDecoders(nn.Module):
    def __init__(self,input_size, output_sizes,out_channels=8, kernel_size=14):
        super(ParallelCNNDecoders, self).__init__()
        self.cnn_decoders = nn.ModuleList([
            SimpleCNN(in_channels=input_size, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding='same') for output_size in output_sizes
        ])
    
    def forward(self, x):
        decoder_outputs = [decoder(x.permute(0, 2, 1)).squeeze(1) for decoder in self.cnn_decoders]
        return decoder_outputs


cnn_layer = ParallelCNNDecoders(1280,[1]*8)
input_tensor = torch.randn(10, 1022,1280)  # Batch size 1, 3 channels, 32x32 image

# Pass the input tensor through the CNN layer
output_tensor = cnn_layer(input_tensor)
output_tensor=torch.stack(output_tensor, dim=1)  #.squeeze(-1) 
print("Input shape:", input_tensor.shape)
print("Output shape:", output_tensor.shape)



input_tensor = torch.randn(3, 2)  # Batch size 1, features size, sequence len
input_tensor_sig=torch.sigmoid(input_tensor)


input_tensor = torch.randn(3, 5)  # Batch size 1, 3 channels, 32x32 image

def relu_max(input):
    # Find the maximum value along each row
    sigmoid_input = torch.sigmoid(input)
    max_values, _ = torch.max(sigmoid_input, dim=-1, keepdim=True)
    
    # Create a mask of the same shape as the tensor where 1 corresponds to the maximum value and 0 otherwise
    mask = torch.eq(sigmoid_input, max_values).to(input.device)
    
    # Set all values to -10000 except for the maximum value in each row
    output = torch.where(mask, input, torch.tensor(-10000.).to(input.device))
    return output

print(input_tensor)
print(relu_max(input_tensor))