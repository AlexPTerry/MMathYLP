import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

# Basic Block with optional noise
class BasicBlock(nn.Module):
    def __init__(self, in_features, out_features, noise_scale=0.0):
        super(BasicBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.noise_scale = noise_scale
    
    def set_noise_scale(self, noise_scale):
        self.noise_scale = noise_scale

    def forward(self, x):
        out = self.linear(x)
        noise_inner = (torch.rand_like(out)*2 - 1) * self.noise_scale
        out = out + noise_inner
        out = nn.functional.relu(out)
        noise_outer = (torch.rand_like(out)*2 - 1) * self.noise_scale
        out = out + noise_outer
        return out

# Simple ResNet-like model with fully connected layers
class SimpleResNet(nn.Module):
    def __init__(self, layer_sizes, noise_scale=0.0):
        super(SimpleResNet, self).__init__()
        self.layers = nn.ModuleList()
        self.noise_scale = noise_scale
        for i in range(len(layer_sizes) - 1):
            self.layers.append(BasicBlock(layer_sizes[i], layer_sizes[i+1], noise_scale))
            
    def set_noise_scale(self, noise_scale):
        for layer in self.layers:
            layer.set_noise_scale(noise_scale)

    def forward(self, x):
        for layer in self.layers:
            identity = x
            out = layer(x)
            if identity.shape == out.shape:
                out += identity  # Residual connection
            x = out
        return x
    
layer_sizes = [784, 128, 128, 10]  # Example layer sizes for a model
batch_size = 512
input_size = 784  # Input size for each data point
num_classes = 10  # Number of classes for classification
num_epochs = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(device)

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

def load_dataset_to_gpu(dataset):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
    data, labels = next(iter(data_loader))
    return data.to(device), labels.to(device)

# Load the entire training and test datasets into GPU memory

train_set = torchvision.datasets.MNIST(root='./data', train=True,
                                       download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='./data', train=False,
                                      download=True, transform=transform)
train_data, train_labels = load_dataset_to_gpu(train_set)
test_data, test_labels = load_dataset_to_gpu(test_set)

# Create data loaders for batch processing
train_tensor_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
train_loader = torch.utils.data.DataLoader(train_tensor_dataset, batch_size=batch_size, shuffle=True)

test_tensor_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
test_loader = torch.utils.data.DataLoader(test_tensor_dataset, batch_size=1, shuffle=True)


# train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
#                                            shuffle=True)

# test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
#                                           shuffle=False)

# Initialize the model
input_size = 28 * 28  # Each image is 28x28 pixels
model = SimpleResNet([input_size, 128, 128, num_classes], noise_scale=0).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
print("Training started")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs = inputs.view(-1, 28*28)  # Reshape the input for fully connected layers
        labels = labels

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}')

# Evaluation
model.eval()
correct = 0
total = 0

num_runs = 25
exp_losses = np.zeros((num_runs,))
overall_losses = np.zeros((num_runs,))
noise_overall_losses = np.zeros((num_runs,))

with torch.no_grad():
    for j in range(num_runs):
        noise_levels = [(2e-1, 0.8), (2e-2, 0.1), (2e-4, 0.05), (2e-8, 0.05)]
        # noise_levels = [(10, 1),]
        n = len(test_loader)
        i = 0
        level = 0
        count = 0
        noise = noise_levels[level][0]
        partition = noise_levels[level][1]
        running_loss = 0
        overall_running_loss = 0
        noise_overall_running_loss = 0
        old_running_loss=0
        exp_loss = 0
        for inputs, labels in test_loader:
            if i >= partition*n:
                print((running_loss - old_running_loss) / count, count)
                exp_loss += (running_loss - old_running_loss) / count
                    
                running_loss = 0
                old_running_loss = 0
                count = 0
                level += 1
                old_noise = noise
                noise = noise_levels[level][0]
                partition += noise_levels[level][1]
                
            inputs = inputs.view(-1, 28*28)
            labels = labels
            i += 1
            count += 1
            model.set_noise_scale(noise)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            if level > 0:
                model.set_noise_scale(old_noise)
                outputs = model(inputs)
                old_loss = criterion(outputs, labels)
                old_running_loss += old_loss.item()
            
            model.set_noise_scale(0)
            outputs = model(inputs)
            overall_loss = criterion(outputs, labels)
            overall_running_loss += overall_loss.item()
            
            model.set_noise_scale(2e-1)
            outputs = model(inputs)
            noise_overall_loss = criterion(outputs, labels)
            noise_overall_running_loss += noise_overall_loss.item()
            # print(loss, overall_loss)
            # print(running_loss, overall_running_loss)
            # print()
        print((running_loss - old_running_loss) / count, count)
        exp_loss += (running_loss - old_running_loss) / count
        overall_loss = overall_running_loss / n 
        noise_overall_loss = noise_overall_running_loss / n
        
        print(f'Run {j+1}')
        print(f'MLMC Loss: {exp_loss}')
        print(f'Exact Loss: {overall_loss}')
        print(f'Noise Loss: {noise_overall_loss}')
        print()
        
        exp_losses[j] = exp_loss
        overall_losses[j] = overall_loss
        noise_overall_losses[j] = noise_overall_loss
    
print(f'MLMC Losses: {exp_losses}')
print(f'Exact Losses: {overall_losses}')
print(f'Noise Losses: {noise_overall_losses}')

print(f'MLMC Losses Diff: {np.abs(exp_losses - overall_losses)}')
print(f'Noise Losses Diff: {np.abs(noise_overall_losses - overall_losses)}')


print(f'MLMC Losses pDiff: {np.abs(exp_losses - overall_losses)/overall_losses*100}, Mean = {np.mean(np.abs(exp_losses - overall_losses)/overall_losses*100)}')
print(f'Noise Losses pDiff: {np.abs(noise_overall_losses - overall_losses)/overall_losses*100}, Mean = {np.mean(np.abs(noise_overall_losses - overall_losses)/overall_losses*100)}')
print(f'MLMC Wins: {sum(np.abs(exp_losses - overall_losses)/overall_losses < np.abs(noise_overall_losses - overall_losses)/overall_losses)}')
