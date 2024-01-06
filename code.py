# NAME : CHICHARI ANUSHA
# ROLL : 21CH10020

# In[1]:


# Importing all the libraries that are required
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


# In[2]:


# Function to define the neural network architecture
class Net(nn.Module):
    def __init__(self, num_hidden_layers, num_neurons, input_size=784):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Flatten())
        for _ in range(num_hidden_layers):
            self.layers.append(nn.Linear(input_size, num_neurons))
            self.layers.append(nn.ReLU())
            input_size = num_neurons
        self.layers.append(nn.Linear(input_size, 10))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# In[3]:


# Function to load MNIST dataset and create data loaders
def load_mnist_data(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader


# In[4]:


# Function to train a neural network
def train_network(net, trainloader, num_epochs=10, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            inputs = inputs.view(inputs.size(0), -1)  # Flatten the input
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / (i + 1)}")

    print("Finished Training")


# In[5]:


# Function to test a neural network on the test set
def test_network(net, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs = inputs.view(inputs.size(0), -1)  # Flatten the input
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


# In[6]:


# Function to perturb a specific layer of a neural network
def perturb_layer(net, layer_index, noise_std, num_hidden_layers, hidden_size):
    perturbed_net = Net(num_hidden_layers, hidden_size)
    perturbed_net.load_state_dict(net.state_dict())

    # Perturbing the specified layer
    layer = perturbed_net.layers[layer_index * 2]  # Get the layer to perturb
    if isinstance(layer, nn.Linear):
        perturbed_weights = layer.weight + torch.randn_like(layer.weight) * noise_std
        layer.weight = nn.Parameter(perturbed_weights)

    return perturbed_net


# In[7]:


# Function to train networks with different depths for Case 1
def train_case1_networks(num_hidden_layers_list, num_neurons, trainloader, num_epochs, learning_rate):
    networks = []
    for num_hidden_layers in num_hidden_layers_list:
        net = Net(num_hidden_layers, num_neurons)
        train_network(net, trainloader, num_epochs, learning_rate)
        networks.append(net)
    return networks


# In[8]:


# Function to perturb networks and measure performance deviations for Case 2
def perturb_and_measure_deviations(networks, noise_std, testloader):
    deviations = []
    for net in networks:
        num_hidden_layers = sum(1 for layer in net.layers if isinstance(layer, nn.Linear))
        hidden_size = None
        for layer in net.layers:
            if isinstance(layer, nn.Linear):
                hidden_size = layer.out_features
                break
        for layer_index in range(num_hidden_layers):
            if isinstance(net.layers[layer_index * 2], nn.Linear):
                perturbed_net = perturb_layer(net, layer_index, noise_std, num_hidden_layers, hidden_size)
                original_accuracy = test_network(net, testloader)
                perturbed_accuracy = test_network(perturbed_net, testloader)
                deviation = original_accuracy - perturbed_accuracy
                deviations.append((layer_index, deviation))
    
    # Sort the deviations in descending order
    deviations.sort(key=lambda x: x[1], reverse=True)
    return deviations


# In[9]:


# Function to print the results
def print_results(case1_accuracies, case2_deviations):
    print("Case 1: Test Accuracies")
    for i, accuracy in enumerate(case1_accuracies):
        print(f"Network {i + 1}: {accuracy:.2f}%")

    print("\nCase 2: Performance Deviations")
    for network_idx, deviations in enumerate(case2_deviations):
        print(f"Network {network_idx + 1}:")
        for layer_index, deviation in deviations:
            print(f"  Layer {layer_index}: Deviation = {deviation:.2f}%")


# In[10]:


if __name__ == "__main__":
    # Setting random seeds for reproducibility
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Defining hyperparameters
    num_hidden_layers_list = [2, 4, 8]
    num_neurons = 50
    learning_rate = 0.01
    num_epochs = 10

    # Loading MNIST data
    trainloader, testloader = load_mnist_data()

    # Case 1: Training networks
    case1_networks = train_case1_networks(num_hidden_layers_list, num_neurons, trainloader, num_epochs, learning_rate)

    # Case 2: Perturbing networks and measuring deviations
    noise_std = 0.1  # Adding a noise level of 0.1
    case2_deviations = perturb_and_measure_deviations(case1_networks, noise_std, testloader)
    # Ranking deviations in Case 2
    case2_deviations.sort(key=lambda x: x[1], reverse=True)

    # Printing the results
    print("Case 1 Accuracies:", [test_network(net, testloader) for net in case1_networks])
    print("Case 2 Deviations:", case2_deviations)





