import torch.optim as optim
import torch.nn as nn
import torch.cuda
import torch

from net import Net
from load_data import load_data
from test import test_model

def main():

    # -------------------
    #      Load Data
    # -------------------
    train_loader, valid_loader, test_loader, classes = load_data()
        
    # -------------------
    #   Define Network
    # -------------------
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    network = Net()
    network.to(device)

    # -------------
    #   Optimiser
    # -------------

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), lr = 0.001, momentum=0.9)

    # -------------
    #   Training
    # -------------

    print("Start training")

    n_epochs = 10
    print_every_n_batch = 200

    for epoch in range(1, n_epochs+1):

        running_loss = 0.0
        i = 0
        
        for data, target in train_loader:

            optimizer.zero_grad()
            inputs, labels = data.to(device), target.to(device)

            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % print_every_n_batch == print_every_n_batch - 1:    # print every n mini-batches
                print(f'[{epoch}, {i + 1:5d}] loss: {running_loss / print_every_n_batch:.3f}')
                running_loss = 0.0
            i += 1

    print('Finished Training')    


    # ---------------
    #   Test model
    # ---------------

    print("Testing the model")
    test_model(network, test_loader, device)

    

if __name__ == "__main__":
    main()