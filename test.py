import torch

def test_model(network, test_loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labls in test_loader:
            images, labels = imgs.to(device), labls.to(device)
            outputs = network(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

