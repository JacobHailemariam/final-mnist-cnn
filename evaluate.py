import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import MNISTNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),  # /255
])

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = MNISTNet().to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

correct, total = 0, 0
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total   += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(f"Test Accuracy: {100 * correct / total:.2f}%")

torch.save({'preds': all_preds, 'labels': all_labels}, 'test_results.pth')
