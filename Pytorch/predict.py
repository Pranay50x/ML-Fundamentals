from main import NeuralNetworkModel
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

test_data = datasets.FashionMNIST(
    root="data",
    download=False, 
    train=False,
    transform=ToTensor()
)

model = NeuralNetworkModel().to(device)

#using locally saved model
model.load_state_dict(torch.load("model.pth"))
model.eval()

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x,y  = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x=x.to(device)
    pred=model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
