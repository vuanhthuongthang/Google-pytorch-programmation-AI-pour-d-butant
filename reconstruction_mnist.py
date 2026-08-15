import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

torch.manual_seed(42)
dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
image_reelle, label = dataset[0]
VRAI_A = image_reelle.squeeze() 

X = torch.rand((28, 800))
Y = torch.matmul(VRAI_A, X)

A_estime = torch.rand((28, 28), requires_grad=True)
taux_apprentissage = 0.005
iterations = 1200

for etape in range(iterations):
    Y_predit = torch.matmul(A_estime, X)
    loss = 0.5 * torch.sum((Y_predit - Y) ** 2)
    loss.backward()
    
    with torch.no_grad():
        A_estime -= taux_apprentissage * A_estime.grad
    A_estime.grad.zero_()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(VRAI_A.detach().numpy(), cmap='hot')
ax1.set_title("Vrai Milieu Physique A")
ax2.imshow(A_estime.detach().numpy(), cmap='hot')
ax2.set_title("Milieu Reconstruit")
plt.show()
