import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import os
import matplotlib
matplotlib.use('Agg') # Mode Headless pour Linux Mint

torch.manual_seed(42)
nom_fichier_entree = "mon_dessin.png"
nom_fichier_sortie = "mon_dessin_reconstruit.png"

image_pil = Image.open(nom_fichier_entree).convert('L').resize((28, 28))
image_tensor = transforms.ToTensor()(image_pil)
VRAI_A = 1.0 - image_tensor.squeeze()

X = torch.rand((28, 800))
Y = torch.matmul(VRAI_A, X)

A_estime = torch.rand((28, 28), requires_grad=True)
taux_apprentissage = 0.005

for etape in range(1200):
    loss = 0.5 * torch.sum((torch.matmul(A_estime, X) - Y) ** 2)
    loss.backward()
    with torch.no_grad():
        A_estime -= taux_apprentissage * A_estime.grad
    A_estime.grad.zero_()

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(VRAI_A.detach().numpy(), cmap='gray')
ax2.imshow(A_estime.detach().numpy(), cmap='gray')
plt.savefig(nom_fichier_sortie, dpi=150)
plt.close()
print("Fichier sauvegardé avec succès sous Linux Mint.")
