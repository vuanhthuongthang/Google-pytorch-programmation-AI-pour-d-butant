import torch

# 1. GÉNaÉRATION DES DONNÉES PHYSIQUES RÉELLES
VRAI_k1, VRAI_k2 = 5.0, 3.0
X = torch.tensor([[1.0, 2.0, 1.5, 3.0], [2.0, 1.0, 4.0, 2.5]])
Y = torch.tensor([[VRAI_k1 * 1.0, VRAI_k1 * 2.0, VRAI_k1 * 1.5, VRAI_k1 * 3.0],
                  [VRAI_k2 * 2.0, VRAI_k2 * 1.0, VRAI_k2 * 4.0, VRAI_k2 * 2.5]])

# 2. MODÉLISATION DU SYSTÈME PHYSIQUE A
k1 = torch.tensor(1.0, requires_grad=True)
k2 = torch.tensor(1.0, requires_grad=True)
taux_apprentissage = 0.05

for etape in range(100):
    A = torch.diag(torch.stack([k1, k2]))
    Y_virtuel = torch.matmul(A, X)
    
    energie_loss = 0.5 * torch.sum((Y_virtuel - Y) ** 2)
    energie_loss.backward()
    
    with torch.no_grad():
        k1 -= taux_apprentissage * k1.grad
        k2 -= taux_apprentissage * k2.grad
        
    k1.grad.zero_()
    k2.grad.zero_()

print(f"Raideur k1 estimée = {k1.item():.4f}")
print(f"Raideur k2 estimée = {k2.item():.4f}")

