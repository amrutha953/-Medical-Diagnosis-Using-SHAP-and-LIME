from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ✅ Correct absolute path (adjust to your PC)
data_dir = r"C:\Users\raoka\OneDrive\Desktop\project\data\chest_xray"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_data = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
test_data = datasets.ImageFolder(root=f"{data_dir}/test", transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

print("Train samples:", len(train_data))
print("Test samples:", len(test_data))
