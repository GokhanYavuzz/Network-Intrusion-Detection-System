import pandas as pd
import numpy as np
import torch
import torch.nn as nn # PyTorch neural network modülü
import torch.optim as optim # PyTorch optimizers
from torch.utils.data import DataLoader, TensorDataset # Veri yükleyici ve dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. AYARLAR VE CİHAZ SEÇİMİ
# ==========================================
# GPU varsa kullan, yoksa CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# Hiperparametreler (Orijinal notebook ile uyumlu)
BATCH_SIZE = 64 
LEARNING_RATE = 1e-3 
EPOCHS = 50 
L2_REGULARIZATION = 1e-4  # Sklearn'deki 'alpha' parametresi PyTorch'ta weight_decay'dir

print("Veriler yükleniyor...")
# Dosya yollarını kendi sisteminize göre güncelleyebilirsiniz

# Orijinal notebooktaki yollar
X_train = pd.read_csv(r"C:\Users\Gökhan\Desktop\nids-adversarial\data\X_train.csv", low_memory=False)
y_train = pd.read_csv(r"C:\Users\Gökhan\Desktop\nids-adversarial\data\y_train.csv", low_memory=False)
X_test = pd.read_csv(r"C:\Users\Gökhan\Desktop\nids-adversarial\data\X_test.csv", low_memory=False)
y_test = pd.read_csv(r"C:\Users\Gökhan\Desktop\nids-adversarial\data\y_test.csv", low_memory=False)

# ==========================================
# 3. VERİ ÖN İŞLEME (StandardScaler & Tensor Dönüşümü)
# ==========================================
print("Veriler ölçeklendiriliyor ve Tensor'a dönüştürülüyor...")

# Özellikleri ölçekleme (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test) 

# Numpy array'leri PyTorch Tensor'larına çevirme
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device) # (N, 1) boyutu için

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

# DataLoader oluşturma (Batch işlemleri için)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ==========================================
# 4. MODEL MİMARİSİ (PyTorch)
# ==========================================
class SurrogateMLP(nn.Module):
    def __init__(self, input_dim):
        super(SurrogateMLP, self).__init__()
        # Sklearn: hidden_layer_sizes=(128, 64)
        self.layer1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(64, 1) # Binary classification için tek çıktı
        self.sigmoid = nn.Sigmoid()    # Olasılık değeri (0-1 arası)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.sigmoid(self.output(x))
        return x

input_dim = X_train.shape[1]
model = SurrogateMLP(input_dim).to(device)
print(f"Model oluşturuldu: {model}")

# Loss ve Optimizer
criterion = nn.BCELoss() # Binary Cross Entropy
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REGULARIZATION)

# ==========================================
# 5. EĞİTİM DÖNGÜSÜ
# ==========================================
print("Eğitim başlıyor...")
loss_history = []

model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        # 1. Gradiyentleri sıfırla
        optimizer.zero_grad()
        
        # 2. İleri besleme (Forward pass)
        predictions = model(X_batch)
        
        # 3. Hata hesaplama
        loss = criterion(predictions, y_batch)
        
        # 4. Geri yayılım (Backpropagation)
        loss.backward()
        
        # 5. Ağırlıkları güncelle
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    loss_history.append(avg_loss)
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

# Kayıp grafiğini çizme (İsteğe bağlı)
plt.figure(figsize=(10,5))
plt.plot(loss_history, label='Training Loss')
plt.title('Eğitim Sürecinde Kayıp (Loss)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# ==========================================
# 6. DEĞERLENDİRME
# ==========================================
print("Test seti üzerinde değerlendirme yapılıyor...")
model.eval() # Değerlendirme modu (Dropout vs. kapatır)

with torch.no_grad():
    y_pred_prob = model(X_test_tensor)
    # Olasılıkları 0 veya 1'e yuvarla (Threshold 0.5)
    y_pred = (y_pred_prob > 0.5).float().cpu().numpy()
    y_test_np = y_test_tensor.cpu().numpy()

# Raporlama
print("\nClassification Report:")
print(classification_report(y_test_np, y_pred, digits=4))

# Confusion Matrix
cm = confusion_matrix(y_test_np, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (PyTorch MLP)')
plt.show()

# ==========================================
# 7. MODELİ KAYDETME (İsteğe bağlı)
# ==========================================
# GAN eğitimi sırasında tekrar yüklemek için:
torch.save(model.state_dict(), "surrogate_mlp_model.pth")
print("Model 'surrogate_mlp_model.pth' olarak kaydedildi.")