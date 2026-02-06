import json
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from MLP import SurrogateMLP 
import pandas as pd
import numpy as np
import os
import sys
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib

# Hiperparametreler
#INPUT_DIM = 43  # MLP ile aynı (43)
HIDDEN_DIM = 64
BATCH_SIZE = 64
LR = 0.0002
EPOCHS = 50

df = pd.read_csv(r"C:\Users\Gökhan\Desktop\Gökhan\nids-adversarial\data\with_attack_cat_clear_data.csv", low_memory=False)

wanted = ["Generic", "Normal"]
df_filtered = df[df["attack_cat"].isin(wanted)].copy()

# kontrol
print(df_filtered["attack_cat"].value_counts(dropna=False))

# kaydetmek istersen
df_filtered.to_csv(r"C:\Users\Gökhan\Desktop\Gökhan\nids-adversarial\data\dataset_generic_normal.csv", index=False)

df_filtered = pd.read_csv(r"C:\Users\Gökhan\Desktop\Gökhan\nids-adversarial\data\dataset_generic_normal.csv", low_memory=False)

# 1) df_filtered kullanacaksan (attack_cat kolonu df icinde)
X = df_filtered.drop(columns=["attack_cat"]).copy()
y = df_filtered["attack_cat"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("X toplam feature sayisi:", X.shape[1])
print("X kolonlari:", len(X.columns))

print("Train label dagilimi:\n", y_train.value_counts(normalize=True))
print("Test  label dagilimi:\n", y_test.value_counts(normalize=True))


# ==========================================
# 1. AYARLAR
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# MLP.py dosyasını import edebilmek için yol ayarı
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 1) Kategorik (string) kolonlari bul
cat_cols = X_train.select_dtypes(include=["object", "category", "bool","string"]).columns.tolist()
print("Kategorik kolonlar:", cat_cols)

# 2) Kategorikleri sayisala cevir (train kategorilerine gore)
for col in cat_cols:
    train_cats = pd.Categorical(X_train[col]).categories
    X_train[col] = pd.Categorical(X_train[col], categories=train_cats).codes
    X_test[col]  = pd.Categorical(X_test[col],  categories=train_cats).codes

    # NaN veya unseen -> -1, ayri bir kategori id yap
    X_train.loc[X_train[col] == -1, col] = len(train_cats)
    X_test.loc[X_test[col] == -1, col]   = len(train_cats)

left = X_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()
print("Hala string kalanlar:", left)
assert len(left) == 0, f"Scaler'a girmeden once string kolon var: {left}"

# ==========================================

X_train = X_train.reindex(columns=surrogate_cols, fill_value=0)
X_test  = X_test.reindex(columns=surrogate_cols, fill_value=0)

INPUT_DIM = len(surrogate_cols)  # 60 olmali
print("Hizalanmis INPUT_DIM:", INPUT_DIM)

# 3) Standard Scaler (MLP ile ayni)
scaler = joblib.load("surrogate_scaler.joblib")

with open("surrogate_cols.json", "r", encoding="utf-8") as f:
    surrogate_cols = json.load(f)

X_train_scaled = scaler.transform(X_train).astype(np.float32)
X_test_scaled  = scaler.transform(X_test).astype(np.float32)


# --- BOYUT KONTROLÜ ---
REAL_INPUT_DIM = X_train.shape[1]
print(f"Veri Seti Özellik Sayısı: {REAL_INPUT_DIM}")
INPUT_DIM = REAL_INPUT_DIM  # Başlangıçta aynı yap
if REAL_INPUT_DIM != INPUT_DIM:
    print(f"UYARI: Kodun başındaki INPUT_DIM={INPUT_DIM} ama veri setinde {REAL_INPUT_DIM} sütun var.")
    print(f"INPUT_DIM otomatik olarak {REAL_INPUT_DIM} yapılıyor.")
    INPUT_DIM = REAL_INPUT_DIM

# --- NORMAL ve GENERIC AYRIMI ---

print("Saldırı kategorilerine göre ayrılıyor...")
X_normal  = (y_train == "Normal").values
X_generic = (y_train == "Generic").values

print(f"Normal Veri: {len(X_normal)}, Generic Veri: {len(X_generic)}")

# Tensor Dönüşümü
real_samples = torch.tensor(X_normal, dtype=torch.float32).to(device)
generic_samples = torch.tensor(X_generic, dtype=torch.float32).to(device)

# DataLoader (Discriminator'ın gerçek verisi için)
train_loader = DataLoader(TensorDataset(real_samples), batch_size=BATCH_SIZE, shuffle=True) # Sadece gerçek normal veriyi kullanır.

# Generator Neural Network for creating adversarial examples
class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(Generator, self).__init__()
        
        # Encoder: Compresses the input feature space
        self.net = nn.Sequential(
            # Input: Real Generic Sample + Noise (concatenated or added)
            # Here we assume input is just the feature vector of the Generic sample
        
            # 1st Hidden Layer
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(), # Activation Function (ReLu makes negative values zero)
            nn.BatchNorm1d(hidden_dim), # Normalization the layer outputs
        
            # 2nd Hidden Layer    
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
        
            # 3rd Hidden Layer
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            
            # Output Layer: Generates the PERTURBATION vector, not the new row
            nn.Linear(hidden_dim, input_dim), 
            
        )

    def forward(self, real_generic_samples, noise_vector=None):
        """
        Args:
            real_generic_samples: Batch of original generic traffic samples
            noise_vector: Optional noise to ensure diversity
        """
        # If you want to use noise, concatenate it with the input features
        # For simplicity here, we assume the network creates variations based on weights
        
        # 1. Calculate the perturbation (the changes to make)
        perturbation = self.net(real_generic_samples)
        
        # 2. Add perturbation to the original sample
        # We might want to scale the perturbation to avoid destroying the attack
        # epsilon controls how much we are allowed to change the features
        epsilon = 0.2 
        generated_sample = real_generic_samples + (epsilon * perturbation)
        
        # 3. Clamp results to ensure valid feature range (e.g., 0 to 1)
        # Change min/max based on your scaler (0,1 for MinMax; -inf, inf for Standard)
        generated_sample = torch.clamp(generated_sample, 0.0, 1.0)
        
        return generated_sample
    
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(Discriminator, self).__init__()
        
        self.net = nn.Sequential(
            # Input: A feature vector (either Real Normal or Adversarial Generic)
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2), # LeakyReLU to avoid dying neurons, instead of ReLU doesn't zero out negative values
            nn.Dropout(0.3), # Regularization to prevent overfitting
            
            nn.Linear(hidden_dim * 2, hidden_dim), 
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, 1),
            
            nn.Sigmoid() # Output: Probability that the sample is Real (Normal)
        )

    def forward(self, x):
        return self.net(x)
    
# Hyperparameters
lambda_gan = 0.5   # Importance of looking like Normal traffic (Discriminator)
lambda_adv = 1.0   # Importance of fooling the IDS (Surrogate)

# Instantiate models
generator = Generator(input_dim=INPUT_DIM).to(device) # Adjust input_dim based on your dataset
discriminator = Discriminator(input_dim=INPUT_DIM).to(device) # Adjust input_dim based on your dataset
surrogate_mlp = SurrogateMLP(INPUT_DIM) # Load your pretrained MLP here
surrogate_mlp.eval() # Surrogate weights must be frozen!

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=LR) # Generator optimizer
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LR) # Discriminator optimizer

# Loss functions
criterion_gan = nn.BCELoss() # Binary Cross Entropy for Real vs Fake


# --- SURROGATE MODEL YÜKLEME ---
# Buradaki 59 sayısı önemli!
surrogate_mlp = SurrogateMLP(INPUT_DIM).to(device) # Surrogate modelini başlat

model_path = "surrogate_mlp_model.pth" # Ana klasörde arar
if os.path.exists(model_path):
    surrogate_mlp.load_state_dict(torch.load(model_path))
    print("Surrogate model başarıyla yüklendi!")
else:
    print(f"HATA: {model_path} bulunamadı! Lütfen önce MLP.py'yi çalıştırın.")

surrogate_mlp.eval() # Değerlendirme moduna al. Ağırlıklar sabit kalacak.
for param in surrogate_mlp.parameters(): 
    param.requires_grad = False

# ==========================================
# 4. EĞİTİM DÖNGÜSÜ
# ==========================================
lambda_gan = 0.5   
lambda_adv = 1.0  

print("GAN eğitimi başlıyor...")
for epoch in range(EPOCHS):
    # 'get_batch_of_normal_data' yerine gerçek DataLoader döngüsü:
    for i, (real_normal_batch,) in enumerate(train_loader):
        
        # Batch boyutu
        curr_batch_size = real_normal_batch.size(0)
        
        # Etiketler
        label_real = torch.ones(curr_batch_size, 1, device=device)
        label_fake = torch.zeros(curr_batch_size, 1, device=device)

        # ---------------------
        # 1. Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # A. Real Normal Data
        pred_real = discriminator(real_normal_batch)
        loss_d_real = criterion_gan(pred_real, label_real)

        # B. Fake (Adversarial) Data
        # Generic verisinden rastgele örnek al (get_batch_of_generic_data yerine)
        idx = torch.randint(0, len(generic_samples), (curr_batch_size,))
        real_generic_batch = generic_samples[idx]
        
        fake_data = generator(real_generic_batch)
        
        pred_fake = discriminator(fake_data.detach()) 
        loss_d_fake = criterion_gan(pred_fake, label_fake)

        loss_d = (loss_d_real + loss_d_fake) / 2
        loss_d.backward()
        optimizer_D.step()

        # -----------------
        # 2. Train Generator
        # -----------------
        optimizer_G.zero_grad()

        # A. GAN Loss (Discriminator'ı kandır)
        pred_fake_G = discriminator(fake_data) 
        loss_g_gan = criterion_gan(pred_fake_G, label_real) 

        # B. Surrogate Loss (Polisi kandır)
        surrogate_out = surrogate_mlp(fake_data)
        loss_g_adv = torch.mean(surrogate_out) 

        total_g_loss = (lambda_gan * loss_g_gan) + (lambda_adv * loss_g_adv)
        total_g_loss.backward()
        optimizer_G.step()

    if (epoch+1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] | D Loss: {loss_d.item():.4f} | G Loss: {total_g_loss.item():.4f} | Surrogate Prob: {surrogate_out.mean().item():.4f}")

print("Eğitim bitti. Generator kaydediliyor...")
torch.save(generator.state_dict(), "generator_model.pth")