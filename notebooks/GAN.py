import torch
import torch.nn as nn
from MLP import SurrogateMLP
from sklearn.model_selection import train_test_split  
import pandas as pd
import numpy as np
import os
import sys
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# Generator Neural Network for creating adversarial examples
class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(Generator, self).__init__()
        
        # Encoder: Compresses the input feature space
        self.net = nn.Sequential(
            # Input: Real Fuzzer Sample + Noise (concatenated or added)
            # Here we assume input is just the feature vector of the Fuzzer
        
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
            
            # Tanh forces output between -1 and 1. 
            # Useful if your data is MinMaxScaled. 
            # If using StandardScaler, you might remove this or use a clamp later.
            nn.Tanh() 
        )

    def forward(self, real_fuzzer_samples, noise_vector=None):
        """
        Args:
            real_fuzzer_samples: Batch of original malicious traffic (Fuzzers)
            noise_vector: Optional noise to ensure diversity
        """
        # If you want to use noise, concatenate it with the input features
        # For simplicity here, we assume the network creates variations based on weights
        
        # 1. Calculate the perturbation (the changes to make)
        perturbation = self.net(real_fuzzer_samples)
        
        # 2. Add perturbation to the original sample
        # We might want to scale the perturbation to avoid destroying the attack
        # epsilon controls how much we are allowed to change the features
        epsilon = 0.2 
        generated_sample = real_fuzzer_samples + (epsilon * perturbation)
        
        # 3. Clamp results to ensure valid feature range (e.g., 0 to 1)
        # Change min/max based on your scaler (0,1 for MinMax; -inf, inf for Standard)
        generated_sample = torch.clamp(generated_sample, 0.0, 1.0)
        
        return generated_sample
    
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(Discriminator, self).__init__()
        
        self.net = nn.Sequential(
            # Input: A feature vector (either Real Normal or Adversarial Fuzzer)
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
generator = Generator(input_dim=49)
discriminator = Discriminator(input_dim=49)
surrogate_mlp = SurrogateMLP(59) # Load your pretrained MLP here
surrogate_mlp.eval() # Surrogate weights must be frozen!

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

# Loss functions
criterion_gan = nn.BCELoss() # Binary Cross Entropy for Real vs Fake

# ==========================================
# 1. AYARLAR VE İMPORTLAR
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# MLP.py dosyasını import edebilmek için yol ayarı
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from MLP import SurrogateMLP  # Senin istediğin yöntem

# Hiperparametreler
INPUT_DIM = 59  # MLP ile aynı (59)
HIDDEN_DIM = 64
BATCH_SIZE = 64
LR = 0.0002
EPOCHS = 50

# ==========================================
# 2. VERİ YÜKLEME VE HAZIRLIK
# ==========================================
# ==========================================
# 1. DOSYA AYARLARI
# ==========================================
# Elindeki birleşik dosyanın yolu:

print("Dosya okunuyor...")
df = pd.read_csv('C:\\Users\\Gökhan\\Desktop\\Gökhan\\nids-adversarial\\data\\with_attack_cat_clear_data.csv', low_memory=False)

df['attack_cat'] = df['attack_cat'].str.strip()

# ==========================================
# 2. FİLTRELEME (SADECE NORMAL VE FUZZERS)
# ==========================================
print("Veri seti filtreleniyor (Sadece Normal ve Fuzzers)...")

# Saldırı türü sütununun adını kontrol et (Genelde 'attack_cat')
# Eğer farklıysa burayı değiştir.
attack_col = 'attack_cat' 

# Sadece 'Normal' veya 'Fuzzers' içeren satırları seç
# (str.contains kullanarak boşluk veya büyük/küçük harf hatalarını önlüyoruz)
df_filtered = df[df[attack_col].astype(str).str.contains("Normal|Fuzzer", case=False, regex=True)].copy()

print(f"Orijinal Veri Sayısı: {len(df)}")
print(f"Filtrelenmiş Veri Sayısı: {len(df_filtered)}")
print("Kalan Sınıflar:", df_filtered[attack_col].unique())

# ==========================================
# 3. ETİKETLEME (LABEL ENCODING)
# ==========================================
# MLP'nin anlayabilmesi için:
# Normal -> 0
# Fuzzers -> 1 yapmamız lazım.

# Önce mevcut 'label' sütununu (varsa) düşürelim, biz kendimiz en doğrusunu oluşturacağız.
if 'label' in df_filtered.columns:
    df_filtered = df_filtered.drop(columns=['label'])

# Yeni label oluşturma: Normal ise 0, değilse (Fuzzer) 1
df_filtered['label'] = df_filtered[attack_col].apply(lambda x: 0 if 'Normal' in str(x) else 1)

print("\nEtiketler güncellendi: Normal=0, Fuzzer=1")
print(df_filtered[[attack_col, 'label']].value_counts())

# ==========================================
# 4. X ve y AYRIMI
# ==========================================
# Etiket sütunlarını X'ten çıkar
y = df_filtered[[attack_col, 'label']] # Hem ismini hem 0/1 halini saklayalım
X = df_filtered.drop(columns=[attack_col, 'label'])

# ==========================================
# 5. EĞİTİM / TEST BÖLME (%80 - %20)
# ==========================================
print("\nVeri bölünüyor (%80 Train - %20 Test)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.20, 
    random_state=42, 
    stratify=y['label'] # Normal/Fuzzer oranı bozulmasın diye
)

# ==========================================
# 6. KAYDETME
# ==========================================
print("Dosyalar kaydediliyor...")

X_train.to_csv(os.path.join('C:\\Users\\Gökhan\\Desktop\\Gökhan\\nids-adversarial\\data\\mlp_data', "X_train_fuzzer.csv"), index=False)
X_test.to_csv(os.path.join('C:\\Users\\Gökhan\\Desktop\\Gökhan\\nids-adversarial\\data\\mlp_data', "X_test_fuzzer.csv"), index=False)
y_train.to_csv(os.path.join('C:\\Users\\Gökhan\\Desktop\\Gökhan\\nids-adversarial\\data\\mlp_data', "y_train_fuzzer.csv"), index=False)
y_test.to_csv(os.path.join('C:\\Users\\Gökhan\\Desktop\\Gökhan\\nids-adversarial\\data\\mlp_data', "y_test_fuzzer.csv"), index=False)

print("\nİŞLEM TAMAM! 🚀")
print("Artık klasöründe sadece Normal ve Fuzzers içeren temiz X_train, y_train dosyaların var.")
# Ölçekleme (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
df_features = pd.DataFrame(X_train_scaled, columns=X_train.columns)

# --- NORMAL ve FUZZER AYRIMI ---
# Eğer 'attack_cat' sütunu yoksa, label 1'i Fuzzer varsayacağız.
if 'attack_cat' in y_train.columns:
    print("Saldırı kategorilerine göre ayrılıyor...")
    normal_idx = y_train['attack_cat'] == 'Normal'
    fuzzer_idx = y_train['attack_cat'] == 'Fuzzers'


X_normal = df_features[normal_idx].values
X_fuzzer = df_features[fuzzer_idx].values

print(f"Normal Veri: {len(X_normal)}, Fuzzer Veri: {len(X_fuzzer)}")

# Tensor Dönüşümü
real_samples = torch.tensor(X_normal, dtype=torch.float32).to(device)
fuzzer_samples = torch.tensor(X_fuzzer, dtype=torch.float32).to(device)

# DataLoader (Discriminator'ın gerçek verisi için)
# İşte hata veren 'get_batch_of_normal_data' yerine bunu kullanacağız
train_loader = DataLoader(TensorDataset(real_samples), batch_size=BATCH_SIZE, shuffle=True)

# ==========================================
# 3. GAN MODELLERİ
# ==========================================
class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, input_dim) # Tanh yok, çünkü StandardScaler kullanıyoruz
        )

    def forward(self, real_fuzzer_samples):
        perturbation = self.net(real_fuzzer_samples)
        epsilon = 0.1 
        generated_sample = real_fuzzer_samples + (epsilon * perturbation)
        return generated_sample

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2), 
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim), 
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.net(x)

# Modelleri Başlat
generator = Generator(input_dim=INPUT_DIM).to(device)
discriminator = Discriminator(input_dim=INPUT_DIM).to(device)

# --- SURROGATE MODEL YÜKLEME ---
# Buradaki 59 sayısı önemli!
surrogate_mlp = SurrogateMLP(INPUT_DIM).to(device) 

model_path = "surrogate_mlp_model.pth" # Ana klasörde arar
if os.path.exists(model_path):
    surrogate_mlp.load_state_dict(torch.load(model_path))
    print("Surrogate model başarıyla yüklendi!")
else:
    print(f"HATA: {model_path} bulunamadı! Lütfen önce MLP.py'yi çalıştırın.")

surrogate_mlp.eval() # Polisin ağırlıklarını dondur
for param in surrogate_mlp.parameters():
    param.requires_grad = False

# Optimizers & Loss
optimizer_G = optim.Adam(generator.parameters(), lr=LR)
optimizer_D = optim.Adam(discriminator.parameters(), lr=LR)
criterion_gan = nn.BCELoss()

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
        # Fuzzer verisinden rastgele örnek al (get_batch_of_fuzzer_data yerine)
        idx = torch.randint(0, len(fuzzer_samples), (curr_batch_size,))
        real_fuzzer_batch = fuzzer_samples[idx]
        
        fake_data = generator(real_fuzzer_batch)
        
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