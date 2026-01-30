import os
import pandas as pd
import numpy as np
import time
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn # PyTorch neural network modülü
import torch.optim as optim # PyTorch optimizers
from torch.utils.data import DataLoader, TensorDataset # Veri yükleyici ve dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 2. VERİ YÜKLEME VE HAZIRLIK
# ==========================================

print("Dosya okunuyor...")
df = pd.read_csv('C:\\Users\\Gökhan\\Desktop\\Gökhan\\nids-adversarial\\data\\with_attack_cat_clear_data.csv', low_memory=False)

df['attack_cat'] = df['attack_cat'].str.strip() # Boşlukları temizle

# ==========================================
# 2. FİLTRELEME (SADECE NORMAL VE FUZZERS)
# ==========================================
print("Veri seti filtreleniyor (Sadece Normal ve Fuzzers)...")

# Saldırı türü sütununun adını kontrol et (Genelde 'attack_cat')
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

print("Kategorik (yazı) sütunlar sayıya çevriliyor...")
# Nesne (object) tipindeki yani yazı olan sütunları bul
cat_cols = X.select_dtypes(include=['object']).columns

if len(cat_cols) > 0:
    print(f"Dönüştürülen sütunlar: {list(cat_cols)}")
    for col in cat_cols:
        le = LabelEncoder()
        # Sütunu string'e çevirip encode ediyoruz (hatayı önlemek için)
        X[col] = le.fit_transform(X[col].astype(str))
else:
    print("Dönüştürülecek metin sütunu bulunamadı (Zaten hepsi sayı).")

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
X_train = pd.read_csv(r"C:\Users\Gökhan\Desktop\Gökhan\nids-adversarial\data\mlp_data\X_train_fuzzer.csv", low_memory=False)
y_train = pd.read_csv(r"C:\Users\Gökhan\Desktop\Gökhan\nids-adversarial\data\mlp_data\y_train_fuzzer.csv", low_memory=False)
X_test = pd.read_csv(r"C:\Users\Gökhan\Desktop\Gökhan\nids-adversarial\data\mlp_data\X_test_fuzzer.csv", low_memory=False)
y_test = pd.read_csv(r"C:\Users\Gökhan\Desktop\Gökhan\nids-adversarial\data\mlp_data\y_test_fuzzer.csv", low_memory=False)

# ==========================================
# 3. VERİ ÖN İŞLEME (StandardScaler & Tensor Dönüşümü)
# ==========================================
print("Veriler ölçeklendiriliyor ve Tensor'a dönüştürülüyor...")

# Özellikleri ölçekleme (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Eğitim verisini standartlaştır
X_test_scaled = scaler.transform(X_test) # Test verisini aynı scaler ile dönüştür

# Numpy array'leri PyTorch Tensor'larına çevirme
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device) # Veriyi Tensor'a çevir ve ondalıklı yapıya çevir
y_train_tensor = torch.tensor(y_train['label'].values, dtype=torch.float32).unsqueeze(1).to(device) # y_train içinden sadece 'label' sütununu al

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device) # Test verisini Tensor'a çevir 
y_test_tensor = torch.tensor(y_test['label'].values, dtype=torch.float32).unsqueeze(1).to(device) # y_test içinden sadece 'label' sütununu al

# DataLoader oluşturma (Batch işlemleri için)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) # Veri karıştırma ve batch'leme. Ram kullanımı için.

# ==========================================
# 4. MODEL MİMARİSİ (PyTorch)
# ==========================================
class SurrogateMLP(nn.Module):
    def __init__(self, input_dim):
        super(SurrogateMLP, self).__init__()
        # Sklearn: hidden_layer_sizes=(128, 64)
        self.layer1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()                                                  #BURADA NEDEN 2 TANE LAYER VAR DAHA FAZLA OLMALI DEĞİL Mİ?
        self.layer2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(64, 1) # Binary classification için tek çıktı
        self.sigmoid = nn.Sigmoid()    # Olasılık değeri (0-1 arası)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.sigmoid(self.output(x))
        return x
if __name__ == "__main__":
    print("MLP.py doğrudan çalıştırıldı, eğitim başlıyor...")

    input_dim = X_train.shape[1]
    model = SurrogateMLP(input_dim).to(device)
    print(f"Model oluşturuldu: {model}")

    # Loss ve Optimizer
    criterion = nn.BCELoss() # Hata yapması durumunda ceza gönderen Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REGULARIZATION) # Adam optimizasyon algoritması, hatayı minimize eder

    # ==========================================
    # 5. EĞİTİM DÖNGÜSÜ
    # ==========================================
    print("Eğitim başlıyor...")
    loss_history = []

    start_time = time.time() # Kronometreyi başlat (Toplam süre için)
    epoch_start_time = time.time() # Her 5'lik blok için ara zamanlayıcı

    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            # 1. Gradiyentleri sıfırla
            optimizer.zero_grad()
        
            # 2. İleri besleme (Forward pass)
            predictions = model(X_batch)
        
            # 3. Hata hesaplama
            loss = criterion(predictions, y_batch.view(-1, 1))
        
            # 4. Geri yayılım (Backpropagation)
            loss.backward()
        
            # 5. Ağırlıkları güncelle
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
    
        if (epoch + 1) % 5 == 0:
            current_time = time.time()
            batch_time = current_time - epoch_start_time # Son 5 epoch ne kadar sürdü?
            total_time = current_time - start_time       # Başlangıçtan beri ne kadar geçti?
        
            print(f"Epoch [{epoch+1}/{EPOCHS}] | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Son 5 Epoch Süresi: {batch_time:.2f} sn | "
                  f"Toplam Süre: {total_time:.2f} sn")
        
            # Ara zamanlayıcıyı sıfırla
            epoch_start_time = current_time

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
    print(classification_report(y_test_np.flatten(), y_pred.flatten(), digits=4))

    # Confusion Matrix
    cm = confusion_matrix(y_test_np.flatten(), y_pred.flatten())
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