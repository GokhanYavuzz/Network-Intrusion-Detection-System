import json
import os
import joblib
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn # PyTorch neural network modülü
import torch.optim as optim # PyTorch optimizers
from torch.utils.data import DataLoader, TensorDataset # Veri yükleyici ve dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder  
# ==========================================
# 1. VERİ YÜKLEME VE HAZIRLIK
# ==========================================

print("Dosya okunuyor...")
df = pd.read_csv('C:\\Users\\Gökhan\\Desktop\\Gökhan\\nids-adversarial\\data\\forSurrogatemodel.csv', low_memory=False)

df['attack_cat'] = df['attack_cat'].str.strip() # Boşlukları temizle


# ==========================================
# 1. AYARLAR VE CİHAZ SEÇİMİ
# ==========================================
attack_col = 'attack_cat'

# GPU varsa kullan, yoksa CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# Hiperparametreler (Orijinal notebook ile uyumlu)
BATCH_SIZE = 64 
LEARNING_RATE = 1e-3 
EPOCHS = 10
L2_REGULARIZATION = 1e-4  # Sklearn'deki 'alpha' parametresi PyTorch'ta weight_decay'dir

# ==========================================
# 2. X ve y AYRIMI
# ==========================================
# Etiket sütunlarını X'ten çıkar

X = df.drop(columns=[attack_col, "Label"], errors="ignore")
y = df[attack_col]




def normalize_col(s: pd.Series):
    return s.fillna("unknown").astype(str).str.lower().str.strip()

def pick_top_categories(s: pd.Series, k=6):
    return normalize_col(s).value_counts().head(k).index.tolist()

def add_binary_flags(df: pd.DataFrame, col: str, keep: list, add_other=True, drop_original=True):
    s = normalize_col(df[col])
    for cat in keep:
        new_col = f"is_{col}_{cat}"
        df[new_col] = (s == cat).astype("uint8")
    if add_other:
        df[f"is_{col}_other"] = (~s.isin(keep)).astype("uint8")
    if drop_original:
        df.drop(columns=[col], inplace=True)
    return df

df["attack_cat"] = df["attack_cat"].replace({"Backdoors":"Backdoor"})


# ==========================================
# 5. EĞİTİM / TEST BÖLME (%80 - %20)
# ==========================================
print("\nVeri bölünüyor (%80 Train - %20 Test)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.20, 
    random_state=42, 
    stratify=y
)

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train.astype(str))
y_test_enc  = le.transform(y_test.astype(str))

y_train_tensor = torch.tensor(y_train_enc, dtype=torch.long).to(device)
y_test_tensor  = torch.tensor(y_test_enc,  dtype=torch.long).to(device)


print(f"Eğitim seti boyutu: {X_train.shape[0]} örnek")
print(f"Test seti boyutu: {X_test.shape[0]} örnek")

# --- top-k sadece TRAIN'den; sonra ikisine de uygula ---
for col in ["proto", "state", "ct_ftp_cmd"]:
    if col in X_train.columns:
        top = pick_top_categories(X_train[col], k=6)
        X_train = add_binary_flags(X_train, col, top, add_other=True, drop_original=True)
        X_test  = add_binary_flags(X_test,  col, top, add_other=True, drop_original=True)

obj_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
if obj_cols:
    X_train = pd.get_dummies(X_train, columns=obj_cols, drop_first=False)
    X_test  = pd.get_dummies(X_test,  columns=obj_cols, drop_first=False)
    X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)


num_classes = len(le.classes_)

# ==========================================
# 3. VERİ ÖN İŞLEME (StandardScaler & Tensor Dönüşümü)
# ==========================================
print("Veriler ölçeklendiriliyor ve Tensor'a dönüştürülüyor...")

# Özellikleri ölçekleme (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Eğitim verisini standartlaştır
X_test_scaled = scaler.transform(X_test) # Test verisini aynı scaler ile dönüştür

# Numpy array'leri PyTorch Tensor'larına çevirme
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
X_test_tensor  = torch.tensor(X_test_scaled,  dtype=torch.float32).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

num_classes = len(le.classes_)

print(f"Eğitim verisi Tensor shape: {X_train_tensor.shape}")
print(f"Test verisi Tensor shape: {X_test_tensor.shape}")   

joblib.dump(scaler, "surrogate_scaler.joblib")

with open("surrogate_cols.json", "w", encoding="utf-8") as f:
    json.dump(list(X_train.columns), f)

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
        self.output = nn.Linear(64, num_classes) 


    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.output(x)
        return x
    
if __name__ == "__main__":
    print("MLP.py doğrudan çalıştırıldı, eğitim başlıyor...")

    input_dim = X_train.shape[1]
    model = SurrogateMLP(input_dim).to(device)
    print(f"Model oluşturuldu: {model}")

    # Loss ve Optimizer
    criterion = nn.CrossEntropyLoss() # Çok sınıflı sınıflandırma için uygun kayıp fonksiyonu
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
            loss = criterion(predictions, y_batch)
        
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
    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor)                   # (N, 11)
        y_pred = torch.argmax(logits, dim=1).cpu().numpy()
        y_true = y_test_tensor.cpu().numpy()

    labels = np.arange(num_classes)
    print(classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=list(le.classes_),
        digits=4,
        zero_division=0
    ))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (attack_cat)')
    plt.show()

    # ==========================================
    # 7. MODELİ KAYDETME
    # ==========================================
    # GAN eğitimi sırasında tekrar yüklemek için:
    torch.save(model.state_dict(), "surrogate_mlp_model.pth")
    print("Model 'surrogate_mlp_model.pth' olarak kaydedildi.")

    if __name__ == "__main__":
        print("MLP eğitimi tamamlandı.")