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
dataWithAttack = pd.read_csv("C:\\Users\\Gökhan\\Desktop\\Gökhan\\nids-adversarial\\data\\with_attack_cat_clear_data.csv", low_memory=False)

TARGET = 'attack_cat'

print(f"Using target column: {TARGET} (unique classes: {dataWithAttack[TARGET].nunique()})")
dataWithAttack[TARGET] = dataWithAttack[TARGET].astype(str)  # ensure consistent dtype

total = len(dataWithAttack)
vc = dataWithAttack[TARGET].value_counts(dropna=False)
vc_pct = (vc / total * 100).round(3)
summary = pd.DataFrame({'count': vc, 'percent': vc_pct})
print(f"Total rows: {total}\nUnique classes: {dataWithAttack[TARGET].nunique()}\n")
print(summary)

# GPU varsa kullan, yoksa CPU
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# Hiperparametreler (Orijinal notebook ile uyumlu)
BATCH_SIZE = 64 
LEARNING_RATE = 1e-3 
EPOCHS = 40
L2_REGULARIZATION = 1e-4  # Sklearn'deki 'alpha' parametresi PyTorch'ta weight_decay'dir

def clean_attack_column(dataWithAttack, column_name='attack_cat'):
    replacement_map = {
        'backdoors': 'backdoor',
        'fuzzers': 'fuzzers',  # olası boşluk hatası
        'dos': 'dos',
        'shellcode': 'shellcode',
    }

    dataWithAttack = dataWithAttack.copy()  # orijinali bozmamak için kopya
    dataWithAttack[column_name] = (
        dataWithAttack[column_name]
          .astype('string')
          .str.strip()
          .str.lower()
          .replace(replacement_map)
          .str.capitalize()
    )
    return dataWithAttack

dataWithAttack = clean_attack_column(dataWithAttack, 'attack_cat')

if TARGET in dataWithAttack.columns:
    total = len(dataWithAttack)
    vc = dataWithAttack[TARGET].value_counts(dropna=False)
    vc_pct = (vc / total * 100).round(3)
    summary = pd.DataFrame({'count': vc, 'percent': vc_pct})
    print(f"Total rows: {total}\nUnique classes: {dataWithAttack[TARGET].nunique()}\n")
    print(summary)

# ==========================================
# 2. ÖZELLİK-HEDEF AYIRMA   
# ==========================================
X = dataWithAttack.drop(columns=[TARGET, "Label"], errors="ignore")
y = dataWithAttack[TARGET]

#Label encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)

#Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, 
    test_size=0.2, 
    random_state=42,
    stratify=y_encoded  # ensure balanced split
)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

# TRAIN
dataWithAttack.loc[X_train.index, "attack_cat"].value_counts()
dataWithAttack.loc[X_train.index, "attack_cat"].value_counts(normalize=True).mul(100)

# Test 
dataWithAttack.loc[X_test.index, "attack_cat"].value_counts()
dataWithAttack.loc[X_test.index, "attack_cat"].value_counts(normalize=True).mul(100)


# 2) Kolon tiplerini belirleyelim
# Sayısal/kategorik ayrımı: object ve category -> kategorik; geri kalan -> sayısal varsayımı
def split_columns(X_train, target):
    cols = [c for c in X_train.columns if c != target]
    cat_cols = []
    num_cols = []

    for c in cols:
        if X_train[c].dtype.name in ["object", "category"]:
            cat_cols.append(c)
        else:
            # Çok-unique ve sayısal görünümlü object'ler varsa dönüştürmeyi düşünebilirsiniz.
            num_cols.append(c)

    return num_cols, cat_cols

num_cols, cat_cols = split_columns(X_train, TARGET)
print("Numeric:", len(num_cols), "\nCategorical:", len(cat_cols))

dataWithAttack = dataWithAttack.drop(columns=['Label'])

print("Categorical features:", cat_cols)

# Counting the unique values of the categorical features...

for col_name in X_train.columns:
    if X_train[col_name].dtypes == 'object':
        unique_cat = len(X_train[col_name].unique())
        print("Feature '{col_name}' has {unique_cat} unique values.".format(col_name = col_name, unique_cat = unique_cat))

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

top_proto = pick_top_categories(X_train['proto'], k=6)

X_train = add_binary_flags(X_train, 'proto', top_proto, add_other=True, drop_original=True)
X_test  = add_binary_flags(X_test,  'proto', top_proto, add_other=True, drop_original=True)

top_state = pick_top_categories(X_train['state'], k = 6)

X_train = add_binary_flags(X_train, 'state', top_state, add_other=True, drop_original=True)
X_test  = add_binary_flags(X_test,  'state', top_state, add_other=True, drop_original=True)

top_ct = pick_top_categories(X_train['ct_ftp_cmd'], k = 6)

X_train = add_binary_flags(X_train, 'ct_ftp_cmd', top_ct, add_other=True, drop_original=True)
X_test  = add_binary_flags(X_test,  'ct_ftp_cmd', top_ct, add_other=True, drop_original=True)

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

y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor  = torch.tensor(y_test,  dtype=torch.long)

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