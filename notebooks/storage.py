import json
import os
import joblib
import pandas as pd
import numpy as np
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F

def grad_norm(model: nn.Module):
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        total += float(torch.sum(g * g).item())
    return math.sqrt(total)

def tstats(name, x: torch.Tensor):
    # x GPU’da olabilir; istatistikleri guvenli cekelim
    x_det = x.detach()
    return {
        "name": name,
        "shape": tuple(x_det.shape),
        "min": float(x_det.min().item()),
        "max": float(x_det.max().item()),
        "mean": float(x_det.mean().item()),
        "std": float(x_det.std(unbiased=False).item()),
        "nan": bool(torch.isnan(x_det).any().item()),
        "inf": bool(torch.isinf(x_det).any().item()),
    }

def param_norm(model: nn.Module):
    total = 0.0
    for p in model.parameters():
        w = p.detach()
        total += float(torch.sum(w * w).item())
    return math.sqrt(total)

# =========================
# 0) DEVICE 
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanilan cihaz: {device}")

# =========================
# 1) SURROGATE ARTIFACTS (ONCE OKU)
# =========================
with open("surrogate_cols.json", "r", encoding="utf-8") as f:
    surrogate_cols = json.load(f)

scaler = joblib.load("surrogate_scaler.joblib")

# =========================
# 2) SURROGATE MODEL (MLP import YOK, BOYUTU state_dict'TEN AL)
# =========================
class SurrogateMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        return self.output(x)

model_path = "surrogate_mlp_model.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"{model_path} bulunamadi. Once MLP.py ile modeli kaydetmelisin.")

state = torch.load(model_path, map_location="cpu")
# output.weight shape: (num_classes, 64)
num_classes = state["output.weight"].shape[0]
# layer1.weight shape: (128, input_dim) -> input_dim buradan da cekilebilir
state_input_dim = state["layer1.weight"].shape[1]

INPUT_DIM = len(surrogate_cols)
if INPUT_DIM != state_input_dim:
    raise ValueError(
        f"INPUT_DIM uyusmuyor. surrogate_cols={INPUT_DIM}, model layer1 input={state_input_dim}. "
        f"Preprocessing hizalamasi hatali."
    )

surrogate_mlp = SurrogateMLP(INPUT_DIM, num_classes).to(device)
surrogate_mlp.load_state_dict(state)
surrogate_mlp.eval()
for p in surrogate_mlp.parameters():
    p.requires_grad = False

print(f"Surrogate yuklendi. input_dim={INPUT_DIM}, num_classes={num_classes}")

# =========================
# 3) DATA LOAD + FILTER (Generic/Normal)
# =========================
df = pd.read_csv(
    r"C:\Users\Gökhan\Desktop\Gökhan\nids-adversarial\data\with_attack_cat_clear_data.csv",
    low_memory=False
)

TARGET = "attack_cat"

def clean_attack_column(df_in: pd.DataFrame, column_name="attack_cat") -> pd.DataFrame:
    replacement_map = {
        "backdoors": "backdoor",
        "fuzzers": "fuzzers",
        "dos": "dos",
        "shellcode": "shellcode",
    }
    df_out = df_in.copy()
    df_out[column_name] = (
        df_out[column_name]
        .astype("string")
        .str.strip()
        .str.lower()
        .replace(replacement_map)
        .str.capitalize()
    )
    return df_out

df = clean_attack_column(df, TARGET)

wanted = ["Generic", "Normal"]
df = df[df[TARGET].isin(wanted)].copy()
print(df[TARGET].value_counts(dropna=False))

# =========================
# 4) SPLIT (y string kalsin, stratify icin y_encoded kullan)
# =========================
X = df.drop(columns=[TARGET, "Label"], errors="ignore").copy()
y = df[TARGET].copy()

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("Train label dagilimi:\n", y_train.value_counts(normalize=True))
print("Test  label dagilimi:\n", y_test.value_counts(normalize=True))

# =========================
# 5) MLP ILE AYNI FEATURE SPACE'E DONUSUM
#    - is_proto_*, is_state_*, is_ct_ftp_cmd_* kolonlarini surrogate_cols'ten ceker
#    - o flagleri uretir
#    - sonra surrogate_cols'a reindex eder
# =========================
def normalize_col(s: pd.Series) -> pd.Series:
    return s.fillna("unknown").astype(str).str.lower().str.strip()

def expected_cats_from_surrogate(cols: list, prefix: str) -> list:
    cats = []
    for c in cols:
        if c.startswith(prefix):
            cat = c[len(prefix):]
            if cat != "other":
                cats.append(cat)
    return sorted(set(cats))

def add_flags_exact(df_in: pd.DataFrame, raw_col: str, cats: list) -> pd.DataFrame:
    df_out = df_in.copy()
    if raw_col not in df_out.columns:
        return df_out

    s = normalize_col(df_out[raw_col])
    flags = pd.DataFrame(index=df_out.index)

    for cat in cats:
        flags[f"is_{raw_col}_{cat}"] = (s == cat).astype("uint8")

    flags[f"is_{raw_col}_other"] = (~s.isin(cats)).astype("uint8")

    df_out = df_out.drop(columns=[raw_col])
    df_out = pd.concat([df_out, flags], axis=1)
    return df_out

# Train/Test icin ayni beklenen kategoriler
proto_cats = expected_cats_from_surrogate(surrogate_cols, "is_proto_")
state_cats = expected_cats_from_surrogate(surrogate_cols, "is_state_")
ct_cats    = expected_cats_from_surrogate(surrogate_cols, "is_ct_ftp_cmd_")

X_train_feat = X_train_raw.copy()
X_test_feat  = X_test_raw.copy()

X_train_feat = add_flags_exact(X_train_feat, "proto", proto_cats)
X_test_feat  = add_flags_exact(X_test_feat,  "proto", proto_cats)

X_train_feat = add_flags_exact(X_train_feat, "state", state_cats)
X_test_feat  = add_flags_exact(X_test_feat,  "state", state_cats)

X_train_feat = add_flags_exact(X_train_feat, "ct_ftp_cmd", ct_cats)
X_test_feat  = add_flags_exact(X_test_feat,  "ct_ftp_cmd", ct_cats)

# Son hizalama: kolon sirasi ve eksik kolonlar 0
X_train_feat = X_train_feat.reindex(columns=surrogate_cols, fill_value=0)
X_test_feat  = X_test_feat.reindex(columns=surrogate_cols,  fill_value=0)

# =========================
# 6) SCALE (SADECE surrogate_scaler ile)
# =========================
X_train_scaled = scaler.transform(X_train_feat).astype(np.float32)
X_test_scaled  = scaler.transform(X_test_feat).astype(np.float32)

# =========================
# 7) NORMAL / GENERIC AYRIMI (MASK DEGIL, FEATURE VEKTRORLERI)
# =========================
mask_normal  = (y_train.values == "Normal")
mask_generic = (y_train.values == "Generic")

X_normal_scaled  = X_train_scaled[mask_normal]
X_generic_scaled = X_train_scaled[mask_generic]

print("Normal scaled shape :", X_normal_scaled.shape)
print("Generic scaled shape:", X_generic_scaled.shape)

def np_stats(name, a: np.ndarray):
    a = a.astype(np.float32)
    return {
        "name": name,
        "shape": a.shape,
        "min": float(np.nanmin(a)),
        "max": float(np.nanmax(a)),
        "mean": float(np.nanmean(a)),
        "std": float(np.nanstd(a)),
        "nan": int(np.isnan(a).sum()),
        "inf": int(np.isinf(a).sum()),
    }

print("=== SANITY: scaled data stats ===")
print(np_stats("X_normal_scaled", X_normal_scaled))
print(np_stats("X_generic_scaled", X_generic_scaled))

# StandardScaler dogrulama (train set normalde ~0 mean, ~1 std civari olur; tam 0/1 olmak zorunda degil)
print("=== SANITY: feature-wise mean/std (first 10 features) ===")
print("normal mean first10:", np.mean(X_normal_scaled, axis=0)[:10])
print("normal std  first10:", np.std(X_normal_scaled, axis=0)[:10])


# CPU tensor yap, training loop icinde device'a tasi
real_samples_cpu = torch.tensor(X_normal_scaled, dtype=torch.float32)
generic_samples_cpu = torch.tensor(X_generic_scaled, dtype=torch.float32)

normal_loader = DataLoader(
    TensorDataset(real_samples_cpu),
    batch_size=64,
    shuffle=True,
    drop_last=True
)

print("Hazir: normal_loader ve generic_samples_cpu olustu.")
# Buradan sonra Generator/Discriminator tanimlarina gecebilirsin.


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
EPOCHS = 40
LR = 2e-4
criterion_kl = nn.KLDivLoss(reduction="batchmean")
lambda_cons = 1.0

# Instantiate models
generator = Generator(input_dim=INPUT_DIM).to(device) # Adjust input_dim based on your dataset
discriminator = Discriminator(input_dim=INPUT_DIM).to(device) # Adjust input_dim based on your dataset

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=LR) # Generator optimizer
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LR) # Discriminator optimizer

# Loss functions
criterion_gan = nn.BCELoss() # Binary Cross Entropy for Real vs Fake

# ==========================================
# 4. EĞİTİM DÖNGÜSÜ
# ==========================================
lambda_gan = 0.5   
lambda_adv = 1.0  

# --- DIAGNOSTICS (her 200 batchte bir gibi) ---
LOG_EVERY = 200

print("GAN eğitimi başlıyor...")
for epoch in range(EPOCHS):
    epoch_t0 = time.time()
    # 'get_batch_of_normal_data' yerine gerçek DataLoader döngüsü:
    for i, (real_normal_batch,) in enumerate(normal_loader):
        
        real_normal_batch = real_normal_batch.to(device)

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
        idx = torch.randint(0, len(generic_samples_cpu), (curr_batch_size,))
        real_generic_batch = generic_samples_cpu[idx].to(device)
        
        fake_data = generator(real_generic_batch)
        
        pred_fake = discriminator(fake_data.detach()) 
        loss_d_fake = criterion_gan(pred_fake, label_fake)

        if i == 0:  # her epoch'un ilk batch'i (spam olmasin)
            print(f"[DIAG-D] epoch {epoch+1} | D(real)mean={pred_real.mean().item():.4f} "
                  f"D(fake)mean={pred_fake.mean().item():.4f} | loss_d_real={loss_d_real.item():.4f} loss_d_fake={loss_d_fake.item():.4f}")
            print(" ", tstats("fake_data(detached)", fake_data))


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

        # B) Consistency terimi (surrogate’u “kandırmak” degil, cikisi sabitlemek)
        with torch.no_grad():
            ref_logits = surrogate_mlp(real_generic_batch)        # S(x)
            ref_prob   = F.softmax(ref_logits, dim=1)

        gen_logits = surrogate_mlp(fake_data)                     # S(G(x))
        gen_logprob = F.log_softmax(gen_logits, dim=1)

        loss_cons = criterion_kl(gen_logprob, ref_prob)

        total_g_loss = (lambda_gan * loss_g_gan) + (lambda_cons * loss_cons)
        total_g_loss.backward()
        optimizer_G.step()

        # diagnostics icin tekrar forward yapma, gen_logits’i kullan
        surrogate_out = gen_logits

        if i == 0:  # her epoch'un ilk batch'i
            g_gn = grad_norm(generator)
            d_gn = grad_norm(discriminator)
            print(f"[DIAG-G] epoch {epoch+1} | loss_g={total_g_loss.item():.4f} | grad_norm G={g_gn:.3e} D={d_gn:.3e}")
            # surrogate çıktısı "prob" degil logits; sadece aralik/NaN kontrolu icin bas
            print(" ", tstats("surrogate_logits", surrogate_out))

            with torch.no_grad():
                print(f"[DIAG] epoch {epoch+1} | D(real)={pred_real.mean().item():.4f} "
                    f"D(fake)={pred_fake.mean().item():.4f} "
                    f"fake_min={fake_data.min().item():.3f} fake_max={fake_data.max().item():.3f} "
                    f"fake_mean={fake_data.mean().item():.3f} fake_std={fake_data.std(unbiased=False).item():.3f} "
                    f"NaN={torch.isnan(fake_data).any().item()} Inf={torch.isinf(fake_data).any().item()}")


        if (i % LOG_EVERY == 0):
            with torch.no_grad():
                print("\n[DIAG] epoch", epoch+1, "batch", i)
                print("  D(real) mean:", float(pred_real.mean().item()),
                      "D(fake) mean:", float(pred_fake.mean().item()))
                print("  loss_d:", float(loss_d.item()), "loss_g:", float(total_g_loss.item()))
                print(" ", tstats("fake_data", fake_data))

            # Gradient / param normlari (patlama/sonme yakalamak icin)
            g_gn = grad_norm(generator)
            d_gn = grad_norm(discriminator)
            g_pn = param_norm(generator)
            d_pn = param_norm(discriminator)
            print(f"  grad_norm G:{g_gn:.3e} D:{d_gn:.3e} | param_norm G:{g_pn:.3e} D:{d_pn:.3e}")

            # Sert alarm: NaN/Inf varsa dur
            if torch.isnan(fake_data).any() or torch.isinf(fake_data).any():
                raise RuntimeError("fake_data NaN/Inf uretildi -> egitim durduruldu (instability).")

    if(epoch % 5 == 0):
        print(f"Epoch [{epoch+1}/{EPOCHS}] | D Loss: {loss_d.item():.4f} | G Loss: {total_g_loss.item():.4f} | Surrogate logits mean: {surrogate_out.mean().item():.4f}")

        print(f"[TIME] epoch {epoch+1} took {time.time()-epoch_t0:.2f}s")
        epoch_t0 = time.time()


print("Eğitim bitti. Generator kaydediliyor...")
torch.save(generator.state_dict(), "generator_model.pth")