import torch
import torch.nn as nn
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
surrogate_mlp = YourMLPClass() # Load your pretrained MLP here
surrogate_mlp.eval() # Surrogate weights must be frozen!

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

# Loss functions
criterion_gan = nn.BCELoss() # Binary Cross Entropy for Real vs Fake

# --- TRAINING STEP (Simplified) ---

# 1. Train Discriminator
# ----------------------
optimizer_D.zero_grad()

# A. Train on Real Normal Data
real_normal_data = get_batch_of_normal_data() 
label_real = torch.ones(batch_size, 1) # Label 1 = Real/Normal
pred_real = discriminator(real_normal_data)
loss_d_real = criterion_gan(pred_real, label_real)

# B. Train on Fake (Adversarial) Data
real_fuzzer_data = get_batch_of_fuzzer_data()
fake_data = generator(real_fuzzer_data) # Generate adversarial samples
label_fake = torch.zeros(batch_size, 1) # Label 0 = Fake/Modified
pred_fake = discriminator(fake_data.detach()) # Detach to stop gradient to Generator
loss_d_fake = criterion_gan(pred_fake, label_fake)

loss_d = loss_d_real + loss_d_fake
loss_d.backward()
optimizer_D.step()

# 2. Train Generator
# ------------------
optimizer_G.zero_grad()

# Generate fresh adversarial samples
adv_samples = generator(real_fuzzer_data)

# A. GAN Loss: Try to fool Discriminator (make it predict 1 for fakes)
pred_discriminator = discriminator(adv_samples)
loss_g_gan = criterion_gan(pred_discriminator, label_real) 

# B. Surrogate Loss: Try to fool the MLP (make it predict Normal)
# IMPORTANT: This assumes your MLP outputs [Prob_Normal, Prob_Attack]
# We want to maximize probability of Normal (Index 0)
surrogate_outputs = surrogate_mlp(adv_samples) 

# If MLP output is Sigmoid (0=Normal, 1=Attack):
# We want output to be close to 0.
loss_g_adv = torch.mean(surrogate_outputs) 

# Combine Losses
total_g_loss = (lambda_gan * loss_g_gan) + (lambda_adv * loss_g_adv)

total_g_loss.backward()
optimizer_G.step()