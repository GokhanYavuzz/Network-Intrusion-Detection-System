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
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, 1),
            
            # Output: Probability that the sample is Real (Normal)
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)