
import random
import numpy as np

def laplace_mechanism(epsilon, sensitivity):
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale)
    return noise

def add_noise_to_counts(epsilon, counts):
    sensitivity = 5  # Assuming sensitivity is 1 for simplicity
    noisy_counts = {}
    
    for category, count in counts.items():
        noise = laplace_mechanism(epsilon, sensitivity)
        noisy_count = count + noise
        noisy_counts[category] = noisy_count
    
    return noisy_counts

epsilon = 1.0  # Your privacy budget

# Original counts of categorical values
original_counts = {'A': 10, 'B': 20, 'C': 15, 'D': 5}
for i in range(10):
    noisy_counts = add_noise_to_counts(epsilon, original_counts)
    print(noisy_counts)
