import numpy as np

# Generate mock 3D localization data
np.random.seed(42)  # For reproducibility
X1 = np.random.uniform(0, 1000, 10000)
X2 = np.random.uniform(0, 1000, 10000)
X3 = np.random.uniform(0, 300, 10000)

# Create a random quantile range
pp, pp2, pp3 = 0.2, 0.3, 0.5
addon = 250

# Pick random quantile bins
cut1 = np.quantile(X1, [0.2, 0.4])
cut2 = np.quantile(X2, [0.3, 0.6])
cut3 = np.quantile(X3, [0.1, 0.6])

# Method 1: using np.where and len()
IND = np.where((X1 >= (cut1[0] - addon)) & (X1 <= (cut1[1] + addon)) & 
               (X2 >= (cut2[0] - addon)) & (X2 <= (cut2[1] + addon)) & 
               (X3 >= (cut3[0] - addon)) & (X3 <= (cut3[1] + addon)))[0]
count_len = len(IND)

# Method 2: using boolean mask and np.sum()
mask = ((X1 >= (cut1[0] - addon)) & (X1 <= (cut1[1] + addon)) & 
        (X2 >= (cut2[0] - addon)) & (X2 <= (cut2[1] + addon)) & 
        (X3 >= (cut3[0] - addon)) & (X3 <= (cut3[1] + addon)))
count_sum = np.sum(mask)

count_len, count_sum, count_len == count_sum