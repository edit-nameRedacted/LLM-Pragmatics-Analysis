from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer

ROOT = Path(__file__).resolve().parent.parent.parent

# 1. Load data
df = pd.read_csv(ROOT / 'data/human/data_compiled_sim.csv')

# Filter for DeepSeek, Implicature Condition
df_sub = df[df['condition'] == 'implicature_information'].dropna(
    subset=['DS_beam_mean_cosine_sim', 'rater_scores']
).copy()

# Define X (Entropy) and Y (Human Rating)
X = df_sub[['DS_beam_mean_cosine_sim']]
y = df_sub[['rater_scores']]

print("\n" + "="*55)
print(" PART 1: MUTUAL INFORMATION I(X;Y)")
print("="*55)

# Discretize continuous values into equal-frequency bins (like infotheo in R)
discretizer = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
x_disc = discretizer.fit_transform(X).flatten()
y_disc = discretizer.fit_transform(y).flatten()

# Calculate MI (in nats)
mi_value = mutual_info_score(x_disc, y_disc)
nmi_value = normalized_mutual_info_score(x_disc, y_disc)

print(f"Beam Entropy Mutual Information: {mi_value:.4f} nats")
print(f"Normalized MI: {nmi_value:.4f}")


print("\n" + "="*55)
print(" PART 2: INFORMATION BOTTLENECK (COMPRESSION X -> T)")
print("="*55)

# We compress the Entropy (X) into K=3 distinct states (T) using k-means.
# This replicates the Deterministic IB clustering by finding the optimal 
# boundaries that minimize intra-state variance.
num_clusters = 3
ib_compressor = KBinsDiscretizer(n_bins=num_clusters, encode='ordinal', strategy='kmeans')

# Apply the bottleneck compression
df_sub['IB_State'] = ib_compressor.fit_transform(X).flatten()

# Map the 3 "Beam Uncertainty States" back to the data
summary_ib = df_sub.groupby('IB_State').agg(
    Count=('DS_beam_mean_cosine_sim', 'count'),
    Min_Beam_Ent=('DS_beam_mean_cosine_sim', 'min'),
    Max_Beam_Ent=('DS_beam_mean_cosine_sim', 'max'),
    Mean_Rater_Score=('rater_scores', 'mean')
).reset_index()

print(summary_ib.to_string(index=False))