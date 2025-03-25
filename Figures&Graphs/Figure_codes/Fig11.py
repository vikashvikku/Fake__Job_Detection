import matplotlib.pyplot as plt
import numpy as np

# Data
models = ["Total Time"]
bert_time = [11]  # BERT Embeddings (in minutes)
rf_time = [4]     # Random Forest Tuning (in minutes)
xgb_time = [14]   # XGBoost Tuning (in minutes)

# Stacked bar chart
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(models, bert_time, label="BERT Embeddings", color="#1f77b4")
ax.bar(models, rf_time, bottom=np.array(bert_time), label="RF Tuning", color="#ff7f0e")
ax.bar(models, xgb_time, bottom=np.array(bert_time) + np.array(rf_time), label="XGBoost Tuning", color="#2ca02c")

# Labels and Title
ax.set_ylabel("Time (minutes)")
ax.set_title(" Time Consumption Breakdown")
ax.legend()

# Show plot
plt.show()
