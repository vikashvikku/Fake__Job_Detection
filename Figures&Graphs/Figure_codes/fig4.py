import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[], frame_on=False)

plt.text(0.5, 0.9, 'Data Collection', ha='center', fontsize=12, bbox=dict(boxstyle="round", facecolor='white'))
plt.text(0.5, 0.75, 'Preprocessing\n(Feature Engg, Tokenization)', ha='center', fontsize=12, bbox=dict(boxstyle="round", facecolor='white'))
plt.text(0.5, 0.6, 'BERT Embeddings', ha='center', fontsize=12, bbox=dict(boxstyle="round", facecolor='white'))
plt.text(0.5, 0.45, 'SMOTE Data Balancing', ha='center', fontsize=12, bbox=dict(boxstyle="round", facecolor='white'))
plt.text(0.5, 0.3, 'Hybrid Model Training\n(Random Forest + XGBoost)', ha='center', fontsize=12, bbox=dict(boxstyle="round", facecolor='white'))
plt.text(0.5, 0.15, 'Performance Evaluation', ha='center', fontsize=12, bbox=dict(boxstyle="round", facecolor='white'))

for y in [0.825, 0.675, 0.525, 0.375, 0.225]:
    plt.arrow(0.5, y, 0, -0.1, head_width=0.02, head_length=0.02, fc='k', ec='k')

plt.axis('off')
plt.tight_layout()
plt.savefig('hybrid_model_flow.png')
plt.show()
