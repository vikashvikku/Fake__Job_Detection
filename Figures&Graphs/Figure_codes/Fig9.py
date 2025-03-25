import matplotlib.pyplot as plt

# Data
models = ["Random Forest", "XGBoost", "Hybrid Stacking"]
accuracy = [80.42, 82.15, 83.35]

# Plot
plt.figure(figsize=(8, 5))
plt.bar(models, accuracy, color=['blue', 'orange', 'green'])

# Labels and title
plt.xlabel("Models", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.title(" Accuracy Comparison of Models", fontsize=14)
plt.ylim(75, 85)  # Setting y-axis limits for better visibility

# Annotating values on bars
for i, v in enumerate(accuracy):
    plt.text(i, v + 0.3, f"{v}%", ha='center', fontsize=12, fontweight='bold')

# Show plot
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
