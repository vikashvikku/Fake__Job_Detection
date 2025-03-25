import matplotlib.pyplot as plt

# Data
labels = ["BERT Embeddings", "Required Experience", "Industry", "Employment Type", "Others"]
sizes = [42, 18, 15, 12, 13]
colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99", "#c2c2f0"]

# Create pie chart
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=140, wedgeprops={'edgecolor': 'black'})

# Title
plt.title(" Feature Importance Distribution (Text vs Categorical)")

# Show plot
plt.show()
