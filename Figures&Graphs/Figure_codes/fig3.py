import matplotlib.pyplot as plt

# Data for challenges in fake job detection
challenges = ['Data Skewness', 'Vagueness in Job Descriptions', 'Changing Modus Operandi', 'High Volume of Data', 'Limited Labeled Data']
values = [25, 20, 30, 15, 10]

# Plotting
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(challenges, values, color='skyblue', edgecolor='black')

# Styling
ax.set_xlabel('Impact Level (%)')
ax.set_title('Challenges in Fake Job Detection')
ax.invert_yaxis()  # Highest impact on top
for bar in bars:
    width = bar.get_width()
    ax.text(width + 1, bar.get_y() + bar.get_height()/2,
            f'{width}%', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('Fig3.png', dpi=300)
plt.show()
