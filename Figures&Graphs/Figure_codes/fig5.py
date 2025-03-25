import matplotlib.pyplot as plt
import numpy as np

features = ['Salary Mentioned', 'Company Website', 'Contact Info', 'Job Description', 'Upfront Payment']
real = [90, 95, 90, 95, 5]
fake = [30, 20, 40, 30, 90]

x = np.arange(len(features))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, real, width, label='Real Job')
rects2 = ax.bar(x + width/2, fake, width, label='Fake Job', color='tomato')

ax.set_ylabel('Presence (%)')
ax.set_title('Sample Fake vs. Real Job Postings')
ax.set_xticks(x)
ax.set_xticklabels(features, rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.savefig('Fake_vs_Real_Comparison.png', dpi=300)
plt.show()
