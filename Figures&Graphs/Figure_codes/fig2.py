import matplotlib.pyplot as plt

# Job fraud categories
fraud_types = ['Phishing Scams', 'Fake Recruitment Agencies', 'Fake Job Ads', 'Advance Fee Scams', 'Identity Theft']
percentages = [30, 25, 20, 15, 10]

plt.figure(figsize=(7, 7))
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0']
plt.pie(percentages, labels=fraud_types, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Common Types of Job Frauds')
plt.tight_layout()
plt.savefig('figure2_common_job_frauds.png', dpi=300)
plt.show()
