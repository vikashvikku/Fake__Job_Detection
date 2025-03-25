import matplotlib.pyplot as plt

# Sample data for growth
years = [2000, 2005, 2010, 2015, 2020, 2024]
portals = [5, 50, 200, 500, 1200, 2500]

plt.figure(figsize=(8, 5))
plt.plot(years, portals, marker='o', color='blue', linewidth=2)
plt.title(' Growth of Online Job Portals Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Job Portals')
plt.grid(True)
plt.tight_layout()
plt.savefig('figure1_growth_of_online_job_portals.png', dpi=300)
plt.show()
