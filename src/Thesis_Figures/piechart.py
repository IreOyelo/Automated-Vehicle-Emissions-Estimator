import matplotlib
matplotlib.use('Agg')  # FIX: Prevents the "Qt platform" crash (Headless mode)
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Data from the table
labels = ['Passenger Car', 'SUV', 'LCV', 'Minibus Taxi', 'Heavy Vehicle']
sizes = [47.4, 21.2, 20.3, 4.8, 6.3]

# Set font to Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12

# Use a colorblind-friendly palette
# "colorblind" is a qualitative palette from seaborn specifically for this purpose
colors = sns.color_palette("colorblind", len(labels))

# Create the plot
fig, ax = plt.subplots(figsize=(8, 8))

# Create a donut chart (pie chart with a hole) to match the example image style
# 'pctdistance' moves the percentage text closer to the center or edge
# 'labeldistance' moves the category labels
wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                  startangle=140, colors=colors, pctdistance=0.85,
                                  wedgeprops=dict(width=0.4, edgecolor='w'))

# Customize text properties for better visibility (optional but good for polish)
for text in texts:
    text.set_color('black')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_weight('bold')

# Add a circle at the center to ensure it looks like a donut (though wedgeprops width handles most of it)
centre_circle = plt.Circle((0,0),0.60,fc='white')
fig.gca().add_artist(centre_circle)

# Equal aspect ratio ensures that pie is drawn as a circle
ax.axis('equal')  

# Add Title
plt.title('Proportional Distribution of Classes', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
# plt.show() # Disabled due to headless mode

# Save the figure
save_path = "piechart_distribution.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✅ Pie chart saved to: {os.path.abspath(save_path)}")