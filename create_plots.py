import matplotlib.pyplot as plt

# Resource usage by module category (from your FPGA table)
lut_usage_by_module = {
    "DVS Memory": 32,                      # 6 conv2d blocks
    "FC Weights": 1.5*11,                       # 3 maxpool blocks
    "LIF Potential": 6 ,                         # 3 LIF units
    "Conv2D Weights": 0.5,                  # 11 fc memory blocks
    "Result Memory": 3                         # input memory
}

# Extract labels and values
labels = list(lut_usage_by_module.keys())
sizes = list(lut_usage_by_module.values())

def autopct_outside(threshold=5):
    def inner_autopct(pct):
        return f'{pct:.1f}%' if pct > threshold else ''
    return inner_autopct

# Plot with small slices having % outside
fig, ax = plt.subplots(figsize=(6, 6))
wedges, texts, autotexts = ax.pie(
    sizes,
    autopct=autopct_outside(threshold=5),
    startangle=140,
    textprops=dict(size=12),
    pctdistance=0.7
)

# Add legend
ax.legend(wedges, labels, title="Module", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=10)
ax.set_title("CLB LUT Usage by Module", fontsize=14)
plt.tight_layout()
plt.show()