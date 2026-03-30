import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
# Path to your dataset's labels folder
DATASET_LABELS_PATH = r"C:\Users\ireoy\OneDrive - University of Cape Town\UCT ACADEMIC YEARS\UCT Masters Second Year\Masters_local_repo\Image_Data\Img_Lbls_B1\auto_labels_roi_intersection_1.1" 

# MAPPING MATCHING YOUR classes.txt EXACTLY
# 0=PC, 1=LCV, 2=MBT, 3=HV, 4=SUV
CLASS_MAP = {
    0: 'Passenger Car',
    1: 'LCV / Bakkie',  
    2: 'Minibus Taxi',
    3: 'Heavy Vehicle',
    4: 'SUV'
}

# ACADEMIC COLOR PALETTE (Safe for printing)
COLORS = ["#2E4057", "#1E6B7A", "#2E8B57", "#8FBC8F", "#DAA520"]

# --- THESIS FORMATTING SETUP ---
# Set global font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
# -------------------------------

def analyze_dataset(labels_path, class_map):
    if not os.path.exists(labels_path):
        print(f"Error: Path '{labels_path}' not found.")
        return {}, 0, 0

    print(f"Scanning labels in: {labels_path}...")
    label_files = glob.glob(os.path.join(labels_path, "*.txt"))
    
    total_frames = len(label_files)
    class_counts = {name: 0 for name in class_map.values()}
    total_instances = 0

    for file_path in label_files:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    parts = line.strip().split()
                    if not parts: continue
                    class_id = int(parts[0])
                    if class_id in class_map:
                        class_counts[class_map[class_id]] += 1
                        total_instances += 1
                except: pass 

    return class_counts, total_instances, total_frames

def generate_thesis_plots(class_counts, total_instances, output_dir="output_report"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Prepare Data
    data = []
    for name, count in class_counts.items():
        pct = (count / total_instances * 100) if total_instances > 0 else 0
        data.append({"Class": name, "Count": count, "Percentage": pct})
    
    df = pd.DataFrame(data)
    # Sort for the Bar Chart (Highest Count on Top)
    df_sorted = df.sort_values(by="Count", ascending=False).reset_index(drop=True)

    # Save CSV for Table 3.4
    df_sorted["Percentage String"] = df_sorted["Percentage"].map('{:.1f}%'.format)
    csv_path = os.path.join(output_dir, "table_3_4_composition.csv")
    df_sorted[["Class", "Count", "Percentage String"]].to_csv(csv_path, index=False)
    print(f"Table saved to: {csv_path}")

    # --- FIGURE 1: HORIZONTAL BAR CHART (SORTED) ---
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Create Plot (Hue fixed to avoid warning)
    ax = sns.barplot(
        data=df_sorted,
        x="Count",
        y="Class",
        hue="Class",
        palette="viridis",
        legend=False
    )

    # Add Clean Annotations
    for i, row in df_sorted.iterrows():
        label = f"{int(row['Count']):,} ({row['Percentage']:.1f}%)"
        # Place text slightly inside the bar end for cleaner look
        if row['Percentage'] > 10:
            ax.text(row['Count'] - (total_instances * 0.02), i, label, va='center', ha='right', color='white', fontweight='bold')
        else:
            ax.text(row['Count'] + (total_instances * 0.01), i, label, va='center', ha='left', color='black', fontweight='bold')

    plt.title(f"Dataset Composition by Vehicle Class (N={total_instances:,})", pad=15, fontweight='bold')
    plt.xlabel("Number of Annotated Instances")
    plt.ylabel("")
    plt.tight_layout()
    
    bar_path = os.path.join(output_dir, "figure_bar_distribution.png")
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    print(f"Bar Chart saved to: {bar_path}")

    # --- FIGURE 2: PIE CHART ---
    plt.figure(figsize=(8, 8))
    
    # Explode the smallest slices slightly
    explode = [0.05 if pct < 10 else 0 for pct in df_sorted['Percentage']]
    
    wedges, texts, autotexts = plt.pie(
        df_sorted['Count'],
        labels=df_sorted['Class'],
        autopct='%1.1f%%',
        startangle=140,
        colors=sns.color_palette("viridis", len(df_sorted)),
        explode=explode,
        pctdistance=0.85,
        wedgeprops=dict(width=0.5, edgecolor='w'),
        textprops={'family': 'Times New Roman'} # Explicitly ensure pie labels match
    )
    
    # Style the text
    plt.setp(texts, size=11, weight="bold")
    plt.setp(autotexts, size=10, weight="bold", color="white")
    
    plt.title(f"Proportional Distribution of Classes", fontweight='bold')
    plt.tight_layout()
    
    pie_path = os.path.join(output_dir, "figure_pie_distribution.png")
    plt.savefig(pie_path, dpi=300, bbox_inches='tight')
    print(f"Pie Chart saved to: {pie_path}")
    print("\n" + "="*50)

if __name__ == "__main__":
    counts, tot_inst, tot_frames = analyze_dataset(DATASET_LABELS_PATH, CLASS_MAP)
    if tot_inst > 0:
        generate_thesis_plots(counts, tot_inst)
    else:
        print("No instances found. Check path.")