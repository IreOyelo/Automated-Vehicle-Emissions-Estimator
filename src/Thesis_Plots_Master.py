import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Qt errors
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output directory for plots
OUT_DIR = "section_4_3_plots"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. SETUP & DATA LOADING
# ---------------------------------------------------------
# Set academic plotting style with Times New Roman
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'], # Forces Times New Roman
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

def load_data(morning_path, evening_path):
    # The prompt meta indicated morning might be semicolon delimited, 
    # so we use a fallback separator if standard comma fails
    df_m = pd.read_csv(morning_path, sep=None, engine='python')
    df_e = pd.read_csv(evening_path, sep=None, engine='python')
    
    # Data cleaning: handle mixed decimal separators (dot and comma)
    for df in [df_m, df_e]:
        for col in df.columns:
            if col not in ['class', 'Traffic State']:
                if df[col].dtype == object:
                    # Attempt to convert to string, replace comma, then to numeric
                    df[col] = df[col].astype(str).str.replace(',', '.')
                    df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df_m['Traffic State'] = 'Morning (Free-Flow)'
    df_e['Traffic State'] = 'Evening (Congested)'
    
    df_combined = pd.concat([df_m, df_e], ignore_index=True)
    
    # Rename 'tier1' to 'baseline' to match new terminology
    if 'tier1_co2_g' in df_combined.columns:
        df_combined.rename(columns={'tier1_co2_g': 'baseline_co2_g'}, inplace=True)
    
    # Map CSV abbreviations to full display names
    df_combined['class'] = df_combined['class'].map(CLASS_MAP).fillna(df_combined['class'])
    
    return df_combined, df_combined[df_combined['Traffic State'] == 'Morning (Free-Flow)'], df_combined[df_combined['Traffic State'] == 'Evening (Congested)']

# Mapping from CSV abbreviations to full display names
CLASS_MAP = {
    'PC':  'Passenger Car',
    'LCV': 'LCV',
    'SUV': 'SUV',
    'MBT': 'Minibus Taxi',
    'HV':  'Heavy Vehicle',
}

# Define class order for consistent plotting
CLASS_ORDER = ['Passenger Car', 'LCV', 'SUV', 'Minibus Taxi', 'Heavy Vehicle']
STATE_ORDER = ['Morning (Free-Flow)', 'Evening (Congested)']

# ---------------------------------------------------------
# 2. PLOTTING FUNCTIONS
# ---------------------------------------------------------

def plot_4_17_aggregate_co2(df):
    """Figure 4.17: Aggregate fleet CO2 inventory (Baseline vs VSP Euro 2)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    agg_df = df.groupby('Traffic State')[['baseline_co2_g', 'e2_co2_g']].sum() / 1000 # Convert to kg
    agg_df = agg_df.reindex(STATE_ORDER)
    
    x = np.arange(len(STATE_ORDER))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, agg_df['baseline_co2_g'], width, label='Distance-Based Baseline', color='#E69F00', edgecolor='black')
    bars2 = ax.bar(x + width/2, agg_df['e2_co2_g'], width, label='VSP-Based (Euro 2)', color='#56B4E9', edgecolor='black')
    
    # Add text labels cleanly above bars
    for bars in [bars1, bars2]:
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + (agg_df.values.max()*0.02), f'{yval:.1f}', 
                    ha='center', va='bottom', fontsize=10, zorder=5)
            
    ax.set_ylabel('Total Fleet CO₂ (kg per 2-hour run)', fontsize=12)
    ax.set_title('Aggregate Fleet CO₂ Inventory by Modelling Approach', fontsize=14, weight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(STATE_ORDER, fontsize=11)
    ax.legend(frameon=False)
    ax.set_ylim(0, agg_df.values.max() * 1.15) # Add 15% headroom for text
    
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/Fig_4_17_Aggregate_CO2.png", dpi=300)
    plt.close()

def plot_4_18_divergence_kde(df):
    """Figure 4.18: KDE distribution of VSP CO2 divergence"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    
    for i, state in enumerate(STATE_ORDER):
        subset = df[df['Traffic State'] == state]
        mean_div = subset['diff_co2_pct'].mean()
        
        sns.kdeplot(data=subset, x='diff_co2_pct', fill=True, ax=axes[i], 
                    color='#0072B2' if i==0 else '#D55E00', alpha=0.3, linewidth=2)
        
        axes[i].axvline(0, color='grey', linestyle='--', label='Zero Divergence', zorder=1)
        axes[i].axvline(mean_div, color='black', linestyle=':', label=f'Mean = {mean_div:.1f}%', zorder=2)
        
        axes[i].set_title(f'({chr(97+i)}) {state}', fontsize=12, weight='bold')
        axes[i].set_xlabel('VSP CO₂ Divergence from Baseline (%)')
        axes[i].set_ylabel('Density')
        axes[i].legend(frameon=False)
    
    axes[0].set_xlim(-100, 150)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/Fig_4_18_Divergence_KDE.png", dpi=300)
    plt.close()

def plot_4_19_mean_co2_by_class(df):
    """Figure 4.19: Mean CO2 per vehicle by class (Baseline vs VSP)"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    
    for i, state in enumerate(STATE_ORDER):
        subset = df[df['Traffic State'] == state]
        agg_df = subset.groupby('class')[['baseline_co2_g', 'e2_co2_g']].mean().reindex(CLASS_ORDER)
        
        x = np.arange(len(CLASS_ORDER))
        width = 0.35
        
        axes[i].bar(x - width/2, agg_df['baseline_co2_g'], width, label='Distance-Based Baseline', color='#E69F00')
        axes[i].bar(x + width/2, agg_df['e2_co2_g'], width, label='VSP-Based (Euro 2)', color='#56B4E9')
        
        axes[i].set_title(f'({chr(97+i)}) {state}', fontsize=12, weight='bold')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(CLASS_ORDER, rotation=25, ha='right')
        axes[i].set_ylabel('Mean CO₂ per vehicle (g)' if i==0 else '')
        if i == 0: axes[i].legend(frameon=False)
        
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/Fig_4_19_Mean_CO2_by_Class.png", dpi=300)
    plt.close()

def plot_4_20_fleet_co2_composition(df):
    """Figure 4.20: Stacked bar chart of fleet CO2 composition by class"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate sum of CO2 per class per traffic state in kg
    agg_df = df.groupby(['Traffic State', 'class'])['e2_co2_g'].sum().unstack() / 1000
    agg_df = agg_df.reindex(index=STATE_ORDER, columns=CLASS_ORDER)
    
    class_colors = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7']
    bottoms = np.zeros(len(STATE_ORDER))
    
    for i, cls in enumerate(CLASS_ORDER):
        values = agg_df[cls].fillna(0).values
        bars = ax.bar(STATE_ORDER, values, bottom=bottoms, label=cls, 
                      color=class_colors[i], edgecolor='white', width=0.55)
        
        for j, bar in enumerate(bars):
            if values[j] > 0:
                total_state_co2 = agg_df.loc[STATE_ORDER[j]].sum()
                pct = (values[j] / total_state_co2) * 100
                if pct > 3.0: # Only annotate segments > 3% to avoid overlap
                    ax.text(bar.get_x() + bar.get_width()/2, bottoms[j] + values[j]/2, 
                            f'{pct:.1f}%', ha='center', va='center', 
                            color='white', weight='bold', fontsize=11)
        bottoms += values
        
    ax.set_ylabel('Total Fleet CO₂ (kg) -- VSP Euro 2', fontsize=12)
    ax.set_title('Fleet CO₂ Composition by Vehicle Class', fontsize=14, weight='bold')
    ax.legend(title='Vehicle Class', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/Fig_4_20_Fleet_CO2_Composition.png", dpi=300)
    plt.close()

def plot_4_21_vsp_kde(df):
    """Figure 4.21: KDE distributions of Vehicle Specific Power"""
    fig, ax = plt.subplots(figsize=(11, 6))
    
    colors = {STATE_ORDER[0]: '#0072B2', STATE_ORDER[1]: '#D55E00'}
    
    for state in STATE_ORDER:
        subset = df[df['Traffic State'] == state]
        mean_vsp = subset['vsp_kw_ton'].mean()
        
        sns.kdeplot(data=subset, x='vsp_kw_ton', fill=True, ax=ax, 
                    color=colors[state], alpha=0.3, linewidth=2, label=state)
        
        ax.axvline(mean_vsp, color=colors[state], linestyle=':', zorder=2)
        ax.text(mean_vsp + 0.5, ax.get_ylim()[1]*0.85, f'Mean: {mean_vsp:.1f} kW/t', 
                color=colors[state], fontsize=11, rotation=90, va='top')
        
    ax.set_xlabel('Vehicle Specific Power (kW/t)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('KDE Distributions of Per-Vehicle VSP', fontsize=14, weight='bold')
    ax.set_xlim(0, 45)
    ax.legend(frameon=False, fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/Fig_4_21_VSP_KDE.png", dpi=300)
    plt.close()

def plot_4_22_co2_boxplots(df):
    """Figure 4.22: Box plots of CO2 distributions by class"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    
    for i, state in enumerate(STATE_ORDER):
        subset = df[df['Traffic State'] == state]
        
        # Plot the variable VSP data as a standard boxplot
        sns.boxplot(data=subset, x='class', y='e2_co2_g', order=CLASS_ORDER, ax=axes[i], 
                    color='#56B4E9', showfliers=True, 
                    flierprops={'marker': '.', 'alpha': 0.3, 'markersize': 4})
        
        # Extract the constant baseline values for each class
        # (Taking the mean works perfectly since the value is identical for every vehicle in that class)
        baseline_vals = subset.groupby('class')['baseline_co2_g'].mean().reindex(CLASS_ORDER)
        
        # Plot the baseline as a prominent 'X' marker aligned with each box
        x_coords = np.arange(len(CLASS_ORDER))
        axes[i].scatter(x_coords, baseline_vals, marker='X', s=150, 
                        color='#D55E00', edgecolor='white', linewidth=1, zorder=5)
        
        axes[i].set_title(f'({chr(97+i)}) {state}', fontsize=13, weight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Per-vehicle CO₂ (g per 100m)' if i==0 else '', fontsize=12)
        axes[i].tick_params(axis='x', rotation=20)
        
        # Custom legend for the first subplot
        if i == 0:
            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#56B4E9', edgecolor='black', label='VSP-Based (Euro 2)'),
                Line2D([0], [0], marker='X', color='w', markerfacecolor='#D55E00', 
                       markeredgecolor='white', markersize=12, label='Distance-Based Baseline')
            ]
            axes[i].legend(handles=legend_elements, frameon=False, fontsize=11, loc='upper left')
            
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/Fig_4_22_CO2_Boxplots.png", dpi=300)
    plt.close()

def plot_4_23_criteria_pollutants(df):
    """Figure 4.23: Fleet-level non-CO2 pollutant inventory (Morning vs Evening) - EURO 2 ONLY"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    pollutants = {'e2_nox_g': 'NOx (g)', 'e2_hc_g': 'HC (g)', 
                  'e2_co_g': 'CO (g)', 'e2_pm25_g': 'PM₂.₅ (g)'}
    
    for i, (col, title) in enumerate(pollutants.items()):
        agg_df = df.groupby('Traffic State')[col].sum().reindex(STATE_ORDER)
        
        bars = axes[i].bar(STATE_ORDER, agg_df.values, color=['#0072B2', '#D55E00'], edgecolor='black', width=0.5)
        axes[i].set_title(f'({chr(97+i)}) Fleet {title} Inventory', fontsize=12, weight='bold')
        axes[i].set_ylabel(title)
        
        # Add labels on top
        for bar in bars:
            yval = bar.get_height()
            offset = agg_df.values.max() * 0.05
            axes[i].text(bar.get_x() + bar.get_width()/2, yval + offset, f'{yval:.1f}', 
                         ha='center', va='bottom', fontsize=10)
            
        axes[i].set_ylim(0, agg_df.values.max() * 1.2) # Headroom
        
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/Fig_4_23_Non_CO2_Pollutants.png", dpi=300)
    plt.close()

def plot_4_24_emission_intensity_heatmap(df):
    """Figure 4.24: Per-vehicle emission intensity heatmaps by class and pollutant"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Configuration for unit conversions
    cols = {
        'e2_co2_g': ('CO₂ (g)', 1),
        'e2_nox_g': ('NOx (mg)', 1000),
        'e2_hc_g': ('HC (mg)', 1000),
        'e2_co_g': ('CO (mg)', 1000),
        'e2_pm25_g': ('PM₂.₅ (µg)', 1000000)
    }
    
    for i, state in enumerate(STATE_ORDER):
        subset = df[df['Traffic State'] == state]
        agg = subset.groupby('class')[list(cols.keys())].mean().reindex(CLASS_ORDER)
        
        # Apply scaling conversions
        for col, (name, mult) in cols.items():
            agg[col] = agg[col] * mult
        
        agg.columns = [cols[c][0] for c in agg.columns]
        
        # Column-normalize for coloring (0 to 1) so heaviest polluter in a category is darkest
        norm_agg = (agg - agg.min()) / (agg.max() - agg.min())
        
        sns.heatmap(norm_agg, annot=agg, fmt=".1f", cmap="YlOrRd", ax=axes[i], 
                    annot_kws={"size": 11, "family": "serif"}, 
                    cbar=(i==1), cbar_kws={'label': 'Normalized Intensity'} if i==1 else None)
                    
        axes[i].set_title(f'({chr(97+i)}) {state}', fontsize=13, weight='bold')
        axes[i].set_ylabel('')
        axes[i].set_xlabel('')
        axes[i].tick_params(axis='y', rotation=0, labelsize=11)
        axes[i].tick_params(axis='x', labelsize=11)
        
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/Fig_4_24_Emission_Intensity_Heatmap.png", dpi=300)
    plt.close()

def plot_4_25_speed_vs_co2_scatter(df):
    """Figure 4.25: Scatter plot of speed vs CO2 output, colored by absolute acceleration"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    for i, state in enumerate(STATE_ORDER):
        subset = df[df['Traffic State'] == state].copy()
        subset['abs_accel'] = subset['accel_avg_mps2'].abs()
        
        # Subsample to 2500 for visual clarity as requested
        if len(subset) > 2500:
            subset = subset.sample(n=2500, random_state=42)
            
        scatter = axes[i].scatter(subset['speed_kmh'], subset['e2_co2_g'], 
                                  c=subset['abs_accel'], cmap='plasma', 
                                  alpha=0.7, s=20, vmin=0, vmax=1.5)
        
        axes[i].set_title(f'({chr(97+i)}) {state}', fontsize=13, weight='bold')
        axes[i].set_xlabel('Mean Vehicle Speed (km/h)', fontsize=12)
        if i == 0:
            axes[i].set_ylabel('CO₂ per vehicle (g per 100m)', fontsize=12)
            
        cbar = fig.colorbar(scatter, ax=axes[i], pad=0.02)
        cbar.set_label('|Acceleration| (m/s²)', fontsize=11)
        
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/Fig_4_25_Speed_CO2_Scatter.png", dpi=300)
    plt.close()

def plot_4_26_mean_divergence_by_class(df):
    """Figure 4.26: Mean VSP CO2 divergence from baseline by class and state"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    agg_df = df.groupby(['class', 'Traffic State'])['diff_co2_pct'].mean().unstack().reindex(CLASS_ORDER)
    
    x = np.arange(len(CLASS_ORDER))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, agg_df[STATE_ORDER[0]], width, label=STATE_ORDER[0], color='#0072B2')
    bars2 = ax.bar(x + width/2, agg_df[STATE_ORDER[1]], width, label=STATE_ORDER[1], color='#D55E00')
    
    ax.axhline(0, color='black', linewidth=1)
    
    # Add text labels: for very negative bars, place inside to avoid clipping the x-axis
    y_min = min(agg_df.min().min(), -10) # Ensure a valid min
    for bars in [bars1, bars2]:
        for bar in bars:
            yval = bar.get_height()
            if yval < y_min * 0.85:
                ax.text(bar.get_x() + bar.get_width()/2, yval + 2.0, f'{yval:.1f}%',
                        ha='center', va='bottom', fontsize=9, color='white', weight='bold')
            else:
                ax.text(bar.get_x() + bar.get_width()/2, yval - 2.0, f'{yval:.1f}%',
                        ha='center', va='top', fontsize=9)
            
    ax.set_ylabel('Mean CO₂ Divergence from Baseline (%)', fontsize=12)
    ax.set_title('Mean VSP CO₂ Divergence from Distance-Based Baseline by Class', fontsize=14, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_ORDER, fontsize=11)
    ax.legend(frameon=False, loc='lower left')
    
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/Fig_4_26_Mean_Divergence_Class.png", dpi=300)
    plt.close()

def plot_4_27_lollipop(df):
    """Figure 4.27: Lollipop chart of Per-vehicle CO2 emission intensity and fleet count"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = {STATE_ORDER[0]: '#0072B2', STATE_ORDER[1]: '#D55E00'}
    
    for i, state in enumerate(STATE_ORDER):
        subset = df[df['Traffic State'] == state]
        counts = subset['class'].value_counts()
        means = subset.groupby('class')['e2_co2_g'].mean()
        
        y_positions = np.arange(len(CLASS_ORDER))
        # Offset slightly so morning and evening lines don't perfectly overlap
        y_offset = y_positions + (0.15 if i==0 else -0.15) 
        
        for j, cls in enumerate(CLASS_ORDER):
            if cls in means:
                mean_val = means[cls]
                count_val = counts[cls]
                
                # Draw stem
                ax.hlines(y=y_offset[j], xmin=0, xmax=mean_val, color=colors[state], alpha=0.5, linewidth=2)
                # Draw bubble (size scaled by count)
                bubble_size = max(50, count_val * 0.15) # Ensure minimum visibility
                ax.scatter(mean_val, y_offset[j], s=bubble_size, color=colors[state], alpha=0.9, 
                           edgecolors='white', zorder=3)
                
                # Dynamic text placement: use whichever is larger — 5% relative or 1.5g absolute
                text_x_offset = mean_val + max(mean_val * 0.05, 1.5)
                ax.text(text_x_offset, y_offset[j], f"{mean_val:.1f}g (n={count_val})",
                        va='center', ha='left', fontsize=9, color=colors[state])

    # Custom legend
    for state in STATE_ORDER:
        ax.scatter([], [], s=200, color=colors[state], label=state)
    ax.legend(frameon=False, loc='upper right')

    ax.set_yticks(np.arange(len(CLASS_ORDER)))
    ax.set_yticklabels(CLASS_ORDER, fontsize=11)
    ax.set_xlabel('Mean CO₂ per vehicle -- VSP Euro 2 (g per 100 m)', fontsize=12)
    ax.set_title('Per-vehicle CO₂ Emission Intensity and Fleet Count by Class', fontsize=14, weight='bold')
    ax.set_xlim(0, df['e2_co2_g'].quantile(0.95) * 1.5) # Auto-scale X axis with padding
    
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/Fig_4_27_Lollipop.png", dpi=300)
    plt.close()

# ---------------------------------------------------------
# 3. EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    # Replace these with your actual CSV file paths
    FILE_MORNING = r"C:\Users\ireoy\OneDrive - University of Cape Town\UCT ACADEMIC YEARS\UCT Masters Second Year\Masters_local_repo\Thesis_Traffic_Emissions_Suite\outputs\study_area_flat_morning_peak\Experiment_HOMOGRAPHY_20260302_0929\emissions_data_morning.csv"
    FILE_EVENING = r"C:\Users\ireoy\OneDrive - University of Cape Town\UCT ACADEMIC YEARS\UCT Masters Second Year\Masters_local_repo\Thesis_Traffic_Emissions_Suite\outputs\2026_02_03 14_59_58 (UTC+02_00)\Experiment_HOMOGRAPHY_20260302_1130\emissions_data_evening.csv"
    
    if os.path.exists(FILE_MORNING) and os.path.exists(FILE_EVENING):
        print("Loading data...")
        df_all, df_m, df_e = load_data(FILE_MORNING, FILE_EVENING)
        
        print("Generating Figure 4.17 (Aggregate CO2)...")
        plot_4_17_aggregate_co2(df_all)
        
        print("Generating Figure 4.18 (Divergence KDE)...")
        plot_4_18_divergence_kde(df_all)
        
        print("Generating Figure 4.19 (Mean CO2 by Class)...")
        plot_4_19_mean_co2_by_class(df_all)
        
        print("Generating Figure 4.20 (Fleet CO2 Composition)...")
        plot_4_20_fleet_co2_composition(df_all)
        
        print("Generating Figure 4.21 (VSP KDE)...")
        plot_4_21_vsp_kde(df_all)
        
        print("Generating Figure 4.22 (CO2 Boxplots)...")
        plot_4_22_co2_boxplots(df_all)
        
        print("Generating Figure 4.23 (Criteria Pollutants)...")
        plot_4_23_criteria_pollutants(df_all)
        
        print("Generating Figure 4.24 (Emission Intensity Heatmap)...")
        plot_4_24_emission_intensity_heatmap(df_all)
        
        print("Generating Figure 4.25 (Speed vs CO2 Scatter)...")
        plot_4_25_speed_vs_co2_scatter(df_all)
        
        print("Generating Figure 4.26 (Mean Divergence by Class)...")
        plot_4_26_mean_divergence_by_class(df_all)
        
        print("Generating Figure 4.27 (Lollipop Intensity)...")
        plot_4_27_lollipop(df_all)
        
        print(f"Done! All 11 plots saved to the '{OUT_DIR}' folder.")
    else:
        print("Error: Could not find CSV files. Please ensure the filenames match and are in the same directory as this script.")