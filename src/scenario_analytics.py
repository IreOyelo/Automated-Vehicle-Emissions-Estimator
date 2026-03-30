import matplotlib
matplotlib.use('Agg')  # MUST be before importing pyplot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os

class ScenarioPlotter:
    def __init__(self, evening_csv_path):
        self.csv_path = evening_csv_path
        
        try:
            self.df = pd.read_csv(evening_csv_path, sep=';')
            if len(self.df.columns) == 1:
                self.df = pd.read_csv(evening_csv_path, sep=',')
        except Exception:
            self.df = pd.read_csv(evening_csv_path, sep=',')

        # Data cleaning: handle mixed decimal separators (dot and comma)
        for col in self.df.columns:
            if col not in ['class', 'full_class', 'count_id', 'track_id']:
                if self.df[col].dtype == object:
                    self.df[col] = self.df[col].astype(str).str.replace(',', '.')
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                
        class_map = {
            'PC': 'PassengerCar', 
            'SUV': 'SUV', 
            'LCV': 'LCV', 
            'MBT': 'Minibus Taxi', 
            'HV': 'Heavy Vehicle', 
            'Passenger Car': 'PassengerCar'
        }
        self.df['full_class'] = self.df['class'].map(class_map).fillna(self.df['class'])

        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['mathtext.fontset'] = 'stix'
        sns.set_theme(style="whitegrid", font="Times New Roman")

    def plot_baseline_lollipop(self):
        """Generates Figure 4.34: Per-vehicle CO2 emission intensity and fleet count."""
        class_stats = self.df.groupby('full_class').agg(
            mean_co2=('e2_co2_g', 'mean'),
            count=('track_id', 'count')
        ).reset_index()
        class_stats = class_stats.sort_values('mean_co2')

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hlines(y=class_stats['full_class'], xmin=0, xmax=class_stats['mean_co2'], color='skyblue', linewidth=2)
        ax.scatter(class_stats['mean_co2'], class_stats['full_class'], 
                   s=class_stats['count'] * 0.5, 
                   c=sns.color_palette("Set2", len(class_stats)), 
                   alpha=0.8, edgecolors="w", linewidth=2)

        for i, row in class_stats.iterrows():
            ax.text(row['mean_co2'] + 0.5, row['full_class'], 
                    f"{row['mean_co2']:.2f}g (n={row['count']})", 
                    va='center', ha='left', fontsize=10, fontname='Times New Roman')

        ax.set_xlabel('Mean CO2 per vehicle - Euro 2 (g per 100 m)', fontsize=12, fontname='Times New Roman')
        ax.set_ylabel('Vehicle Class', fontsize=12, fontname='Times New Roman')
        ax.set_title('Per-vehicle CO2 Emission Intensity and Fleet Count by Class', fontsize=14, fontweight='bold', fontname='Times New Roman')
        ax.set_xlim(0, max(class_stats['mean_co2']) * 1.3)
        plt.tight_layout()
        plt.savefig('fig_4_34_lollipop.png', dpi=300)
        plt.close()

    def plot_scenario_a1_nox(self):
        """Generates Figure 4.28: Scenario A1 NOx Reduction (Two Panels)."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Panel A: Fleet NOx by vehicle class
        class_nox = self.df.groupby('full_class')['e2_nox_g'].sum().sort_values(ascending=True)
        total_nox_base = class_nox.sum()
        
        axes[0].barh(class_nox.index, class_nox.values, color=sns.color_palette("colorblind", len(class_nox)))
        axes[0].set_xlabel('Fleet NOx total (g) -- Euro 2 baseline', fontsize=12, fontname='Times New Roman')
        axes[0].set_title('(a) Fleet NOx by vehicle class', fontsize=14, fontweight='bold', fontname='Times New Roman')
        
        for i, (cls, val) in enumerate(class_nox.items()):
            pct = (val / total_nox_base) * 100
            axes[0].text(val + 0.5, i, f"{val:.2f} g ({pct:.1f}%)", va='center', fontsize=10, fontname='Times New Roman')

        # Panel B: Scenario A1 NOx avoided
        mbt_data = self.df[self.df['full_class'] == 'Minibus Taxi']
        mbt_count = len(mbt_data)
        mbt_mean_nox = mbt_data['e2_nox_g'].mean() if not mbt_data.empty else 0
        converted_mbt = int(0.30 * mbt_count)
        avoided_nox = converted_mbt * mbt_mean_nox
        residual_nox = total_nox_base - avoided_nox

        axes[1].bar(['Baseline fleet\nNOx (Euro 2)', 'After 30% MBT\nelectrification'], 
               [total_nox_base, residual_nox], 
               color=['#2b8cbe', '#31a354'], width=0.5)

        axes[1].bar(['After 30% MBT\nelectrification'], [avoided_nox], bottom=[residual_nox], 
               color='#a1d99b', width=0.5, hatch='//')

        axes[1].set_ylabel('Fleet NOx (g)', fontsize=12, fontname='Times New Roman')
        axes[1].set_title('(b) Scenario A1: NOx avoided', fontsize=14, fontweight='bold', fontname='Times New Roman')
        axes[1].set_ylim(0, total_nox_base * 1.15)
        
        axes[1].text(0, total_nox_base + 1, f"{total_nox_base:.2f} g", ha='center', va='bottom', fontname='Times New Roman')
        axes[1].text(1, residual_nox + avoided_nox/2, f"-{avoided_nox:.2f} g", ha='center', va='center', color='black', fontname='Times New Roman')

        plt.tight_layout()
        plt.savefig('fig_4_28_scenario_a1.png', dpi=300)
        plt.close()

    def plot_scenario_a2_co2(self):
        """Generates Figure 4.29: Scenario A2 CO2 Savings (Two Panels)."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        pc_data = self.df[self.df['full_class'] == 'PassengerCar']
        mbt_data = self.df[self.df['full_class'] == 'Minibus Taxi']
        
        pc_count = len(pc_data)
        pc_mean_co2 = pc_data['e2_co2_g'].mean() if not pc_data.empty else 0
        converted_pc = int(0.30 * pc_count)
        avoided_co2_pc = converted_pc * pc_mean_co2

        mbt_count = len(mbt_data)
        mbt_mean_co2 = mbt_data['e2_co2_g'].mean() if not mbt_data.empty else 0
        converted_mbt = int(0.30 * mbt_count)
        avoided_co2_mbt = converted_mbt * mbt_mean_co2

        # Panel A: Total CO2 avoided
        bars = axes[0].bar(['PassengerCar\npathway', 'Minibus Taxi\npathway'], 
                      [avoided_co2_pc, avoided_co2_mbt], 
                      color=['#1f78b4', '#e31a1c'], width=0.5)

        axes[0].set_ylabel('Total CO2 avoided (g per 2-hour run)', fontsize=12, fontname='Times New Roman')
        axes[0].set_title('(a) Total CO2 avoided per electrification pathway', fontsize=14, fontweight='bold', fontname='Times New Roman')
        axes[0].set_ylim(0, max(avoided_co2_pc, avoided_co2_mbt) * 1.2)

        for bar, avoided in zip(bars, [avoided_co2_pc, avoided_co2_mbt]):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + (avoided_co2_pc * 0.02), 
                    f"{avoided:.0f} g", ha='center', va='bottom', fontname='Times New Roman')

        if avoided_co2_mbt > 0:
            ratio = avoided_co2_pc / avoided_co2_mbt
            axes[0].text(0.5, max(avoided_co2_pc, avoided_co2_mbt) * 0.5, f"PC pathway\n{ratio:.1f}x larger", 
                    ha='center', va='center', color='#d95f02', fontname='Times New Roman')

        # Panel B: Per-vehicle intensity
        bars2 = axes[1].bar(['PassengerCar\n(30% converted)', 'Minibus Taxi\n(30% converted)'], 
                      [pc_mean_co2, mbt_mean_co2], 
                      color=['#1f78b4', '#e31a1c'], width=0.5)

        axes[1].set_ylabel('Mean CO2 per vehicle (g per 100 m, Euro 2)', fontsize=12, fontname='Times New Roman')
        axes[1].set_title('(b) Per-vehicle intensity and conversion count', fontsize=14, fontweight='bold', fontname='Times New Roman')
        axes[1].set_ylim(0, max(pc_mean_co2, mbt_mean_co2) * 1.2)

        for bar, intensity, count in zip(bars2, [pc_mean_co2, mbt_mean_co2], [converted_pc, converted_mbt]):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f"{intensity:.2f} g\nper vehicle", ha='center', va='bottom', fontname='Times New Roman')
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() / 2, 
                    f"n = {count}\nconversions", ha='center', va='center', color='white', fontname='Times New Roman')

        plt.tight_layout()
        plt.savefig('fig_4_29_scenario_a2.png', dpi=300)
        plt.close()

    def plot_scenario_b_absolute_totals(self):
        """Generates Figure 4.30: Scenario B Absolute Emission Totals (5 Panels)."""
        pollutants = ['co2', 'nox', 'hc', 'co', 'pm25']
        labels = ['CO2', 'NOx', 'HC', 'CO', 'PM2.5']
        
        fig, axes = plt.subplots(1, 5, figsize=(16, 6))
        colors = ['#2b8cbe', '#2ca25f']

        for i, p in enumerate(pollutants):
            ax = axes[i]
            e2_val = self.df[f'e2_{p}_g'].sum()
            e5_val = self.df[f'e5_{p}_g'].sum()
            
            if p == 'co2':
                e2_val /= 1000
                e5_val /= 1000
                ax.set_ylabel(f'{labels[i]} (kg)', fontsize=12, fontname='Times New Roman')
            else:
                ax.set_ylabel(f'{labels[i]} (g)', fontsize=12, fontname='Times New Roman')
                
            bars = ax.bar(['Euro 2', 'Euro 5'], [e2_val, e5_val], color=colors, width=0.6)
            ax.set_ylim(0, e2_val * 1.15)
            ax.set_title(labels[i], fontsize=14, fontweight='bold', fontname='Times New Roman')
            
            red_pct = ((e2_val - e5_val) / e2_val * 100) if e2_val > 0 else 0
            ax.text(0.5, e2_val * 1.05 if e2_val > 0 else 1, f"-{red_pct:.0f}%", 
                    ha='center', va='bottom', color='#006d2c', fontweight='bold', fontname='Times New Roman')
            
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (e2_val*0.01), 
                        f"{bar.get_height():.1f}", ha='center', va='bottom', fontsize=10, fontname='Times New Roman')

        plt.tight_layout()
        plt.savefig('fig_4_30_scenario_b_abs.png', dpi=300)
        plt.close()

    def plot_scenario_b_reductions(self):
        """Generates Figure 4.31: Scenario B Percentage Emission Reduction."""
        pollutants = ['co2', 'nox', 'hc', 'co', 'pm25']
        labels = ['CO2', 'NOx', 'HC', 'CO', 'PM2.5']
        reductions = []

        for p in pollutants:
            e2_sum = self.df[f'e2_{p}_g'].sum()
            e5_sum = self.df[f'e5_{p}_g'].sum()
            if e2_sum > 0:
                red = ((e2_sum - e5_sum) / e2_sum) * 100
                reductions.append(red)
            else:
                reductions.append(0)
                print(f"Warning: Sum of e2_{p}_g is 0. Reduction for {p.upper()} set to 0%.")

        fig, ax = plt.subplots(figsize=(8, 5))
        y = np.arange(len(labels))
        bars = ax.barh(y, reductions, height=0.6, color=['#e6ab02', '#7570b3', '#1b9e77', '#d95f02', '#1a9850'])

        ax.set_xlabel('Emission Reduction (%)', fontsize=12, fontname='Times New Roman')
        ax.set_title('Scenario B: Percentage emission reduction', fontsize=14, fontweight='bold', fontname='Times New Roman')
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=12, fontname='Times New Roman')
        ax.set_xlim(0, 100)

        ax.invert_yaxis()
        ax.bar_label(bars, fmt='%.1f%%', padding=5, fontname='Times New Roman', fontweight='bold')
        plt.tight_layout()
        plt.savefig('fig_4_31_scenario_b_red.png', dpi=300)
        plt.close()

    def plot_scenario_b_class_disaggregation(self):
        """Generates Figure 4.32: Scenario B CO2 reduction disaggregated by class."""
        class_co2 = self.df.groupby('full_class').agg(
            e2=('e2_co2_g', 'sum'),
            e5=('e5_co2_g', 'sum')
        ).reset_index()
        
        class_co2['avoided'] = class_co2['e2'] - class_co2['e5']
        class_co2 = class_co2.sort_values('e2', ascending=False)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Panel A: Stacked Bar
        x = np.arange(len(class_co2))
        axes[0].bar(x, class_co2['e5'], label='Retained (Euro 5)', color='#2b8cbe', width=0.5)
        axes[0].bar(x, class_co2['avoided'], bottom=class_co2['e5'], label='CO2 avoided', color='#a6bddb', hatch='//', width=0.5)
        
        axes[0].set_ylabel('Fleet CO2 (g per 2-hour run)', fontsize=12, fontname='Times New Roman')
        axes[0].set_title('(a) CO2 avoided by class (Euro 2 to Euro 5)', fontsize=14, fontweight='bold', fontname='Times New Roman')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(class_co2['full_class'], rotation=15, ha='right', fontname='Times New Roman')
        axes[0].legend(prop={'family': 'Times New Roman'})
        
        for i, avoided in enumerate(class_co2['avoided']):
            axes[0].text(i, class_co2['e2'].iloc[i] + 1000, f"{avoided:.0f} g", ha='center', va='bottom', fontweight='bold', fontname='Times New Roman')

        # Panel B: Pie Chart
        axes[1].pie(class_co2['avoided'], labels=class_co2['full_class'], autopct='%1.1f%%', 
                    startangle=90, colors=sns.color_palette("colorblind", len(class_co2)), 
                    textprops={'family': 'Times New Roman'})
        axes[1].set_title('(b) Share of CO2 avoided by class', fontsize=14, fontweight='bold', fontname='Times New Roman')
        
        plt.tight_layout()
        plt.savefig('fig_4_32_scenario_b_class.png', dpi=300)
        plt.close()

    def plot_scenario_b_relative_levels(self):
        """Generates Figure 4.33: Euro 5 fleet emission levels relative to Euro 2 baseline."""
        pollutants = ['co2', 'nox', 'hc', 'co', 'pm25']
        labels = ['CO2', 'NOx', 'HC', 'CO', 'PM2.5']
        e5_pcts = []

        for p in pollutants:
            e2_sum = self.df[f'e2_{p}_g'].sum()
            e5_sum = self.df[f'e5_{p}_g'].sum()
            e5_pct = (e5_sum / e2_sum * 100) if e2_sum > 0 else 100
            e5_pcts.append(e5_pct)

        fig, ax = plt.subplots(figsize=(10, 6))
        y = np.arange(len(labels))
        height = 0.35

        ax.barh(y + height/2, [100]*len(labels), height, label='Euro 2 (baseline = 100%)', color='#3288bd')
        ax.barh(y - height/2, e5_pcts, height, label='Euro 5 (scenario)', color='#1a9850')

        ax.set_xlabel('Emission as percentage of Euro 2 baseline (%)', fontsize=12, fontname='Times New Roman')
        ax.set_title('Scenario B: Euro 5 fleet emission levels relative to Euro 2 baseline', fontsize=14, fontweight='bold', fontname='Times New Roman')
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=12, fontname='Times New Roman')
        ax.set_xlim(0, 115)
        ax.invert_yaxis()
        # Move legend to the bottom center (outside plot) to avoid overlapping with bars
        ax.legend(prop={'family': 'Times New Roman'}, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

        for i, pct in enumerate(e5_pcts):
            ax.text(pct - 1, i - height/2, f"{pct:.1f}%", ha='right', va='center', color='white', fontweight='bold', fontname='Times New Roman')
            ax.text(101, i + height/2, f"-{100-pct:.0f}%", ha='left', va='center', color='#1a9850', fontweight='bold', fontname='Times New Roman')

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2) # Make room for the legend underneath
        plt.savefig('fig_4_33_scenario_b_relative.png', dpi=300)
        plt.close()

    def plot_cross_scenario_comparison(self):
        """Generates Figure 4.35: Cross-Scenario CO2 Comparison."""
        pc_data = self.df[self.df['full_class'] == 'PassengerCar']
        mbt_data = self.df[self.df['full_class'] == 'Minibus Taxi']
        
        pc_mean_co2 = pc_data['e2_co2_g'].mean() if not pc_data.empty else 0
        mbt_mean_co2 = mbt_data['e2_co2_g'].mean() if not mbt_data.empty else 0
        
        avoided_co2_pc = int(0.30 * len(pc_data)) * pc_mean_co2
        avoided_co2_mbt = int(0.30 * len(mbt_data)) * mbt_mean_co2

        total_co2_base = self.df['e2_co2_g'].sum()
        avoided_co2_b = total_co2_base - self.df['e5_co2_g'].sum()

        fig, ax = plt.subplots(figsize=(10, 6))
        x = ['Scenario A2:\nPC pathway', 'Scenario B:\nFleet modernisation', 'Scenario A2:\nMBT pathway']
        y_vals = [avoided_co2_pc, avoided_co2_b, avoided_co2_mbt]
        pcts = [(v / total_co2_base * 100) if total_co2_base > 0 else 0 for v in y_vals]

        bars = ax.bar(x, y_vals, color=['#2b8cbe', '#1a9850', '#d95f02'], width=0.5)

        ax.set_ylabel('Total CO2 avoided (g per 2-hour observation)', fontsize=12, fontname='Times New Roman')
        ax.set_title('CO2 reduction per scenario relative to congested Euro 2 baseline', fontsize=14, fontweight='bold', fontname='Times New Roman')
        ax.set_ylim(0, max(y_vals) * 1.2)

        for bar, val, pct in zip(bars, y_vals, pcts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(y_vals)*0.02), 
                    f"{val:.0f} g\n({pct:.1f}%)", ha='center', va='bottom', fontname='Times New Roman', fontweight='bold')

        plt.tight_layout()
        plt.savefig('fig_4_35_cross_scenario.png', dpi=300)
        plt.close()

    def generate_all_plots(self):
        """Executes all plotting functions for Section 4.5."""
        self.plot_baseline_lollipop()
        self.plot_scenario_a1_nox()
        self.plot_scenario_a2_co2()
        self.plot_scenario_b_absolute_totals()
        self.plot_scenario_b_reductions()
        self.plot_scenario_b_class_disaggregation()
        self.plot_scenario_b_relative_levels()
        self.plot_cross_scenario_comparison()

# To execute the code, place this block at the bottom of the script:
if __name__ == "__main__":
    EVENING_PATH = r"C:\Users\ireoy\OneDrive - University of Cape Town\UCT ACADEMIC YEARS\UCT Masters Second Year\Masters_local_repo\Thesis_Traffic_Emissions_Suite\outputs\2026_02_03 14_59_58 (UTC+02_00)\Experiment_HOMOGRAPHY_20260302_1130\emissions_data_evening.csv"
    
    if os.path.exists(EVENING_PATH):
        plotter = ScenarioPlotter(EVENING_PATH)
        plotter.generate_all_plots()
        print("All scenario plots generated successfully.")
    else:
        print(f"Error: Could not find CSV file at {EVENING_PATH}")