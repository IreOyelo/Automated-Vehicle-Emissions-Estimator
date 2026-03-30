import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend – avoids Qt platform plugin errors
import matplotlib.pyplot as plt
import seaborn as sns
import os

class SensitivityAnalyzer:
    def __init__(self, evening_csv_path):
        self.csv_path = evening_csv_path
        self.df = pd.read_csv(evening_csv_path, sep=None, engine='python')

        # Clean mixed decimal separators (comma vs dot), same as Thesis_Plots_Master
        for col in self.df.columns:
            if col != 'class':
                if self.df[col].dtype == object:
                    self.df[col] = self.df[col].astype(str).str.replace(',', '.')
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # Enforce Times New Roman for all text elements in the plots
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['mathtext.fontset'] = 'stix'

    def calculate_parameter_sensitivity(self):
        """Calculates mass and BSFC sensitivity and generates a bar chart."""
        pc_data = self.df[self.df['class'].isin(['PC', 'Passenger Car'])].copy()

        def calc_co2_row(row, mass_mult=1.0, bsfc_mult=1.0):
            v_kmh = row['speed_kmh']
            vsp = row['vsp_kw_ton']
            mass = 1.4 * mass_mult
            bsfc = 300 * bsfc_mult

            v_mps = max(0.1, v_kmh / 3.6)
            duration = 100.0 / v_mps
            P_total_kw = vsp * mass

            if P_total_kw > 0.5:
                fuel_g = (P_total_kw * bsfc * duration) / 3600.0
            else:
                fuel_g = 0.2 * duration

            co2_g = (fuel_g / 740.0) * 2.26 * 1000.0
            return co2_g

        base_co2 = pc_data.apply(lambda r: calc_co2_row(r), axis=1).mean()
        m_low  = pc_data.apply(lambda r: calc_co2_row(r, mass_mult=0.8), axis=1).mean()
        m_high = pc_data.apply(lambda r: calc_co2_row(r, mass_mult=1.2), axis=1).mean()
        b_low  = pc_data.apply(lambda r: calc_co2_row(r, bsfc_mult=0.8), axis=1).mean()
        b_high = pc_data.apply(lambda r: calc_co2_row(r, bsfc_mult=1.2), axis=1).mean()
        c_low  = pc_data.apply(lambda r: calc_co2_row(r, mass_mult=0.8, bsfc_mult=0.8), axis=1).mean()
        c_high = pc_data.apply(lambda r: calc_co2_row(r, mass_mult=1.2, bsfc_mult=1.2), axis=1).mean()

        results = {
            'Mass':     [((m_low  - base_co2) / base_co2) * 100, ((m_high  - base_co2) / base_co2) * 100],
            'BSFC':     [((b_low  - base_co2) / base_co2) * 100, ((b_high  - base_co2) / base_co2) * 100],
            'Combined': [((c_low  - base_co2) / base_co2) * 100, ((c_high  - base_co2) / base_co2) * 100],
        }

        fig, ax = plt.subplots(figsize=(9, 5))  # slightly wider for label room
        labels = list(results.keys())
        negative_impact = [results[l][0] for l in labels]
        positive_impact = [results[l][1] for l in labels]

        y = np.arange(len(labels))
        height = 0.4
        rects1 = ax.barh(y + height / 2, positive_impact, height, label='+20% Adjustment', color='#d95f02')
        rects2 = ax.barh(y - height / 2, negative_impact, height, label='-20% Adjustment', color='#1b9e77')

        ax.set_xlabel('Percentage Change in Mean CO₂ Output (%)', fontsize=12)
        ax.set_title('Parameter Sensitivity Analysis for Passenger Car', fontsize=14)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=12)
        ax.legend(prop={'family': 'Times New Roman', 'size': 10})
        ax.axvline(0, color='black', linewidth=1)

        # Adaptive label placement: inside the bar (white) when bar is long,
        # outside (default colour) when bar is short – avoids clipping on Combined.
        all_vals = positive_impact + [abs(v) for v in negative_impact]
        x_range  = max(all_vals) - min(min(negative_impact), 0)
        threshold = x_range * 0.15  # bars longer than 15% of range get inside label

        for rect, val in zip(list(rects1) + list(rects2),
                             positive_impact + negative_impact):
            bar_len = abs(val)
            sign    = 1 if val >= 0 else -1
            if bar_len > threshold:
                # Place inside the bar, near its far end
                x_pos = val - sign * 1.5
                ax.text(x_pos, rect.get_y() + rect.get_height() / 2,
                        f'{val:.1f}%', va='center',
                        ha='right' if val >= 0 else 'left',
                        color='white', fontsize=9, weight='bold')
            else:
                # Place just outside the bar end
                x_pos = val + sign * 0.8
                ax.text(x_pos, rect.get_y() + rect.get_height() / 2,
                        f'{val:.1f}%', va='center',
                        ha='left' if val >= 0 else 'right',
                        color='black', fontsize=9)

        plt.tight_layout()
        plt.savefig('parameter_sensitivity.png', dpi=300)
        plt.close()
        print("Saved: parameter_sensitivity.png")

    def calculate_error_propagation(self):
        """Calculates expected emission variance based on post-audit confusion matrix."""
        raw_counts = np.array([
            [373, 2,  0,  0, 27],
            [2,  140,  0,  0,  4],
            [0,    1, 28,  0,  0],
            [0,    1,  0, 42,  0],
            [6,    7,  0,  0, 147],
        ])

        norm_matrix = raw_counts / raw_counts.sum(axis=1, keepdims=True)
        classes   = ['PassengerCar', 'LCV', 'Minibus Taxi', 'Heavy Vehicle', 'SUV']
        class_map = {'PassengerCar': 'PC', 'LCV': 'LCV', 'Minibus Taxi': 'MBT',
                     'Heavy Vehicle': 'HV', 'SUV': 'SUV'}

        mean_co2 = {}
        for cls in classes:
            subset = self.df[self.df['class'] == class_map[cls]]
            mean_co2[cls] = subset['e2_co2_g'].mean() if not subset.empty else 0.0

        expected_co2   = []
        true_co2_list  = []
        for i, cls_true in enumerate(classes):
            exp_val = sum(norm_matrix[i, j] * mean_co2[classes[j]] for j in range(5))
            expected_co2.append(exp_val)
            true_co2_list.append(mean_co2[cls_true])

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(classes))
        width = 0.35

        rects1 = ax.bar(x - width / 2, true_co2_list, width, label='True Class Mean', color='#2b8cbe')
        rects2 = ax.bar(x + width / 2, expected_co2,  width, label='Expected Mean (with Confusion)', color='#fdb462')

        ax.set_ylabel('Mean CO₂ Output (g per 100 m)', fontsize=12)
        ax.set_title('Impact of Classification Errors on Class Mean CO₂ Output', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(classes, fontsize=12)
        ax.legend(prop={'family': 'Times New Roman', 'size': 10})

        for i in range(len(classes)):
            diff = expected_co2[i] - true_co2_list[i]
            pct  = (diff / true_co2_list[i]) * 100 if true_co2_list[i] != 0 else 0
            ax.text(x[i] + width / 2, expected_co2[i] + 0.2,
                    f"{diff:+.2f}g\n({pct:+.1f}%)",
                    ha='center', va='bottom', fontsize=10, color='darkred')

        ax.set_ylim(0, max(max(true_co2_list), max(expected_co2)) * 1.2)
        plt.tight_layout()
        plt.savefig('error_propagation.png', dpi=300)
        plt.close()
        print("Saved: error_propagation.png")


if __name__ == "__main__":
    analyzer = SensitivityAnalyzer(
        r"C:\Users\ireoy\OneDrive - University of Cape Town\UCT ACADEMIC YEARS\UCT Masters Second Year"
        r"\Masters_local_repo\Thesis_Traffic_Emissions_Suite\outputs"
        r"\2026_02_03 14_59_58 (UTC+02_00)\Experiment_HOMOGRAPHY_20260302_1130\emissions_data_evening.csv"
    )
    analyzer.calculate_parameter_sensitivity()
    analyzer.calculate_error_propagation()