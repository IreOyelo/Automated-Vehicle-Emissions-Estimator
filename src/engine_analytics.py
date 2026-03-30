import pandas as pd
import numpy as np
import os
import scipy.signal

class AnalyticsEngine:
    def __init__(self):
        # Thesis Table 3.7: Tier 2 Emission Factors (g/km) for Baseline Comparison
        # These are static national factors used to audit your dynamic VSP model.
        self.tier2_factors = {
            'PC':  {'CO2': 170.0, 'CO': 2.2, 'NOx': 0.5},
            'SUV': {'CO2': 215.0, 'CO': 2.2, 'NOx': 0.5},
            'LCV': {'CO2': 268.0, 'CO': 1.0, 'NOx': 0.7},
            'MBT': {'CO2': 271.0, 'CO': 2.2, 'NOx': 0.5},
            'HV':  {'CO2': 938.0, 'CO': 4.0, 'NOx': 7.0}
        }

    def apply_savgol_smoothing(self, df):
        """
        Thesis Section 3.4.3: Comparative Analysis Filter.
        Creates a '_sg' version of speed for plotting against Kalman filter output.
        """
        if df.empty: return df
        
        # Create column for smoothed speed
        df['speed_sg'] = df['speed_kmh']
        
        unique_ids = df['id'].unique()
        for vid in unique_ids:
            idx = df[df['id'] == vid].index
            # Window 11 (approx 0.3s), Poly order 3
            if len(idx) > 11:
                try:
                    df.loc[idx, 'speed_sg'] = scipy.signal.savgol_filter(
                        df.loc[idx, 'speed_kmh'], window_length=11, polyorder=3
                    )
                except:
                    pass
        return df

    def generate_report(self, data_csv_path):
        """
        Generates the Thesis Results Table: VSP Model vs Tier 2 Baseline.
        """
        if not os.path.exists(data_csv_path):
            print("Error: CSV file not found.")
            return pd.DataFrame()

        df = pd.read_csv(data_csv_path)
        if df.empty: return pd.DataFrame()
        
        # 1. Apply Smoothing (for graphing later)
        df = self.apply_savgol_smoothing(df)
        df.to_csv(data_csv_path.replace(".csv", "_SMOOTHED.csv"), index=False)

        results = []
        
        # 2. Group by Class for Comparative Analysis
        for class_name, group in df.groupby('class'):
            count = group['id'].nunique()
            
            # --- METRIC 1: DYNAMIC VSP MODEL (Our Thesis) ---
            # Sum of instantaneous emissions calculated by engine_emissions.py
            vsp_co2_kg = group['co2_rate'].sum() / 1000.0
            
            # --- METRIC 2: STATIC TIER 2 BASELINE (Standard) ---
            # Formula: Total Distance (km) * Factor (g/km)
            # Total Distance = Sum(Speed m/s * 1s) / 1000
            total_vkt_km = (group['speed_kmh'].sum() / 3.6) / 1000.0
            
            # Lookup Tier 2 Factor (Default to PC if unknown)
            # Note: Maps 'Passenger Car' -> 'PC' if needed, or ensures names match exactly
            key_map = {
                'Passenger Car': 'PC', 'Minibus Taxi': 'MBT', 
                'Light Comm. Vehicle': 'LCV', 'Heavy Vehicle': 'HV', 'SUV': 'SUV'
            }
            lookup_key = key_map.get(class_name, 'PC')
            factors = self.tier2_factors.get(lookup_key, self.tier2_factors['PC'])
            
            baseline_co2_kg = (total_vkt_km * factors['CO2']) / 1000.0
            
            # --- METRIC 3: DIVERGENCE (%) ---
            # Thesis Eq: (VSP - Tier2) / Tier2
            if baseline_co2_kg > 0:
                divergence = ((vsp_co2_kg - baseline_co2_kg) / baseline_co2_kg) * 100.0
            else:
                divergence = 0.0

            results.append({
                'Class': class_name,
                'Count': count,
                'Total VKT (km)': round(total_vkt_km, 3),
                'Avg Speed': round(group['speed_kmh'].mean(), 1),
                'VSP Model CO2 (kg)': round(vsp_co2_kg, 2),
                'Tier 2 Baseline CO2 (kg)': round(baseline_co2_kg, 2),
                'Divergence (%)': round(divergence, 1)
            })

        report_df = pd.DataFrame(results)
        output_path = data_csv_path.replace(".csv", "_REPORT.csv")
        report_df.to_csv(output_path, index=False)
        
        print(f"--- THESIS COMPLIANCE REPORT ---")
        print(report_df.to_string(index=False))
        print(f"Saved to: {output_path}")
        
        return report_df