import numpy as np
import yaml
import os

class EmissionCalculator:
    def __init__(self, config_path="config/vehicle_specs.yaml"):
        self.g = 9.81
        self.load_config(config_path)

        # --- THESIS TABLE 3.3: FUEL CONSTANTS (SA STANDARDS) ---
        # 2.26 is kg CO2 / Liter (Volumetric)
        # We must convert Mass -> Volume before applying this.
        self.fuel_factors = {
            'Petrol': {
                'density_g_L': 740.0,   # Grams per Liter
                'co2_factor_kg_L': 2.26 # kg CO2 per Liter
            },
            'Diesel': {
                'density_g_L': 835.0,   # Grams per Liter
                'co2_factor_kg_L': 2.68 # kg CO2 per Liter
            }
        }

        # --- CLASS CONSTANTS (EURO 2 BASELINE) ---
        # g/kWh (Gravimetric emission factors based on Energy)
        # Preserving your existing logic
        self.constants = {
            'PassengerCar': {
                'mass': 1.4, 'bsfc': 300, 'fuel': 'Petrol', 
                'nox_kwh': 0.5, 'co_kwh': 2.2, 'hc_kwh': 0.3, 'pm_kwh': 0.002 
            },
            'SUV': {
                'mass': 1.8, 'bsfc': 320, 'fuel': 'Petrol',
                'nox_kwh': 0.6, 'co_kwh': 2.5, 'hc_kwh': 0.35, 'pm_kwh': 0.003
            },
            'LCV': {
                'mass': 1.9, 'bsfc': 280, 'fuel': 'Diesel',
                'nox_kwh': 1.2, 'co_kwh': 1.5, 'hc_kwh': 0.5, 'pm_kwh': 0.15 
            },
            'Minibus Taxi': {
                'mass': 2.2, 'bsfc': 310, 'fuel': 'Petrol', # Many SA Taxis are Petrol (Toyota Quantums)
                'nox_kwh': 0.8, 'co_kwh': 3.0, 'hc_kwh': 0.6, 'pm_kwh': 0.05
            },
            'Heavy Vehicle': {
                'mass': 12.0, 'bsfc': 240, 'fuel': 'Diesel',
                'nox_kwh': 7.0, 'co_kwh': 4.0, 'hc_kwh': 1.1, 'pm_kwh': 0.30
            },
            'Bus': {
                'mass': 14.0, 'bsfc': 250, 'fuel': 'Diesel',
                'nox_kwh': 7.5, 'co_kwh': 4.5, 'hc_kwh': 1.2, 'pm_kwh': 0.35
            }
        }

        # --- EURO 5 REDUCTION FACTORS (Technology Multipliers) ---
        # Relative to the Euro 2 Baseline above.
        # Derived from Regulation Ratios (e.g. Euro 5 limit / Euro 2 limit)
        self.euro5_mult = {
            'PassengerCar':  {'fuel': 0.90, 'nox': 0.40, 'pm': 0.10, 'hc': 0.50, 'co': 0.60},
            'SUV':           {'fuel': 0.90, 'nox': 0.40, 'pm': 0.10, 'hc': 0.50, 'co': 0.60},
            'LCV':           {'fuel': 0.90, 'nox': 0.72, 'pm': 0.10, 'hc': 0.70, 'co': 0.70}, # Diesel
            'Minibus Taxi':  {'fuel': 0.90, 'nox': 0.50, 'pm': 0.20, 'hc': 0.50, 'co': 0.60},
            'Heavy Vehicle': {'fuel': 0.95, 'nox': 0.60, 'pm': 0.10, 'hc': 0.70, 'co': 0.70}, # SCR + DPF
            'Bus':           {'fuel': 0.95, 'nox': 0.60, 'pm': 0.10, 'hc': 0.70, 'co': 0.70}
        }

    def load_config(self, path):
        if os.path.exists(path):
            with open(path) as f: self.yaml_cfg = yaml.safe_load(f)

    # --- HEAVY-DUTY VEHICLE CLASS SET ---
    # Used to route VSP calculation to the correct formula.
    HDV_CLASSES = {'Heavy Vehicle', 'Bus'}

    def calculate_vsp(self, class_name, v_mps, a_mps2, grade_percent):
        """
        Calculates VSP in kW/ton.
        Routes to the appropriate formula based on vehicle class:
          - Light-Duty (LDV): Passenger Car, SUV, LCV, Minibus Taxi
          - Heavy-Duty (HDV): Heavy Vehicle, Bus

        CHANGE LOG:
          Previously used a single simplified PKE formula for ALL classes.
          Now splits into two separate formulations per standard practice:
            LDV  — Jimenez-Palacios (1999) light-duty VSP
            HDV  — Zhai et al. (2008) heavy-duty VSP with mass-specific coefficients
        """
        if class_name in self.HDV_CLASSES:
            return self._vsp_heavy_duty(class_name, v_mps, a_mps2, grade_percent)
        else:
            return self._vsp_light_duty(class_name, v_mps, a_mps2, grade_percent)

    def _vsp_light_duty(self, class_name, v_mps, a_mps2, grade_percent):
        """
        Light-Duty VSP (Passenger Car, SUV, LCV, Minibus Taxi).
        Formula (Jimenez-Palacios, 1999):
            VSP = v * (1.1*a + 9.81*grade + 0.132) + 0.000302 * v^3
        Units: kW/ton
        
        This is the ORIGINAL formula that was previously applied to all classes.
        NO CHANGE to the math here — only scoped to LDV classes now.
        """
        grade = grade_percent / 100.0
        term1 = 1.1 * a_mps2
        term2 = 9.81 * grade
        term3 = 0.132                   # Rolling resistance term (LDV)
        term4 = 0.000302 * (v_mps**3)   # Aerodynamic drag term (LDV)

        vsp = v_mps * (term1 + term2 + term3) + term4
        return max(0.0, vsp)

    def _vsp_heavy_duty(self, class_name, v_mps, a_mps2, grade_percent):
        """
        Heavy-Duty VSP (Heavy Vehicle, Bus).
        Formula (Zhai et al., 2008 — adapted for HDV):
            VSP = v * (a + 9.81*grade + 0.09199) + 0.000169 * v^3
        Units: kW/ton

        Key differences from LDV:
          - Acceleration coefficient: 1.0 (no 1.1 rotational mass factor;
            HDV drivetrains have lower rotational inertia relative to total mass)
          - Rolling resistance: 0.09199 (higher tire deformation losses on
            multi-axle HDV, but normalised per ton yields a lower coefficient)
          - Aerodynamic drag: 0.000169 (HDV have larger frontal area but
            per-ton drag is lower due to high gross vehicle mass)

        CHANGE LOG:
          NEW — this formula was not present before. Previously all classes
          used the LDV formula above, which over-estimates the per-ton
          rolling resistance and aerodynamic drag terms for heavy vehicles.
        """
        grade = grade_percent / 100.0
        term1 = 1.0 * a_mps2            # No rotational mass uplift for HDV
        term2 = 9.81 * grade
        term3 = 0.09199                  # Rolling resistance term (HDV)
        term4 = 0.000169 * (v_mps**3)   # Aerodynamic drag term (HDV)

        vsp = v_mps * (term1 + term2 + term3) + term4
        return max(0.0, vsp)

    def calculate_emissions(self, vsp, cls, duration_sec=1.0, standard='euro_2'):
        """
        Calculates Total Mass (Grams) for the event.
        standard: Ignored (Returns both Euro 2 and Euro 5)
        """
        c = self.constants.get(cls, self.constants['PassengerCar'])
        
        # 1. Calculate Power Demand (kW)
        p_total_kw = vsp * c['mass'] 
        
        # 2. Determine Fuel Consumption (Grams) - BASELINE
        if p_total_kw > 0.5:
            # RUNNING
            fuel_g = (p_total_kw * c['bsfc'] * duration_sec) / 3600.0
            
            # 3. Calculate Pollutants (Euro 2 Baseline)
            nox_g = (p_total_kw * c['nox_kwh'] * duration_sec) / 3600.0
            co_g  = (p_total_kw * c['co_kwh']  * duration_sec) / 3600.0
            hc_g  = (p_total_kw * c['hc_kwh']  * duration_sec) / 3600.0
            pm_g  = (p_total_kw * c['pm_kwh']  * duration_sec) / 3600.0
            
        else:
            # IDLING
            idle_rate = 0.6 if cls in ['Heavy Vehicle', 'Bus'] else 0.2
            fuel_g = idle_rate * duration_sec
            
            # Idle Pollutants
            nox_g = 0.0001 * duration_sec; co_g = 0.0020 * duration_sec
            hc_g = 0.0003 * duration_sec; pm_g = 0.00005 * duration_sec

        # CO2 Euro 2
        fuel_type = c['fuel']
        props = self.fuel_factors.get(fuel_type, self.fuel_factors['Petrol'])
        co2_g = (fuel_g / props['density_g_L']) * props['co2_factor_kg_L'] * 1000.0

        # --- EURO 5 (Apply Multipliers) ---
        m = self.euro5_mult.get(cls, self.euro5_mult['PassengerCar'])

        return {
            # Euro 2
            'e2_co2': co2_g, 'e2_nox': nox_g, 'e2_fuel': fuel_g, 
            'e2_pm': pm_g, 'e2_hc': hc_g, 'e2_co': co_g,
            # Euro 5 (Applied multipliers)
            'e5_co2': (fuel_g * m['fuel'] / props['density_g_L']) * props['co2_factor_kg_L'] * 1000.0,
            'e5_nox': nox_g * m['nox'],
            'e5_fuel': fuel_g * m['fuel'],
            'e5_pm': pm_g * m['pm'],
            'e5_hc': hc_g * m['hc'],
            'e5_co': co_g * m['co']
        }