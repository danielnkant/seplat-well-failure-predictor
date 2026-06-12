"""
Seplat Energy - Well Failure Prediction System
Step 1: Synthetic Dataset Generator
Generates 5,475 sensor readings across 15 wells with realistic
failure signatures for 6 failure modes.
"""
 
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os
 
np.random.seed(42)
random.seed(42)
 
# ── Well Configuration ──────────────────────────────────────────────
WELL_CONFIG = {
    "OML_40_W01": {"type":"oil",  "depth_m":2800,"age_years":12,"field":"Sapele",   "risk_class":"high"},
    "OML_40_W02": {"type":"oil",  "depth_m":3100,"age_years":8, "field":"Sapele",   "risk_class":"medium"},
    "OML_40_W03": {"type":"gas",  "depth_m":2200,"age_years":15,"field":"Sapele",   "risk_class":"high"},
    "OML_38_W01": {"type":"oil",  "depth_m":2600,"age_years":5, "field":"Amukpe",  "risk_class":"low"},
    "OML_38_W02": {"type":"oil",  "depth_m":2900,"age_years":11,"field":"Amukpe",  "risk_class":"medium"},
    "OML_38_W03": {"type":"gas",  "depth_m":1900,"age_years":7, "field":"Amukpe",  "risk_class":"low"},
    "OML_56_W01": {"type":"oil",  "depth_m":3400,"age_years":18,"field":"Oben",    "risk_class":"high"},
    "OML_56_W02": {"type":"oil",  "depth_m":2700,"age_years":9, "field":"Oben",    "risk_class":"medium"},
    "OML_56_W03": {"type":"gas",  "depth_m":2100,"age_years":6, "field":"Oben",    "risk_class":"low"},
    "SEPNU_OML67_W01":{"type":"oil","depth_m":3800,"age_years":22,"field":"Qua_Iboe","risk_class":"high"},
    "SEPNU_OML67_W02":{"type":"oil","depth_m":3500,"age_years":14,"field":"Qua_Iboe","risk_class":"medium"},
    "SEPNU_OML68_W01":{"type":"oil","depth_m":4100,"age_years":25,"field":"Yoho",   "risk_class":"high"},
    "SEPNU_OML68_W02":{"type":"gas","depth_m":2800,"age_years":10,"field":"Yoho",   "risk_class":"medium"},
    "SEPNU_OML70_W01":{"type":"oil","depth_m":3200,"age_years":16,"field":"Amenam", "risk_class":"high"},
    "SEPNU_OML70_W02":{"type":"oil","depth_m":2900,"age_years":4, "field":"Amenam", "risk_class":"low"},
}
 
FAILURE_MODES = [
    "ESP_failure", "tubing_leak", "scale_buildup",
    "sand_production", "wax_deposition", "gas_lift_failure",
]
 
def get_baseline(well_id, cfg):
    is_oil = cfg['type'] == 'oil'
    age, depth = cfg['age_years'], cfg['depth_m']
    return {
        'wellhead_pressure_psi':    2800 + depth*0.15 - age*12,
        'tubing_pressure_psi':      1900 + depth*0.10 - age*8,
        'casing_pressure_psi':      1100 + depth*0.05 - age*5,
        'bottomhole_temp_c':        80   + depth*0.018,
        'flow_rate_bopd':           1200 if is_oil else 0,
        'gas_rate_mscfd':           500  if is_oil else 3500,
        'water_cut_pct':            min(10 + age*1.8, 60),
        'gor_scf_stb':              600  if is_oil else 0,
        'choke_size_64ths':         32,
        'vibration_mm_s':           2.5,
        'motor_current_amps':       85   if is_oil else 40,
        'pump_intake_pressure_psi': 800  + depth*0.08,
        'fluid_temperature_c':      65   + depth*0.012,
        'h2s_ppm':                  15   + random.uniform(0,10),
        'co2_pct':                  3.5,
    }
 
def inject_failure_signature(df, failure_mode, failure_day, window=30):
    idx = df[df['days_to_failure'] <= window].index
    if failure_mode == 'ESP_failure':
        for i in idx:
            s = 1 - (df.loc[i,'days_to_failure'] / window)
            df.loc[i,'motor_current_amps']      *= (1 + s*0.45 + np.random.normal(0,0.05))
            df.loc[i,'vibration_mm_s']           *= (1 + s*2.80 + np.random.normal(0,0.10))
            df.loc[i,'pump_intake_pressure_psi'] *= (1 - s*0.35 + np.random.normal(0,0.02))
            df.loc[i,'flow_rate_bopd']           *= (1 - s*0.50 + np.random.normal(0,0.03))
    elif failure_mode == 'tubing_leak':
        for i in idx:
            s = 1 - (df.loc[i,'days_to_failure'] / window)
            df.loc[i,'casing_pressure_psi'] *= (1 + s*0.55 + np.random.normal(0,0.04))
            df.loc[i,'tubing_pressure_psi'] *= (1 - s*0.30 + np.random.normal(0,0.03))
    elif failure_mode == 'scale_buildup':
        for i in idx:
            s = 1 - (df.loc[i,'days_to_failure'] / window)
            df.loc[i,'flow_rate_bopd']           *= (1 - s*0.65 + np.random.normal(0,0.02))
            df.loc[i,'pump_intake_pressure_psi'] *= (1 + s*0.30 + np.random.normal(0,0.02))
    elif failure_mode == 'sand_production':
        for i in idx:
            s = 1 - (df.loc[i,'days_to_failure'] / window)
            df.loc[i,'vibration_mm_s'] *= (1 + s*3.5 + np.random.normal(0,0.15))
    elif failure_mode == 'wax_deposition':
        for i in idx:
            s = 1 - (df.loc[i,'days_to_failure'] / window)
            df.loc[i,'fluid_temperature_c'] *= (1 - s*0.18 + np.random.normal(0,0.01))
            df.loc[i,'flow_rate_bopd']       *= (1 - s*0.45 + np.random.normal(0,0.03))
    elif failure_mode == 'gas_lift_failure':
        for i in idx:
            s = 1 - (df.loc[i,'days_to_failure'] / window)
            df.loc[i,'gas_rate_mscfd'] *= (1 - s*0.75 + np.random.normal(0,0.05))
            df.loc[i,'flow_rate_bopd'] *= (1 - s*0.55 + np.random.normal(0,0.04))
    return df
 
def generate_well_data(well_id, cfg, days=365):
    dates    = [datetime(2024,1,1) + timedelta(days=d) for d in range(days)]
    baseline = get_baseline(well_id, cfg)
    risk_map = {'high':0.70,'medium':0.45,'low':0.20}
    will_fail   = random.random() < risk_map[cfg['risk_class']]
    failure_day  = random.randint(60,340) if will_fail else None
    failure_mode = random.choice(FAILURE_MODES) if will_fail else None
    records = []
    for d, dt in enumerate(dates):
        seasonal = 0.03 * np.sin(2*np.pi*d/365)
        noise    = lambda s=0.02: np.random.normal(0,s)
        rec = {
            'date': dt, 'well_id': well_id, 'field': cfg['field'],
            'well_type': cfg['type'], 'well_age_years': cfg['age_years']+d/365,
            'depth_m': cfg['depth_m'],
            'wellhead_pressure_psi':    max(0,baseline['wellhead_pressure_psi']*(1+seasonal+noise())),
            'tubing_pressure_psi':      max(0,baseline['tubing_pressure_psi']*(1+seasonal+noise())),
            'casing_pressure_psi':      max(0,baseline['casing_pressure_psi']*(1+seasonal+noise(0.015))),
            'pump_intake_pressure_psi': max(0,baseline['pump_intake_pressure_psi']*(1+seasonal+noise(0.015))),
            'flow_rate_bopd':  max(0,baseline['flow_rate_bopd']*(1+seasonal+noise(0.03))),
            'gas_rate_mscfd':  max(0,baseline['gas_rate_mscfd']*(1+seasonal+noise(0.03))),
            'water_cut_pct':   max(0,min(100,baseline['water_cut_pct']+noise(0.5)*10)),
            'gor_scf_stb':     max(0,baseline['gor_scf_stb']*(1+noise(0.03))),
            'choke_size_64ths':baseline['choke_size_64ths']+random.choice([-4,-2,0,0,0,2,4]),
            'bottomhole_temp_c':   max(0,baseline['bottomhole_temp_c']*(1+noise(0.005))),
            'fluid_temperature_c': max(0,baseline['fluid_temperature_c']*(1+noise(0.008))),
            'vibration_mm_s':    max(0,baseline['vibration_mm_s']*(1+noise(0.08))),
            'motor_current_amps':max(0,baseline['motor_current_amps']*(1+noise(0.04))),
            'h2s_ppm':  max(0,baseline['h2s_ppm']*(1+noise(0.05))),
            'co2_pct':  max(0,baseline['co2_pct']+noise(0.01)),
            'failure_mode':    failure_mode if will_fail else 'none',
            'days_to_failure': (failure_day-d) if (will_fail and d<=failure_day) else 999,
            'failed':          int(will_fail and d==failure_day),
        }
        records.append(rec)
    df = pd.DataFrame(records)
    if will_fail:
        df = inject_failure_signature(df, failure_mode, failure_day)
    df['flow_rate_bopd']      = df['flow_rate_bopd'].clip(0,5000)
    df['vibration_mm_s']      = df['vibration_mm_s'].clip(0,25)
    df['motor_current_amps']  = df['motor_current_amps'].clip(0,200)
    df['casing_pressure_psi'] = df['casing_pressure_psi'].clip(0,5000)
    return df
 
def engineer_features(df):
    df = df.sort_values(['well_id','date']).copy()
    for col in ['wellhead_pressure_psi','tubing_pressure_psi',
                'casing_pressure_psi','flow_rate_bopd','gas_rate_mscfd',
                'vibration_mm_s','motor_current_amps','fluid_temperature_c',
                'pump_intake_pressure_psi']:
        df[f'{col}_7d_mean']  = df.groupby('well_id')[col].transform(
            lambda x: x.rolling(7, min_periods=1).mean())
        df[f'{col}_14d_mean'] = df.groupby('well_id')[col].transform(
            lambda x: x.rolling(14,min_periods=1).mean())
        df[f'{col}_7d_std']   = df.groupby('well_id')[col].transform(
            lambda x: x.rolling(7, min_periods=1).std().fillna(0))
        df[f'{col}_roc']      = df.groupby('well_id')[col].transform(
            lambda x: x.pct_change().fillna(0).clip(-1,1))
    df['pressure_differential'] = df['wellhead_pressure_psi'] - df['tubing_pressure_psi']
    df['casing_tubing_ratio']   = df['casing_pressure_psi'] / (df['tubing_pressure_psi']+1)
    df['liquid_rate_bopd']      = df['flow_rate_bopd'] * (1 - df['water_cut_pct']/100)
    df['productivity_index']    = df['flow_rate_bopd'] / (df['wellhead_pressure_psi']+1)
    df['failure_within_30d']    = (df['days_to_failure'] <= 30).astype(int)
    df['failure_within_14d']    = (df['days_to_failure'] <= 14).astype(int)
    df['failure_within_7d']     = (df['days_to_failure'] <= 7).astype(int)
    return df
 
# ── Main ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    print('Generating synthetic well data...')
    all_data = []
    for well_id, cfg in WELL_CONFIG.items():
        print(f'  Processing {well_id}...')
        all_data.append(generate_well_data(well_id, cfg))
    raw_df = pd.concat(all_data, ignore_index=True)
    df = engineer_features(raw_df)
    df.to_csv('data/seplat_well_data.csv', index=False)
    print(f'Done. {len(df):,} records saved to data/seplat_well_data.csv')
