"""
Seplat Energy - Well Failure Prediction System
Step 2: Exploratory Data Analysis
Reads data/seplat_well_data.csv and generates 6 charts in outputs/
"""
 
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')          # IMPORTANT: use non-interactive backend for Codespaces
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import stats
import os, warnings
warnings.filterwarnings('ignore')
 
# ── Output directory ────────────────────────────────────────────────
os.makedirs('outputs', exist_ok=True)
 
# ── Colour constants ─────────────────────────────────────────────────
SEPLAT_GREEN = '#006B3C'
SEPLAT_GOLD  = '#C8922A'
DANGER_RED   = '#C0392B'
WARN_ORANGE  = '#E67E22'
SAFE_BLUE    = '#2980B9'
NEUTRAL_GREY = '#7F8C8D'
 
plt.rcParams.update({
    'figure.facecolor':'#F8F9FA','axes.facecolor':'#FFFFFF',
    'axes.grid':True,'grid.alpha':0.3,'grid.color':'#CCCCCC',
    'font.family':'DejaVu Sans',
    'axes.spines.top':False,'axes.spines.right':False,
})
 
# ── Load data ────────────────────────────────────────────────────────
df = pd.read_csv('data/seplat_well_data.csv', parse_dates=['date'])
df['month'] = df['date'].dt.month
print(f'Loaded {len(df):,} records | {df["well_id"].nunique()} wells')
 
# ── Chart 1: Failure Landscape ───────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('SEPLAT ENERGY | Well Failure Landscape',
             fontsize=14, fontweight='bold', color=SEPLAT_GREEN, y=1.02)
failed  = df[df['failed']==1]
mcounts = failed['failure_mode'].value_counts()
mlabels = [m.replace('_','\n') for m in mcounts.index]
bcolors = [DANGER_RED,WARN_ORANGE,SEPLAT_GOLD,SAFE_BLUE,'#8E44AD',NEUTRAL_GREY]
bars = axes[0].barh(mlabels, mcounts.values, color=bcolors[:len(mcounts)],
                    edgecolor='white', height=0.6)
axes[0].set_xlabel('Wells Affected'); axes[0].set_title('Failure Mode Frequency')
for bar,val in zip(bars,mcounts.values):
    axes[0].text(val+0.05,bar.get_y()+bar.get_height()/2,str(val),va='center',fontsize=10,fontweight='bold')
ff = df[df['failed']==1].groupby('field')['well_id'].nunique()
ft = df.groupby('field')['well_id'].nunique()
fr = (ff/ft*100).fillna(0).sort_values()
fc = [DANGER_RED if v>=60 else WARN_ORANGE if v>=40 else SAFE_BLUE for v in fr.values]
axes[1].barh(fr.index,fr.values,color=fc,edgecolor='white',height=0.6)
axes[1].set_xlabel('Failure Rate (%)'); axes[1].set_title('Failure Rate by Field')
mf = df[df['failed']==1].groupby('month').size().reindex(range(1,13),fill_value=0)
axes[2].bar(mf.index,mf.values,color=SEPLAT_GREEN,alpha=0.8,edgecolor='white')
axes[2].set_xticks(range(1,13))
axes[2].set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'],fontsize=8)
axes[2].axvspan(4.5,10.5,alpha=0.08,color=SAFE_BLUE,label='Wet season')
axes[2].legend(fontsize=8); axes[2].set_title('Failures by Month')
plt.tight_layout()
plt.savefig('outputs/chart1_failure_landscape.png', dpi=150, bbox_inches='tight')
plt.close(); print('  chart1 saved')
 
# ── Chart 2: Degradation Signatures ─────────────────────────────────
fig = plt.figure(figsize=(18,12))
fig.suptitle('SEPLAT ENERGY | Pre-Failure Sensor Degradation Signatures',
             fontsize=14, fontweight='bold', color=SEPLAT_GREEN)
fm = {
    'ESP_failure':      ('motor_current_amps','vibration_mm_s','Motor I (A)','Vibration'),
    'tubing_leak':      ('casing_pressure_psi','tubing_pressure_psi','Casing P','Tubing P'),
    'scale_buildup':    ('flow_rate_bopd','pump_intake_pressure_psi','Flow Rate','Pump IP'),
    'gas_lift_failure': ('gas_rate_mscfd','flow_rate_bopd','Gas Rate','Flow Rate'),
    'wax_deposition':   ('fluid_temperature_c','flow_rate_bopd','Fluid Temp','Flow Rate'),
    'sand_production':  ('vibration_mm_s','flow_rate_bopd','Vibration','Flow Rate'),
}
gs = gridspec.GridSpec(3,4,figure=fig,hspace=0.50,wspace=0.38)
for pi,(mode,(s1,s2,l1,l2)) in enumerate(fm.items()):
    wd = df[(df['failure_mode']==mode)&(df['days_to_failure']<999)].copy()
    if not len(wd): continue
    for c in [s1,s2]:
        mn,mx = wd[c].min(),wd[c].max()
        wd[f'{c}_n'] = (wd[c]-mn)/(mx-mn+1e-9)*100
    r,cp = divmod(pi,3)
    ax = fig.add_subplot(gs[r,cp])
    a1 = wd.groupby('days_to_failure')[f'{s1}_n'].mean()
    a2 = wd.groupby('days_to_failure')[f'{s2}_n'].mean()
    ax.plot(a1.index,a1.values,color=DANGER_RED,lw=2.2,label=l1)
    ax.plot(a2.index,a2.values,color=SAFE_BLUE, lw=2.2,label=l2,linestyle='--')
    ax.axvspan(0,7,alpha=0.10,color=DANGER_RED)
    ax.axvspan(7,14,alpha=0.06,color=WARN_ORANGE)
    ax.set_title(mode.replace('_',' ').title(),fontsize=9)
    ax.set_xlabel('Days Before Failure',fontsize=8)
    ax.set_ylabel('Normalised (0-100)',fontsize=7)
    ax.set_xlim(30,0); ax.legend(fontsize=6.5)
    ax.tick_params(labelsize=7)
plt.savefig('outputs/chart2_degradation_signatures.png',dpi=150,bbox_inches='tight')
plt.close(); print('  chart2 saved')
 
# ── Chart 3: Correlation Matrix ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(13,10))
fig.suptitle('SEPLAT ENERGY | Sensor Correlation Matrix',fontsize=13,
             fontweight='bold',color=SEPLAT_GREEN)
core = ['wellhead_pressure_psi','tubing_pressure_psi','casing_pressure_psi',
        'pump_intake_pressure_psi','flow_rate_bopd','gas_rate_mscfd',
        'water_cut_pct','vibration_mm_s','motor_current_amps',
        'fluid_temperature_c','vibration_mm_s_7d_std','flow_rate_bopd_7d_std',
        'pressure_differential','casing_tubing_ratio','productivity_index',
        'failure_within_30d']
short = ['WH_P','Tub_P','Cas_P','PumpIP','Flow','Gas','WCut','Vib',
         'MotorI','FluidT','Vib7dSD','Flow7dSD','PDiff','CTRatio','ProdIdx','Fail30d']
corr = df[core].corr()
corr.columns = short; corr.index = short
mask = np.triu(np.ones_like(corr,dtype=bool),k=1)
sns.heatmap(corr,mask=mask,cmap=sns.diverging_palette(220,10,as_cmap=True),
            center=0,vmin=-1,vmax=1,annot=True,fmt='.2f',
            annot_kws={'size':7.5},square=True,linewidths=0.3,ax=ax,
            cbar_kws={'shrink':0.6})
ax.tick_params(axis='x',rotation=45,labelsize=8)
ax.tick_params(axis='y',rotation=0, labelsize=8)
plt.tight_layout()
plt.savefig('outputs/chart3_correlation_matrix.png',dpi=150,bbox_inches='tight')
plt.close(); print('  chart3 saved')
 
# ── Chart 4: Well Health Timeline ────────────────────────────────────
target = next(w for w in df['well_id'].unique()
              if df[(df['well_id']==w)&(df['failed']==1)].shape[0]>0)
wdf   = df[df['well_id']==target].sort_values('date')
fdate = pd.Timestamp(wdf[wdf['failed']==1]['date'].values[0])
fmode = wdf[wdf['failed']==1]['failure_mode'].values[0]
fig, axes4 = plt.subplots(3,1,figsize=(16,11),sharex=True)
fig.suptitle(f'SEPLAT ENERGY | Well Health Timeline — {target}\n{fmode.replace("_"," ").title()}',
             fontsize=13,fontweight='bold',color=SEPLAT_GREEN)
axes4[0].plot(wdf['date'],wdf['wellhead_pressure_psi'],color=SEPLAT_GREEN,lw=1.6,label='Wellhead')
axes4[0].plot(wdf['date'],wdf['tubing_pressure_psi'],  color=SAFE_BLUE,   lw=1.6,label='Tubing')
axes4[0].plot(wdf['date'],wdf['casing_pressure_psi'],  color=WARN_ORANGE, lw=1.6,label='Casing')
ax4b = axes4[1].twinx()
axes4[1].plot(wdf['date'],wdf['flow_rate_bopd'],color=SEPLAT_GOLD,lw=1.8,label='Flow (BOPD)')
ax4b.plot(wdf['date'],wdf['vibration_mm_s'],   color=DANGER_RED, lw=1.6,linestyle='--',label='Vibration')
ax4c = axes4[2].twinx()
axes4[2].plot(wdf['date'],wdf['motor_current_amps'],       color='#8E44AD',lw=1.8,label='Motor I (A)')
ax4c.plot(wdf['date'],wdf['pump_intake_pressure_psi'],color=SAFE_BLUE,lw=1.6,linestyle='--',label='Pump IP')
for ax in axes4:
    ax.axvline(fdate,color=DANGER_RED,lw=2.5,linestyle='--')
    ax.axvspan(fdate-pd.Timedelta(days=30),fdate,alpha=0.05,color=WARN_ORANGE)
    ax.axvspan(fdate-pd.Timedelta(days=7), fdate,alpha=0.09,color=DANGER_RED)
for ax,title in zip(axes4,['A — Pressures','B — Flow & Vibration','C — ESP Health']):
    ax.set_title(title,fontsize=10)
    ax.legend(fontsize=8,loc='upper left')
axes4[2].set_xlabel('Date')
plt.tight_layout()
plt.savefig('outputs/chart4_well_health_timeline.png',dpi=150,bbox_inches='tight')
plt.close(); print('  chart4 saved')
 
# ── Chart 5: Distribution Shift ──────────────────────────────────────
fig, axes5 = plt.subplots(2,3,figsize=(16,9))
fig.suptitle('SEPLAT ENERGY | Sensor Distribution: Normal vs Pre-Failure',
             fontsize=13,fontweight='bold',color=SEPLAT_GREEN)
sc = [('vibration_mm_s','Vibration (mm/s)'),
      ('motor_current_amps','Motor Current (A)'),
      ('casing_pressure_psi','Casing Pressure (PSI)'),
      ('flow_rate_bopd','Flow Rate (BOPD)'),
      ('fluid_temperature_c','Fluid Temperature (C)'),
      ('pump_intake_pressure_psi','Pump Intake Pressure')]
nd = df[df['days_to_failure']==999]
pd2 = df[(df['days_to_failure']<=30)&(df['days_to_failure']>=0)]
for ax,(sensor,label) in zip(axes5.flatten(),sc):
    nv,pv = nd[sensor].dropna(), pd2[sensor].dropna()
    ax.hist(nv,bins=40,alpha=0.5,color=SAFE_BLUE, density=True,label=f'Normal (n={len(nv):,})')
    ax.hist(pv,bins=40,alpha=0.5,color=DANGER_RED,density=True,label=f'Pre-fail (n={len(pv):,})')
    for vals,c in [(nv,SAFE_BLUE),(pv,DANGER_RED)]:
        if len(vals)>10:
            kde=stats.gaussian_kde(vals)
            xr=np.linspace(vals.min(),vals.max(),200)
            ax.plot(xr,kde(xr),color=c,lw=2.2)
    ks,p = stats.ks_2samp(nv,pv)
    sig  = '★ p<0.001' if p<0.001 else 'p<0.05' if p<0.05 else 'n.s.'
    ax.set_title(f'{label}\nKS={ks:.3f} {sig}',fontsize=9)
    ax.legend(fontsize=7.5); ax.tick_params(labelsize=7)
plt.tight_layout()
plt.savefig('outputs/chart5_distribution_shift.png',dpi=150,bbox_inches='tight')
plt.close(); print('  chart5 saved')
 
# ── Chart 6: Volatility & Age ────────────────────────────────────────
fig, axes6 = plt.subplots(1,2,figsize=(15,6))
fig.suptitle('SEPLAT ENERGY | Volatility & Well Age Analysis',
             fontsize=13,fontweight='bold',color=SEPLAT_GREEN)
valid = df[df['days_to_failure']<999].copy()
dr    = list(range(0,31))
for col,color,label in [
    ('vibration_mm_s_7d_std',DANGER_RED,'Vibration volatility'),
    ('wellhead_pressure_psi_7d_std',SEPLAT_GREEN,'Pressure volatility'),
    ('flow_rate_bopd_7d_std',SAFE_BLUE,'Flow rate volatility')]:
    agg = valid.groupby('days_to_failure')[col].mean().reindex(dr)
    axes6[0].fill_between(dr,agg.values,alpha=0.18,color=color)
    axes6[0].plot(dr,agg.values,color=color,lw=2.2,label=label)
axes6[0].axvspan(0,7,alpha=0.08,color=DANGER_RED)
axes6[0].set_xlim(30,0); axes6[0].legend(fontsize=8)
axes6[0].set_xlabel('Days Before Failure'); axes6[0].set_ylabel('7-Day Std Dev')
axes6[0].set_title('Sensor Volatility Escalation (14-21 days early warning)')
af = df.groupby('well_id').agg(
    age=('well_age_years','mean'),failed=('failed','max'),field=('field','first')
).reset_index()
flds   = af['field'].unique()
fcolors= dict(zip(flds,[SEPLAT_GREEN,SEPLAT_GOLD,DANGER_RED,SAFE_BLUE,WARN_ORANGE,'#8E44AD']))
for _,row in af.iterrows():
    axes6[1].scatter(row['age'],np.random.uniform(0.5,1.5),
        c=fcolors.get(row['field'],NEUTRAL_GREY),
        marker='X' if row['failed'] else 'o',
        s=130 if row['failed'] else 80,alpha=0.85,edgecolors='white')
axes6[1].axvline(10,color=WARN_ORANGE,linestyle='--',alpha=0.6)
axes6[1].set_xlabel('Well Age (Years)'); axes6[1].set_yticks([])
axes6[1].set_title('Well Age vs Failure (X=failed, O=active)')
lp = [mpatches.Patch(color=c,label=f) for f,c in fcolors.items()]
axes6[1].legend(handles=lp,fontsize=7,loc='upper left')
plt.tight_layout()
plt.savefig('outputs/chart6_volatility_and_age.png',dpi=150,bbox_inches='tight')
plt.close(); print('  chart6 saved')
 
print('\n✓ All 6 EDA charts saved to outputs/')
print('✓ Step 2 complete — ready for Step 3: Model Building')
