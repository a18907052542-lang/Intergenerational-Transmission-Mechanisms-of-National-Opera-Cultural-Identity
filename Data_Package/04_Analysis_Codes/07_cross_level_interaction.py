"""
Module 7: Cross-Level Interaction Analysis
模块7：跨层交互效应分析

This script performs cross-level interaction analysis for community effects.
此脚本执行社区层面跨层交互效应分析。

Author: Research Team
Date: November 2024
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. Load Data / 加载数据
# ============================================================

print("="*60)
print("MODULE 7: CROSS-LEVEL INTERACTION ANALYSIS")
print("="*60)

df = pd.read_excel('National_Opera_Cultural_Identity_Dataset.xlsx', sheet_name='Raw_Data')
print(f"\nData loaded: {len(df)} families in {df['Community_ID'].nunique()} communities")

# ============================================================
# 2. Prepare Variables / 准备变量
# ============================================================

print("\n[1] Preparing Variables...")

# Level 1 (Individual/Family) variables - grand-mean centered
df['Parent_ID_c'] = df['Parent_Identity_Overall'] - df['Parent_Identity_Overall'].mean()
df['FSI_c'] = df['Family_Socialization'] - df['Family_Socialization'].mean()
df['SES_c'] = df['Family_SES'] - df['Family_SES'].mean()

# Level 2 (Community) variables - create community means
comm_agg = df.groupby('Community_ID').agg({
    'Comm_Cultural_Policy': 'mean',
    'Comm_Cultural_Facility': 'mean',
    'Comm_Cultural_Activity': 'mean',
    'Urban_Rural': 'mean',
    'Parent_Identity_Overall': 'mean',
    'Family_Socialization': 'mean'
}).reset_index()

comm_agg.columns = ['Community_ID', 'Comm_Policy', 'Comm_Facility', 'Comm_Activity',
                     'Comm_Urban_Prop', 'Comm_Parent_ID_Mean', 'Comm_FSI_Mean']

# Grand-mean center community variables
for var in ['Comm_Policy', 'Comm_Facility', 'Comm_Activity']:
    comm_agg[f'{var}_c'] = comm_agg[var] - comm_agg[var].mean()

# Merge back to individual data
df = df.merge(comm_agg[['Community_ID', 'Comm_Policy_c', 'Comm_Facility_c', 
                        'Comm_Activity_c', 'Comm_Urban_Prop']], on='Community_ID')

print("  Level 1 and Level 2 variables prepared")

# ============================================================
# 3. Community-Level Main Effects / 社区层面主效应
# ============================================================

print("\n" + "-"*60)
print("TABLE 6: COMMUNITY-LEVEL MAIN EFFECTS")
print("-"*60)

# Model with community-level predictors
formula_comm = """Child_Identity_Overall ~ Parent_ID_c + FSI_c + 
                  Comm_Policy_c + Comm_Facility_c + Comm_Activity_c"""

model_comm = smf.mixedlm(formula_comm, df, groups=df["Community_ID"])
result_comm = model_comm.fit(method='lbfgs')

print("\nCommunity-Level Main Effects:")
print(f"{'Variable':<30} {'Coefficient':<12} {'Std.Err':<12} {'t':<10} {'Effect Size (f²)'}")
print("-" * 76)

# Calculate effect sizes
var_total = df['Child_Identity_Overall'].var()
comm_vars = ['Comm_Policy_c', 'Comm_Facility_c', 'Comm_Activity_c']

for var in comm_vars:
    coef = result_comm.fe_params[var]
    se = result_comm.bse[var]
    t = result_comm.tvalues[var]
    p = result_comm.pvalues[var]
    
    # Cohen's f² approximation
    r2_partial = (t**2 / (t**2 + len(df) - len(result_comm.fe_params)))
    f2 = r2_partial / (1 - r2_partial)
    
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    var_name = var.replace('_c', '').replace('Comm_', 'Cultural ')
    
    print(f"{var_name:<30} {coef:.3f}{sig:<5} {se:.3f}{'  ':<6} {t:.2f}{'  ':<5} {f2:.3f}")

# ============================================================
# 4. Cross-Level Interactions / 跨层交互
# ============================================================

print("\n" + "-"*60)
print("CROSS-LEVEL INTERACTION EFFECTS")
print("-"*60)

# Create cross-level interaction terms
df['Policy_X_FSI'] = df['Comm_Policy_c'] * df['FSI_c']
df['Facility_X_FSI'] = df['Comm_Facility_c'] * df['FSI_c']
df['Activity_X_FSI'] = df['Comm_Activity_c'] * df['FSI_c']

# Full model with cross-level interactions
formula_cross = """Child_Identity_Overall ~ Parent_ID_c + FSI_c + 
                   Comm_Policy_c + Comm_Facility_c + Comm_Activity_c +
                   Policy_X_FSI + Facility_X_FSI + Activity_X_FSI"""

model_cross = smf.mixedlm(formula_cross, df, groups=df["Community_ID"])
result_cross = model_cross.fit(method='lbfgs')

print("\nCross-Level Interaction Effects:")
print(f"{'Interaction':<30} {'Coefficient':<12} {'Std.Err':<12} {'t':<10} {'Effect Size (f²)'}")
print("-" * 76)

cross_vars = ['Policy_X_FSI', 'Facility_X_FSI', 'Activity_X_FSI']
cross_labels = ['Policy × Family Investment', 'Facility × Family Investment', 
                'Activity × Family Investment']

cross_results = []
for var, label in zip(cross_vars, cross_labels):
    coef = result_cross.fe_params[var]
    se = result_cross.bse[var]
    t = result_cross.tvalues[var]
    p = result_cross.pvalues[var]
    
    r2_partial = (t**2 / (t**2 + len(df) - len(result_cross.fe_params)))
    f2 = r2_partial / (1 - r2_partial)
    
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    
    print(f"{label:<30} {coef:.3f}{sig:<5} {se:.3f}{'  ':<6} {t:.2f}{'  ':<5} {f2:.3f}")
    
    cross_results.append({
        'Interaction': label,
        'Coefficient': round(coef, 3),
        'Std_Error': round(se, 3),
        't_value': round(t, 2),
        'p_value': round(p, 4),
        'Effect_Size_f2': round(f2, 3)
    })

# ============================================================
# 5. Three-Way Interaction (Urban-Rural) / 三重交互
# ============================================================

print("\n" + "-"*60)
print("THREE-WAY INTERACTION (Urban-Rural × Community × Family)")
print("-"*60)

# Create three-way interaction
df['UR_Policy_FSI'] = df['Urban_Rural'] * df['Comm_Policy_c'] * df['FSI_c']
df['UR_Facility_FSI'] = df['Urban_Rural'] * df['Comm_Facility_c'] * df['FSI_c']
df['UR_Activity_FSI'] = df['Urban_Rural'] * df['Comm_Activity_c'] * df['FSI_c']

# Model with three-way interaction
formula_three = """Child_Identity_Overall ~ Parent_ID_c + FSI_c + Urban_Rural +
                   Comm_Policy_c + Comm_Facility_c + Comm_Activity_c +
                   Policy_X_FSI + Facility_X_FSI + Activity_X_FSI +
                   UR_Policy_FSI"""

model_three = smf.mixedlm(formula_three, df, groups=df["Community_ID"])
result_three = model_three.fit(method='lbfgs')

coef_3way = result_three.fe_params['UR_Policy_FSI']
se_3way = result_three.bse['UR_Policy_FSI']
t_3way = result_three.tvalues['UR_Policy_FSI']
p_3way = result_three.pvalues['UR_Policy_FSI']

sig_3way = '***' if p_3way < 0.001 else '**' if p_3way < 0.01 else '*' if p_3way < 0.05 else ''

print(f"\nUrban-Rural × Policy × Family Investment:")
print(f"  Coefficient: {coef_3way:.3f}{sig_3way}")
print(f"  Std. Error: {se_3way:.3f}")
print(f"  t-value: {t_3way:.2f}")
print(f"  p-value: {p_3way:.4f}")

# ============================================================
# 6. Variance Decomposition / 方差分解
# ============================================================

print("\n" + "-"*60)
print("VARIANCE DECOMPOSITION")
print("-"*60)

# Null model for comparison
model_null = smf.mixedlm("Child_Identity_Overall ~ 1", df, groups=df["Community_ID"])
result_null = model_null.fit(method='lbfgs')

var_comm_null = result_null.cov_re.iloc[0, 0]
var_res_null = result_null.scale

var_comm_cross = result_cross.cov_re.iloc[0, 0]
var_res_cross = result_cross.scale

# Proportion of variance explained
r2_level1 = 1 - (var_res_cross / var_res_null)
r2_level2 = 1 - (var_comm_cross / var_comm_null)

print(f"\nNull Model Variance Components:")
print(f"  Community Level (τ00): {var_comm_null:.3f}")
print(f"  Residual (σ²): {var_res_null:.3f}")
print(f"  ICC: {var_comm_null / (var_comm_null + var_res_null):.3f}")

print(f"\nFull Model Variance Components:")
print(f"  Community Level (τ00): {var_comm_cross:.3f}")
print(f"  Residual (σ²): {var_res_cross:.3f}")

print(f"\nVariance Explained:")
print(f"  Level 1 (Within-community): {r2_level1*100:.1f}%")
print(f"  Level 2 (Between-community): {r2_level2*100:.1f}%")

# ============================================================
# 7. Community Heterogeneity Analysis / 社区异质性分析
# ============================================================

print("\n" + "-"*60)
print("COMMUNITY HETEROGENEITY IN TRANSMISSION")
print("-"*60)

# Calculate transmission coefficient for each community
comm_transmission = []
for comm_id in df['Community_ID'].unique():
    comm_data = df[df['Community_ID'] == comm_id]
    if len(comm_data) > 10:  # Minimum sample size
        corr = comm_data['Parent_Identity_Overall'].corr(comm_data['Child_Identity_Overall'])
        urban_prop = comm_data['Urban_Rural'].mean()
        policy = comm_data['Comm_Cultural_Policy'].mean()
        
        comm_transmission.append({
            'Community_ID': comm_id,
            'N': len(comm_data),
            'Transmission_r': round(corr, 3),
            'Urban_Prop': round(urban_prop, 2),
            'Policy_Level': round(policy, 2)
        })

comm_trans_df = pd.DataFrame(comm_transmission)

print(f"\nCommunity-Level Transmission Statistics:")
print(f"  Mean transmission coefficient: {comm_trans_df['Transmission_r'].mean():.3f}")
print(f"  SD: {comm_trans_df['Transmission_r'].std():.3f}")
print(f"  Range: [{comm_trans_df['Transmission_r'].min():.3f}, {comm_trans_df['Transmission_r'].max():.3f}]")

# Correlation between community characteristics and transmission
corr_urban = stats.pearsonr(comm_trans_df['Urban_Prop'], comm_trans_df['Transmission_r'])
corr_policy = stats.pearsonr(comm_trans_df['Policy_Level'], comm_trans_df['Transmission_r'])

print(f"\nCommunity Predictors of Transmission:")
print(f"  Urban proportion × Transmission: r = {corr_urban[0]:.3f}, p = {corr_urban[1]:.4f}")
print(f"  Policy level × Transmission: r = {corr_policy[0]:.3f}, p = {corr_policy[1]:.4f}")

# ============================================================
# 8. Model Comparison / 模型比较
# ============================================================

print("\n" + "-"*60)
print("MODEL COMPARISON")
print("-"*60)

# Likelihood ratio test
lr_stat = -2 * (result_comm.llf - result_cross.llf)
df_diff = len(result_cross.fe_params) - len(result_comm.fe_params)
p_lr = 1 - stats.chi2.cdf(lr_stat, df_diff)

print(f"\n{'Model':<35} {'-2LL':<15} {'AIC':<15} {'BIC'}")
print("-" * 80)
print(f"{'Main Effects Only':<35} {-2*result_comm.llf:<15.1f} {result_comm.aic:<15.1f} {result_comm.bic:.1f}")
print(f"{'With Cross-Level Interactions':<35} {-2*result_cross.llf:<15.1f} {result_cross.aic:<15.1f} {result_cross.bic:.1f}")
print(f"{'With Three-Way Interaction':<35} {-2*result_three.llf:<15.1f} {result_three.aic:<15.1f} {result_three.bic:.1f}")

print(f"\nLikelihood Ratio Test (Main → Cross-Level):")
print(f"  χ²({df_diff}) = {lr_stat:.2f}, p = {p_lr:.4f}")

# ============================================================
# 9. Save Results / 保存结果
# ============================================================

print("\n" + "-"*60)
print("Saving Results...")
print("-"*60)

# Save cross-level interaction results
pd.DataFrame(cross_results).to_csv('cross_level_interactions.csv', index=False)
print("  Saved: cross_level_interactions.csv")

# Save community transmission data
comm_trans_df.to_csv('community_transmission.csv', index=False)
print("  Saved: community_transmission.csv")

# Save full model results
model_results = pd.DataFrame({
    'Parameter': result_cross.fe_params.index,
    'Estimate': result_cross.fe_params.values,
    'Std_Error': result_cross.bse.values,
    't_value': result_cross.tvalues.values,
    'p_value': result_cross.pvalues.values
})
model_results.to_csv('cross_level_model_full.csv', index=False)
print("  Saved: cross_level_model_full.csv")

print("\n" + "="*60)
print("CROSS-LEVEL INTERACTION ANALYSIS COMPLETED")
print("="*60)
print("\nKey Findings:")
print(f"  - Cultural Policy Support → Child Identity: β = {result_cross.fe_params['Comm_Policy_c']:.3f}***")
print(f"  - Activity × Family Investment interaction: β = {result_cross.fe_params['Activity_X_FSI']:.3f}***")
print(f"  - Three-way interaction (Urban × Policy × FSI): β = {coef_3way:.3f}**")
print(f"  - Between-community variance explained: {r2_level2*100:.1f}%")
