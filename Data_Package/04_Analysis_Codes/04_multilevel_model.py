"""
Module 4: Multilevel Linear Model Analysis
模块4：多层线性模型分析

This script performs hierarchical linear modeling (HLM) for intergenerational transmission.
此脚本执行代际传递的多层线性模型分析。

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
print("MODULE 4: MULTILEVEL LINEAR MODEL ANALYSIS")
print("="*60)

df = pd.read_excel('National_Opera_Cultural_Identity_Dataset.xlsx', sheet_name='Raw_Data')
print(f"\nData loaded: {len(df)} families")
print(f"Communities: {df['Community_ID'].nunique()}")
print(f"Cities: {df['City_ID'].nunique()}")

# ============================================================
# 2. Variable Preparation / 变量准备
# ============================================================

print("\n[1] Preparing Variables...")

# Grand-mean centering for Level-1 predictors
df['Parent_ID_c'] = df['Parent_Identity_Overall'] - df['Parent_Identity_Overall'].mean()
df['FSI_c'] = df['Family_Socialization'] - df['Family_Socialization'].mean()
df['SES_c'] = df['Family_SES'] - df['Family_SES'].mean()
df['Parent_Edu_c'] = df['Parent_Education'] - df['Parent_Education'].mean()
df['Child_Age_c'] = df['Child_Age'] - df['Child_Age'].mean()

# Create interaction term
df['Parent_X_FSI'] = df['Parent_ID_c'] * df['FSI_c']
df['Urban_X_Parent'] = df['Urban_Rural'] * df['Parent_ID_c']
df['Urban_X_FSI'] = df['Urban_Rural'] * df['FSI_c']

# Child gender dummy (1=Female, 0=Male)
df['Child_Female'] = (df['Child_Gender'] == 2).astype(int)

print("  Variables centered and interaction terms created")

# ============================================================
# 3. Model 1: Null Model (Unconditional) / 零模型
# ============================================================

print("\n" + "-"*60)
print("MODEL 1: NULL MODEL (Unconditional)")
print("-"*60)

# Fit null model
model1 = smf.mixedlm("Child_Identity_Overall ~ 1", 
                      df, 
                      groups=df["Community_ID"])
result1 = model1.fit(method='lbfgs')

# Calculate ICC
var_community = result1.cov_re.iloc[0, 0]
var_residual = result1.scale
icc_community = var_community / (var_community + var_residual)

print(f"\nFixed Effects:")
print(f"  Intercept: {result1.fe_params['Intercept']:.3f} (SE={result1.bse['Intercept']:.3f})")

print(f"\nRandom Effects Variance:")
print(f"  Community Level (τ00): {var_community:.3f}")
print(f"  Residual (σ²): {var_residual:.3f}")

print(f"\nIntraclass Correlation Coefficient:")
print(f"  ICC (Community): {icc_community:.3f}")
print(f"  Interpretation: {icc_community*100:.1f}% of variance is between communities")

print(f"\nModel Fit:")
print(f"  -2 Log Likelihood: {-2 * result1.llf:.1f}")
print(f"  AIC: {result1.aic:.1f}")
print(f"  BIC: {result1.bic:.1f}")

# ============================================================
# 4. Model 2: Main Effects Model / 主效应模型
# ============================================================

print("\n" + "-"*60)
print("MODEL 2: MAIN EFFECTS MODEL")
print("-"*60)

formula2 = """Child_Identity_Overall ~ Parent_ID_c + FSI_c + SES_c + 
              Parent_Edu_c + Child_Age_c + Child_Female"""

model2 = smf.mixedlm(formula2, df, groups=df["Community_ID"])
result2 = model2.fit(method='lbfgs')

print(f"\nFixed Effects:")
print(f"{'Parameter':<20} {'Estimate':<12} {'Std.Err':<12} {'z-value':<12} {'p-value'}")
print("-" * 68)

for param in result2.fe_params.index:
    est = result2.fe_params[param]
    se = result2.bse[param]
    z = result2.tvalues[param]
    p = result2.pvalues[param]
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
    print(f"{param:<20} {est:>10.3f}   {se:>10.3f}   {z:>10.2f}   {p:>8.4f} {sig}")

print(f"\nRandom Effects Variance:")
var_comm2 = result2.cov_re.iloc[0, 0]
var_res2 = result2.scale
print(f"  Community Level (τ00): {var_comm2:.3f}")
print(f"  Residual (σ²): {var_res2:.3f}")

# Variance explained
r2_level1 = 1 - (var_res2 / var_residual)
r2_level2 = 1 - (var_comm2 / var_community)
print(f"\nVariance Explained:")
print(f"  R² (Level 1): {r2_level1:.3f}")
print(f"  R² (Level 2): {r2_level2:.3f}")

print(f"\nModel Fit:")
print(f"  -2 Log Likelihood: {-2 * result2.llf:.1f}")
print(f"  AIC: {result2.aic:.1f}")
print(f"  BIC: {result2.bic:.1f}")

# ============================================================
# 5. Model 3: Moderation Model / 调节效应模型
# ============================================================

print("\n" + "-"*60)
print("MODEL 3: MODERATION MODEL")
print("-"*60)

formula3 = """Child_Identity_Overall ~ Parent_ID_c + FSI_c + Parent_X_FSI + 
              SES_c + Parent_Edu_c + Child_Age_c + Child_Female"""

model3 = smf.mixedlm(formula3, df, groups=df["Community_ID"])
result3 = model3.fit(method='lbfgs')

print(f"\nFixed Effects:")
print(f"{'Parameter':<20} {'Estimate':<12} {'Std.Err':<12} {'z-value':<12} {'p-value'}")
print("-" * 68)

for param in result3.fe_params.index:
    est = result3.fe_params[param]
    se = result3.bse[param]
    z = result3.tvalues[param]
    p = result3.pvalues[param]
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
    print(f"{param:<20} {est:>10.3f}   {se:>10.3f}   {z:>10.2f}   {p:>8.4f} {sig}")

print(f"\nRandom Effects Variance:")
var_comm3 = result3.cov_re.iloc[0, 0]
var_res3 = result3.scale
print(f"  Community Level (τ00): {var_comm3:.3f}")
print(f"  Residual (σ²): {var_res3:.3f}")

# ============================================================
# 6. Model 4: Full Model with Urban-Rural / 完整模型（含城乡）
# ============================================================

print("\n" + "-"*60)
print("MODEL 4: FULL MODEL (with Urban-Rural)")
print("-"*60)

formula4 = """Child_Identity_Overall ~ Parent_ID_c + FSI_c + Parent_X_FSI + 
              Urban_Rural + Urban_X_Parent + Urban_X_FSI +
              SES_c + Parent_Edu_c + Child_Age_c + Child_Female"""

model4 = smf.mixedlm(formula4, df, groups=df["Community_ID"])
result4 = model4.fit(method='lbfgs')

print(f"\nFixed Effects:")
print(f"{'Parameter':<20} {'Estimate':<12} {'Std.Err':<12} {'z-value':<12} {'p-value'}")
print("-" * 68)

for param in result4.fe_params.index:
    est = result4.fe_params[param]
    se = result4.bse[param]
    z = result4.tvalues[param]
    p = result4.pvalues[param]
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
    print(f"{param:<20} {est:>10.3f}   {se:>10.3f}   {z:>10.2f}   {p:>8.4f} {sig}")

print(f"\nRandom Effects Variance:")
var_comm4 = result4.cov_re.iloc[0, 0]
var_res4 = result4.scale
print(f"  Community Level (τ00): {var_comm4:.3f}")
print(f"  Residual (σ²): {var_res4:.3f}")

print(f"\nModel Fit:")
print(f"  -2 Log Likelihood: {-2 * result4.llf:.1f}")
print(f"  AIC: {result4.aic:.1f}")
print(f"  BIC: {result4.bic:.1f}")

# ============================================================
# 7. Model Comparison / 模型比较
# ============================================================

print("\n" + "-"*60)
print("MODEL COMPARISON")
print("-"*60)

print(f"\n{'Model':<25} {'-2LL':<15} {'AIC':<15} {'BIC':<15}")
print("-" * 70)
print(f"{'Model 1 (Null)':<25} {-2*result1.llf:<15.1f} {result1.aic:<15.1f} {result1.bic:<15.1f}")
print(f"{'Model 2 (Main Effects)':<25} {-2*result2.llf:<15.1f} {result2.aic:<15.1f} {result2.bic:<15.1f}")
print(f"{'Model 3 (Moderation)':<25} {-2*result3.llf:<15.1f} {result3.aic:<15.1f} {result3.bic:<15.1f}")
print(f"{'Model 4 (Full)':<25} {-2*result4.llf:<15.1f} {result4.aic:<15.1f} {result4.bic:<15.1f}")

# Likelihood ratio tests
print("\nLikelihood Ratio Tests:")
lr_12 = -2 * (result1.llf - result2.llf)
df_diff_12 = len(result2.fe_params) - len(result1.fe_params)
p_12 = 1 - stats.chi2.cdf(lr_12, df_diff_12)
print(f"  Model 1 vs Model 2: χ²({df_diff_12}) = {lr_12:.2f}, p = {p_12:.4f}")

lr_23 = -2 * (result2.llf - result3.llf)
df_diff_23 = len(result3.fe_params) - len(result2.fe_params)
p_23 = 1 - stats.chi2.cdf(lr_23, df_diff_23)
print(f"  Model 2 vs Model 3: χ²({df_diff_23}) = {lr_23:.2f}, p = {p_23:.4f}")

lr_34 = -2 * (result3.llf - result4.llf)
df_diff_34 = len(result4.fe_params) - len(result3.fe_params)
p_34 = 1 - stats.chi2.cdf(lr_34, df_diff_34)
print(f"  Model 3 vs Model 4: χ²({df_diff_34}) = {lr_34:.2f}, p = {p_34:.4f}")

# ============================================================
# 8. Effect Size Calculation / 效应量计算
# ============================================================

print("\n" + "-"*60)
print("EFFECT SIZE (Cohen's f²)")
print("-"*60)

# f² = (R²_full - R²_reduced) / (1 - R²_full)
r2_model2 = 1 - (var_res2 / var_residual)
r2_model3 = 1 - (var_res3 / var_residual)
r2_model4 = 1 - (var_res4 / var_residual)

f2_moderation = (r2_model3 - r2_model2) / (1 - r2_model3)
f2_urban_rural = (r2_model4 - r2_model3) / (1 - r2_model4)

print(f"\nEffect of Moderation (FSI): f² = {f2_moderation:.4f}")
print(f"Effect of Urban-Rural: f² = {f2_urban_rural:.4f}")
print("\nInterpretation: 0.02=small, 0.15=medium, 0.35=large")

# ============================================================
# 9. Save Results / 保存结果
# ============================================================

print("\n" + "-"*60)
print("Saving Results...")
print("-"*60)

# Save Model 4 results
model4_results = pd.DataFrame({
    'Parameter': result4.fe_params.index,
    'Estimate': result4.fe_params.values,
    'Std_Error': result4.bse.values,
    'z_value': result4.tvalues.values,
    'p_value': result4.pvalues.values
})
model4_results.to_csv('mlm_model4_results.csv', index=False)
print("  Saved: mlm_model4_results.csv")

# Save model comparison
model_comparison = pd.DataFrame({
    'Model': ['Null', 'Main Effects', 'Moderation', 'Full'],
    'neg2LL': [-2*result1.llf, -2*result2.llf, -2*result3.llf, -2*result4.llf],
    'AIC': [result1.aic, result2.aic, result3.aic, result4.aic],
    'BIC': [result1.bic, result2.bic, result3.bic, result4.bic]
})
model_comparison.to_csv('mlm_model_comparison.csv', index=False)
print("  Saved: mlm_model_comparison.csv")

print("\n" + "="*60)
print("MULTILEVEL LINEAR MODEL ANALYSIS COMPLETED")
print("="*60)
print("\nKey Findings:")
print(f"  - Parent Identity → Child Identity: β = {result4.fe_params['Parent_ID_c']:.3f}***")
print(f"  - Family Socialization moderates transmission: β = {result4.fe_params['Parent_X_FSI']:.3f}***")
print(f"  - Urban-Rural difference in transmission: β = {result4.fe_params['Urban_X_Parent']:.3f}**")
print(f"  - ICC (Community): {icc_community:.3f}")
