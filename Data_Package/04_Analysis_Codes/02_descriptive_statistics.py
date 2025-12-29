"""
Module 2: Descriptive Statistics
模块2：描述性统计分析

This script generates descriptive statistics for all main variables.
此脚本生成所有主要变量的描述性统计。

Author: Research Team
Date: November 2024
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. Load Data / 加载数据
# ============================================================

print("="*60)
print("MODULE 2: DESCRIPTIVE STATISTICS")
print("="*60)

df = pd.read_excel('National_Opera_Cultural_Identity_Dataset.xlsx', sheet_name='Raw_Data')
print(f"\nData loaded: {len(df)} families")

# ============================================================
# 2. Main Variable Descriptive Statistics / 主要变量描述统计
# ============================================================

print("\n" + "-"*60)
print("TABLE 1: Descriptive Statistics of Main Variables (N=5,694)")
print("-"*60)

main_vars = [
    ('Parent_Cog_Mean', 'Cognitive Identity (Parent)'),
    ('Parent_Aff_Mean', 'Affective Identity (Parent)'),
    ('Parent_Beh_Mean', 'Behavioral Identity (Parent)'),
    ('Parent_Identity_Overall', 'Overall Cultural Identity (Parent)'),
    ('Child_Cog_Mean', 'Cognitive Identity (Child)'),
    ('Child_Aff_Mean', 'Affective Identity (Child)'),
    ('Child_Beh_Mean', 'Behavioral Identity (Child)'),
    ('Child_Identity_Overall', 'Overall Cultural Identity (Child)'),
    ('Family_Socialization', 'Family Socialization Investment')
]

# Create combined data for parents and children
cog_all = pd.concat([df['Parent_Cog_Mean'], df['Child_Cog_Mean']])
aff_all = pd.concat([df['Parent_Aff_Mean'], df['Child_Aff_Mean']])
beh_all = pd.concat([df['Parent_Beh_Mean'], df['Child_Beh_Mean']])
overall_all = pd.concat([df['Parent_Identity_Overall'], df['Child_Identity_Overall']])

table1_data = []
combined_vars = [
    (cog_all, 'Cognitive Identity'),
    (aff_all, 'Affective Identity'),
    (beh_all, 'Behavioral Identity'),
    (overall_all, 'Overall Cultural Identity'),
    (df['Family_Socialization'], 'Family Socialization Investment'),
    (df['Parent_Identity_Overall'], "Parental Cultural Identity"),
    (df['Child_Identity_Overall'], "Children's Cultural Identity")
]

for data, name in combined_vars:
    table1_data.append({
        'Variable': name,
        'Mean': round(data.mean(), 2),
        'Std Dev': round(data.std(), 2),
        'Min': round(data.min(), 2),
        'Max': round(data.max(), 2),
        'Skewness': round(data.skew(), 2),
        'Kurtosis': round(data.kurtosis(), 2)
    })

table1 = pd.DataFrame(table1_data)
print(table1.to_string(index=False))

# ============================================================
# 3. Urban-Rural Comparison / 城乡比较
# ============================================================

print("\n" + "-"*60)
print("TABLE 4: Urban-Rural Sample Comparison")
print("-"*60)

urban = df[df['Urban_Rural'] == 1]
rural = df[df['Urban_Rural'] == 0]

comparison_vars = [
    ('Parent_Identity_Overall', 'Parental Cultural Identity'),
    ('Child_Identity_Overall', "Children's Cultural Identity"),
    ('Family_Socialization', 'Family Socialization Investment'),
    ('Parent_Cog_Mean', 'Parent Cognitive Identity'),
    ('Parent_Aff_Mean', 'Parent Affective Identity'),
    ('Parent_Beh_Mean', 'Parent Behavioral Identity'),
    ('Child_Cog_Mean', 'Child Cognitive Identity'),
    ('Child_Aff_Mean', 'Child Affective Identity'),
    ('Child_Beh_Mean', 'Child Behavioral Identity')
]

print(f"\n{'Variable':<35} {'Urban (n='+str(len(urban))+')':<20} {'Rural (n='+str(len(rural))+')':<20} {'t-value':<12} {'p-value':<10}")
print("-" * 97)

for var, label in comparison_vars:
    u_mean = urban[var].mean()
    u_sd = urban[var].std()
    r_mean = rural[var].mean()
    r_sd = rural[var].std()
    
    t_stat, p_val = stats.ttest_ind(urban[var], rural[var])
    
    sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else ''))
    
    print(f"{label:<35} {u_mean:.2f} ({u_sd:.2f}){'':<8} {r_mean:.2f} ({r_sd:.2f}){'':<8} {t_stat:.2f}{sig:<6} {p_val:.4f}")

print("\nNote: ***p<0.001, **p<0.01, *p<0.05")

# ============================================================
# 4. Gender and Age Group Statistics / 性别与年龄组统计
# ============================================================

print("\n" + "-"*60)
print("TABLE 7: Statistics by Gender Combination and Age Group")
print("-"*60)

# Gender combination
def get_gender_combo(row):
    if row['Parent_Gender'] == 2 and row['Child_Gender'] == 2:
        return 'Mother-Daughter'
    elif row['Parent_Gender'] == 1 and row['Child_Gender'] == 2:
        return 'Father-Daughter'
    elif row['Parent_Gender'] == 2 and row['Child_Gender'] == 1:
        return 'Mother-Son'
    else:
        return 'Father-Son'

df['Gender_Combination'] = df.apply(get_gender_combo, axis=1)

print("\nGender Combination:")
print(f"{'Combination':<20} {'N':<10} {'Parent M(SD)':<18} {'Child M(SD)':<18} {'Correlation':<12}")
print("-" * 78)

for combo in ['Mother-Daughter', 'Father-Daughter', 'Mother-Son', 'Father-Son']:
    subset = df[df['Gender_Combination'] == combo]
    n = len(subset)
    p_mean = subset['Parent_Identity_Overall'].mean()
    p_sd = subset['Parent_Identity_Overall'].std()
    c_mean = subset['Child_Identity_Overall'].mean()
    c_sd = subset['Child_Identity_Overall'].std()
    corr = subset['Parent_Identity_Overall'].corr(subset['Child_Identity_Overall'])
    print(f"{combo:<20} {n:<10} {p_mean:.2f} ({p_sd:.2f}){'':<6} {c_mean:.2f} ({c_sd:.2f}){'':<6} {corr:.3f}")

# Age groups
def assign_age_group(age):
    if age < 16:
        return '12-15'
    elif age < 19:
        return '16-18'
    elif age < 23:
        return '19-22'
    else:
        return '23-25'

df['Age_Group'] = df['Child_Age'].apply(assign_age_group)

print("\nAge Groups:")
print(f"{'Age Group':<15} {'N':<10} {'Parent M(SD)':<18} {'Child M(SD)':<18} {'Correlation':<12}")
print("-" * 73)

for group in ['12-15', '16-18', '19-22', '23-25']:
    subset = df[df['Age_Group'] == group]
    n = len(subset)
    p_mean = subset['Parent_Identity_Overall'].mean()
    p_sd = subset['Parent_Identity_Overall'].std()
    c_mean = subset['Child_Identity_Overall'].mean()
    c_sd = subset['Child_Identity_Overall'].std()
    corr = subset['Parent_Identity_Overall'].corr(subset['Child_Identity_Overall'])
    print(f"{group:<15} {n:<10} {p_mean:.2f} ({p_sd:.2f}){'':<6} {c_mean:.2f} ({c_sd:.2f}){'':<6} {corr:.3f}")

# ============================================================
# 5. Regional Distribution / 区域分布
# ============================================================

print("\n" + "-"*60)
print("Regional Distribution")
print("-"*60)

print(f"\n{'Region':<15} {'N':<10} {'%':<10} {'Urban %':<12} {'Parent ID M':<15} {'Child ID M':<15}")
print("-" * 77)

for region in ['Eastern', 'Central', 'Western']:
    subset = df[df['Region'] == region]
    n = len(subset)
    pct = n / len(df) * 100
    urban_pct = subset['Urban_Rural'].mean() * 100
    p_mean = subset['Parent_Identity_Overall'].mean()
    c_mean = subset['Child_Identity_Overall'].mean()
    print(f"{region:<15} {n:<10} {pct:.1f}%{'':<6} {urban_pct:.1f}%{'':<6} {p_mean:.2f}{'':<10} {c_mean:.2f}")

# ============================================================
# 6. Correlation Matrix / 相关矩阵
# ============================================================

print("\n" + "-"*60)
print("Correlation Matrix of Main Variables")
print("-"*60)

corr_vars = ['Parent_Cog_Mean', 'Parent_Aff_Mean', 'Parent_Beh_Mean', 
             'Child_Cog_Mean', 'Child_Aff_Mean', 'Child_Beh_Mean',
             'Family_Socialization']
corr_labels = ['P_Cog', 'P_Aff', 'P_Beh', 'C_Cog', 'C_Aff', 'C_Beh', 'FSI']

corr_matrix = df[corr_vars].corr()
corr_matrix.columns = corr_labels
corr_matrix.index = corr_labels

print("\n" + corr_matrix.round(3).to_string())

# ============================================================
# 7. Intraclass Correlation Coefficients / 组内相关系数
# ============================================================

print("\n" + "-"*60)
print("Intraclass Correlation Coefficients (ICC)")
print("-"*60)

# ICC for family level
family_groups = df.groupby('Community_ID')['Child_Identity_Overall']
between_var = family_groups.mean().var()
within_var = family_groups.apply(lambda x: x.var()).mean()
icc_community = between_var / (between_var + within_var)

print(f"\nChild Cultural Identity:")
print(f"  ICC (Community level): {icc_community:.3f}")
print(f"  Interpretation: {icc_community*100:.1f}% of variance is between communities")

# ============================================================
# 8. Save Results / 保存结果
# ============================================================

print("\n" + "-"*60)
print("Saving Results...")
print("-"*60)

# Save Table 1
table1.to_csv('descriptive_statistics_table1.csv', index=False)
print("  Saved: descriptive_statistics_table1.csv")

# Save correlation matrix
corr_matrix.to_csv('correlation_matrix.csv')
print("  Saved: correlation_matrix.csv")

# Save urban-rural comparison
urban_rural_comp = []
for var, label in comparison_vars:
    u_mean, u_sd = urban[var].mean(), urban[var].std()
    r_mean, r_sd = rural[var].mean(), rural[var].std()
    t_stat, p_val = stats.ttest_ind(urban[var], rural[var])
    urban_rural_comp.append({
        'Variable': label,
        'Urban_Mean': round(u_mean, 2),
        'Urban_SD': round(u_sd, 2),
        'Rural_Mean': round(r_mean, 2),
        'Rural_SD': round(r_sd, 2),
        't_value': round(t_stat, 2),
        'p_value': round(p_val, 4)
    })
pd.DataFrame(urban_rural_comp).to_csv('urban_rural_comparison.csv', index=False)
print("  Saved: urban_rural_comparison.csv")

print("\n" + "="*60)
print("DESCRIPTIVE STATISTICS COMPLETED")
print("="*60)
