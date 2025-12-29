"""
Module 1: Data Preprocessing
模块1：数据预处理

This script loads raw data and creates processed variables for analysis.
此脚本加载原始数据并创建用于分析的处理变量。

Author: Research Team
Date: November 2024
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. Load Raw Data / 加载原始数据
# ============================================================

print("="*60)
print("MODULE 1: DATA PREPROCESSING")
print("="*60)

# Load data
df = pd.read_excel('National_Opera_Cultural_Identity_Dataset.xlsx', sheet_name='Raw_Data')
print(f"\n[1] Raw data loaded: {df.shape[0]} families, {df.shape[1]} variables")

# ============================================================
# 2. Data Cleaning / 数据清洗
# ============================================================

print("\n[2] Data Cleaning...")

# Check for missing values
missing_counts = df.isnull().sum()
missing_vars = missing_counts[missing_counts > 0]
if len(missing_vars) > 0:
    print(f"   Variables with missing values: {len(missing_vars)}")
    for var, count in missing_vars.items():
        print(f"   - {var}: {count} ({count/len(df)*100:.2f}%)")
else:
    print("   No missing values detected")

# Check for outliers using IQR method
numeric_cols = df.select_dtypes(include=[np.number]).columns
outlier_report = []
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
    if outliers > 0:
        outlier_report.append((col, outliers, outliers/len(df)*100))

print(f"\n   Variables with outliers (IQR method): {len(outlier_report)}")
if len(outlier_report) > 0:
    for var, count, pct in outlier_report[:5]:
        print(f"   - {var}: {count} ({pct:.2f}%)")

# ============================================================
# 3. Variable Transformation / 变量转换
# ============================================================

print("\n[3] Variable Transformation...")

# Create processed dataframe
df_processed = df.copy()

# 3.1 Standardize continuous variables (Z-scores)
vars_to_standardize = [
    'Parent_Cog_Mean', 'Parent_Aff_Mean', 'Parent_Beh_Mean', 'Parent_Identity_Overall',
    'Child_Cog_Mean', 'Child_Aff_Mean', 'Child_Beh_Mean', 'Child_Identity_Overall',
    'Family_Socialization', 'Family_SES',
    'Comm_Cultural_Policy', 'Comm_Cultural_Facility', 'Comm_Cultural_Activity'
]

for var in vars_to_standardize:
    df_processed[f'{var}_z'] = stats.zscore(df_processed[var])
print(f"   Created {len(vars_to_standardize)} standardized variables (Z-scores)")

# 3.2 Create centered variables for moderation analysis
df_processed['Parent_Identity_c'] = df_processed['Parent_Identity_Overall'] - df_processed['Parent_Identity_Overall'].mean()
df_processed['Family_Socialization_c'] = df_processed['Family_Socialization'] - df_processed['Family_Socialization'].mean()
df_processed['Parent_Cog_c'] = df_processed['Parent_Cog_Mean'] - df_processed['Parent_Cog_Mean'].mean()
df_processed['Parent_Aff_c'] = df_processed['Parent_Aff_Mean'] - df_processed['Parent_Aff_Mean'].mean()
df_processed['Parent_Beh_c'] = df_processed['Parent_Beh_Mean'] - df_processed['Parent_Beh_Mean'].mean()
print("   Created mean-centered variables for moderation analysis")

# 3.3 Create interaction terms
df_processed['Parent_X_Socialization'] = df_processed['Parent_Identity_c'] * df_processed['Family_Socialization_c']
df_processed['Urban_X_Parent'] = df_processed['Urban_Rural'] * df_processed['Parent_Identity_c']
df_processed['Urban_X_Socialization'] = df_processed['Urban_Rural'] * df_processed['Family_Socialization_c']
df_processed['Three_Way_Interaction'] = df_processed['Urban_Rural'] * df_processed['Parent_Identity_c'] * df_processed['Family_Socialization_c']
print("   Created interaction terms")

# 3.4 Create age group variable
def assign_age_group(age):
    if age < 16:
        return '12-15'
    elif age < 19:
        return '16-18'
    elif age < 23:
        return '19-22'
    else:
        return '23-25'

df_processed['Child_Age_Group'] = df_processed['Child_Age'].apply(assign_age_group)
print("   Created age group variable")

# 3.5 Create gender combination variable
def get_gender_combo(row):
    if row['Parent_Gender'] == 2 and row['Child_Gender'] == 2:
        return 'Mother-Daughter'
    elif row['Parent_Gender'] == 1 and row['Child_Gender'] == 2:
        return 'Father-Daughter'
    elif row['Parent_Gender'] == 2 and row['Child_Gender'] == 1:
        return 'Mother-Son'
    else:
        return 'Father-Son'

df_processed['Gender_Combination'] = df_processed.apply(get_gender_combo, axis=1)
print("   Created gender combination variable")

# 3.6 Create community-level aggregated variables
comm_vars = df_processed.groupby('Community_ID').agg({
    'Parent_Identity_Overall': 'mean',
    'Family_Socialization': 'mean',
    'Urban_Rural': 'mean'
}).reset_index()
comm_vars.columns = ['Community_ID', 'Comm_Parent_Identity_Mean', 'Comm_Socialization_Mean', 'Comm_Urban_Prop']
df_processed = df_processed.merge(comm_vars, on='Community_ID', how='left')
print("   Created community-level aggregated variables")

# 3.7 Create dummy variables
df_processed['Region_Central'] = (df_processed['Region'] == 'Central').astype(int)
df_processed['Region_Western'] = (df_processed['Region'] == 'Western').astype(int)
print("   Created region dummy variables")

# ============================================================
# 4. Create Analysis Subsets / 创建分析子集
# ============================================================

print("\n[4] Creating Analysis Subsets...")

# Urban subset
df_urban = df_processed[df_processed['Urban_Rural'] == 1].copy()
print(f"   Urban subset: {len(df_urban)} families")

# Rural subset
df_rural = df_processed[df_processed['Urban_Rural'] == 0].copy()
print(f"   Rural subset: {len(df_rural)} families")

# Age group subsets
age_subsets = {}
for group in ['12-15', '16-18', '19-22', '23-25']:
    age_subsets[group] = df_processed[df_processed['Child_Age_Group'] == group].copy()
    print(f"   Age group {group}: {len(age_subsets[group])} families")

# Gender combination subsets
gender_subsets = {}
for combo in ['Mother-Daughter', 'Father-Daughter', 'Mother-Son', 'Father-Son']:
    gender_subsets[combo] = df_processed[df_processed['Gender_Combination'] == combo].copy()
    print(f"   {combo}: {len(gender_subsets[combo])} families")

# ============================================================
# 5. Save Processed Data / 保存处理后数据
# ============================================================

print("\n[5] Saving Processed Data...")

# Save full processed dataset
df_processed.to_csv('processed_data_full.csv', index=False)
print("   Saved: processed_data_full.csv")

# Save urban and rural subsets
df_urban.to_csv('processed_data_urban.csv', index=False)
df_rural.to_csv('processed_data_rural.csv', index=False)
print("   Saved: processed_data_urban.csv, processed_data_rural.csv")

# Save variable list
var_list = pd.DataFrame({
    'Variable': df_processed.columns,
    'Type': [str(df_processed[col].dtype) for col in df_processed.columns],
    'Non_Missing': [df_processed[col].notna().sum() for col in df_processed.columns],
    'Mean': [df_processed[col].mean() if df_processed[col].dtype in ['int64', 'float64'] else np.nan for col in df_processed.columns],
    'SD': [df_processed[col].std() if df_processed[col].dtype in ['int64', 'float64'] else np.nan for col in df_processed.columns]
})
var_list.to_csv('variable_list.csv', index=False)
print("   Saved: variable_list.csv")

# ============================================================
# 6. Summary Statistics / 汇总统计
# ============================================================

print("\n[6] Summary Statistics...")
print("\n   Sample Characteristics:")
print(f"   - Total families: {len(df_processed)}")
print(f"   - Urban: {(df_processed['Urban_Rural']==1).sum()} ({(df_processed['Urban_Rural']==1).mean()*100:.1f}%)")
print(f"   - Rural: {(df_processed['Urban_Rural']==0).sum()} ({(df_processed['Urban_Rural']==0).mean()*100:.1f}%)")
print(f"   - Cities: {df_processed['City_ID'].nunique()}")
print(f"   - Communities: {df_processed['Community_ID'].nunique()}")
print(f"   - Child age: M={df_processed['Child_Age'].mean():.1f}, SD={df_processed['Child_Age'].std():.1f}")
print(f"   - Parent age: M={df_processed['Parent_Age'].mean():.1f}, SD={df_processed['Parent_Age'].std():.1f}")

print("\n" + "="*60)
print("DATA PREPROCESSING COMPLETED")
print("="*60)
print("\nOutput files:")
print("  - processed_data_full.csv")
print("  - processed_data_urban.csv")
print("  - processed_data_rural.csv")
print("  - variable_list.csv")
