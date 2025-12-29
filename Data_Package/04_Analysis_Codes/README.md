# National Opera Cultural Identity Intergenerational Transmission Study
# 民族歌剧文化认同代际传递研究

## Data and Analysis Repository
## 数据与分析资源库

---

## Overview / 概述

This repository contains all data files, survey instruments, analysis codes, and supporting materials for the study "Intergenerational Transmission Mechanisms of National Opera Cultural Identity: The Moderating Effect of Family Socialization Investment and Multilevel Analysis of Urban-Rural Differences".

本资源库包含研究"民族歌剧文化认同代际传递机制：家庭社会化投入的调节效应及城乡差异多层分析"的所有数据文件、调查工具、分析代码和支持材料。

---

## File Structure / 文件结构

```
├── Data Files / 数据文件
│   ├── National_Opera_Cultural_Identity_Dataset.xlsx (Raw data / 原始数据)
│   │   ├── Raw_Data (2,847 families × 63 variables)
│   │   ├── Codebook (Variable definitions)
│   │   ├── Scale_Items (Questionnaire items EN/CN)
│   │   └── Descriptive_Stats (Summary statistics)
│   └── processed_data_full.csv (Processed data / 处理后数据)
│
├── Survey Instruments / 调查工具
│   └── Survey_Instrument_and_Protocol.md
│       ├── Part I: Survey Questionnaire (Full scales)
│       ├── Part II: Interview Guide
│       ├── Part III: Data Collection Procedure
│       └── Part IV: Ethical Considerations
│
├── Analysis Codes / 分析代码
│   ├── 01_data_preprocessing.py
│   ├── 02_descriptive_statistics.py
│   ├── 03_reliability_validity.py
│   ├── 04_multilevel_model.py
│   ├── 05_structural_equation_model.py
│   ├── 06_moderation_urban_rural.py
│   └── 07_cross_level_interaction.py
│
└── Supporting Files / 支持文件
    └── README.md (This file)
```

---

## Data Description / 数据说明

### Sample / 样本
- **Total families**: 2,847
- **Communities**: 126
- **Cities**: 18 prefecture-level cities
- **Regions**: Eastern, Central, Western China

### Key Variables / 主要变量

| Variable Category | Variables | Scale |
|-------------------|-----------|-------|
| Cultural Identity (Parent) | Cognitive, Affective, Behavioral | 1-7 Likert |
| Cultural Identity (Child) | Cognitive, Affective, Behavioral | 1-7 Likert |
| Family Socialization Investment | 4 items | 1-7 Likert |
| Community Cultural Ecology | Policy, Facility, Activity | 1-7 Likert |
| Demographics | Age, Gender, Education, SES | Various |

---

## Analysis Code Descriptions / 分析代码说明

### Module 1: Data Preprocessing (01_data_preprocessing.py)
- Load and clean raw data
- Create centered and standardized variables
- Generate interaction terms
- Create analysis subsets (urban/rural, age groups, gender combinations)

### Module 2: Descriptive Statistics (02_descriptive_statistics.py)
- Generate Table 1: Main variable descriptives
- Urban-rural mean comparisons with t-tests
- Regional and demographic distributions
- Correlation matrices

### Module 3: Reliability and Validity (03_reliability_validity.py)
- Cronbach's alpha for all scales
- Item-total correlations
- Average Variance Extracted (AVE)
- Composite Reliability (CR)
- Measurement invariance across groups

### Module 4: Multilevel Linear Model (04_multilevel_model.py)
- Model 1: Null model (ICC calculation)
- Model 2: Main effects model
- Model 3: Moderation model
- Model 4: Full model with urban-rural
- Model comparison and effect sizes

### Module 5: Structural Equation Model (05_structural_equation_model.py)
- Path analysis by dimension
- Multigroup analysis (urban vs rural)
- Mediation effect decomposition
- Model fit indices

### Module 6: Moderation Analysis (06_moderation_urban_rural.py)
- Simple slopes analysis
- Dimensional moderation effects
- Urban-rural comparison
- Gender and age group analysis

### Module 7: Cross-Level Interaction (07_cross_level_interaction.py)
- Community-level main effects
- Cross-level interaction effects
- Three-way interaction analysis
- Variance decomposition
- Community heterogeneity analysis

---

## How to Run / 运行方法

### Requirements / 环境要求
```python
# Python 3.8+
pandas>=1.3.0
numpy>=1.20.0
scipy>=1.7.0
statsmodels>=0.13.0
openpyxl>=3.0.0
```

### Execution Order / 执行顺序
```bash
# 1. First, ensure data file is in working directory
# 2. Run modules in order:
python 01_data_preprocessing.py
python 02_descriptive_statistics.py
python 03_reliability_validity.py
python 04_multilevel_model.py
python 05_structural_equation_model.py
python 06_moderation_urban_rural.py
python 07_cross_level_interaction.py
```

---

## Output Files / 输出文件

Each module generates CSV files with analysis results:

| Module | Output Files |
|--------|--------------|
| 01 | processed_data_full.csv, variable_list.csv |
| 02 | descriptive_statistics_table1.csv, correlation_matrix.csv |
| 03 | reliability_analysis.csv, item_analysis.csv |
| 04 | mlm_model4_results.csv, mlm_model_comparison.csv |
| 05 | sem_dimensional_paths.csv, sem_mediation_results.csv |
| 06 | moderation_by_dimension.csv, urban_rural_comparison_detailed.csv |
| 07 | cross_level_interactions.csv, community_transmission.csv |

---

## Ethical Approval / 伦理审批

- **Approval Body**: Research Ethics Committee of the School of Music, University of Jinan
- **Approval Number**: JNU-SM-20241128
- **Approval Date**: November 20, 2024
- **Protocol**: All research performed in accordance with the Declaration of Helsinki

---

## Contact / 联系方式

For questions regarding data access or analysis:
- **Email**: 411844292@163.com
- **Institutional Email**: yjsy@ujn.edu.cn
- **ORCID**: 0009-0002-9431-6016

---

## Citation / 引用

If you use this data or code, please cite:
```
Jia, J. (2024). Intergenerational Transmission Mechanisms of National Opera 
Cultural Identity: The Moderating Effect of Family Socialization Investment 
and Multilevel Analysis of Urban-Rural Differences. 
Humanities & Social Sciences Communications.
```

---

## License / 许可

This dataset is made available for academic research purposes only. 
Commercial use is prohibited without explicit permission.

---

**Last Updated / 最后更新**: November 2024
**Version / 版本**: 1.0
