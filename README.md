[README.md](https://github.com/user-attachments/files/24363977/README.md)
# Research Data and Materials Package
# 研究数据与材料包

## Study Title / 研究标题
**Intergenerational Transmission Mechanisms of National Opera Cultural Identity: The Moderating Effect of Family Socialization Investment and Multilevel Analysis of Urban-Rural Differences**

**民族歌剧文化认同代际传递机制：家庭社会化投入的调节效应及城乡差异多层分析**

---

## Package Contents / 包内容

### 01_Data_Files / 数据文件
| File | Description | Format |
|------|-------------|--------|
| National_Opera_Cultural_Identity_Dataset.xlsx | Raw data with codebook | Excel |
| processed_data_full.csv | Full processed dataset | CSV |
| processed_data_urban.csv | Urban subset | CSV |
| processed_data_rural.csv | Rural subset | CSV |

### 02_Survey_Instruments / 调查工具
| File | Description |
|------|-------------|
| Survey_Instrument_and_Protocol.docx | Complete questionnaire, interview guide, and data collection procedures |

### 03_Variable_Documentation / 变量文档
| File | Description |
|------|-------------|
| Variable_Definitions_and_Transformations.docx | Variable definitions, coding schemes, and transformations |
| variable_list.csv | Complete list of all variables with summary statistics |

### 04_Analysis_Codes / 分析代码
| File | Description |
|------|-------------|
| 01_data_preprocessing.py | Data cleaning and variable creation |
| 02_descriptive_statistics.py | Descriptive statistics and comparisons |
| 03_reliability_validity.py | Scale reliability and validity tests |
| 04_multilevel_model.py | Hierarchical linear modeling |
| 05_structural_equation_model.py | Path analysis and SEM |
| 06_moderation_urban_rural.py | Moderation and urban-rural analysis |
| 07_cross_level_interaction.py | Cross-level interaction effects |
| README.md | Code documentation |

### 05_Supporting_Files / 支持文件
Analysis output files including:
- Descriptive statistics tables
- Reliability analysis results
- Correlation matrices
- Model estimation results
- Mediation analysis results
- Group comparison results

---

## Sample Information / 样本信息
- **Total families**: 2,847
- **Communities**: 126
- **Cities**: 18 prefecture-level cities
- **Regions**: Eastern (6), Central (6), Western (6)
- **Urban/Rural**: 54.3% Urban, 45.7% Rural

---

## How to Use / 使用方法

### For Data Access / 数据访问
1. Open `01_Data_Files/National_Opera_Cultural_Identity_Dataset.xlsx`
2. Sheet "Raw_Data" contains all original survey responses
3. Sheet "Codebook" explains all variables
4. Sheet "Scale_Items" provides questionnaire items in English and Chinese

### For Replication / 复现分析
1. Install Python 3.8+ with required packages:
   ```
   pip install pandas numpy scipy statsmodels openpyxl
   ```
2. Place data file in `04_Analysis_Codes/` directory
3. Run scripts in numerical order (01 through 07)

---

## Ethics Approval / 伦理审批
- **Approval Body**: Research Ethics Committee, School of Music, University of Jinan
- **Approval Number**: JNU-SM-20241128
- **Approval Date**: November 20, 2024

---

## Contact / 联系方式
- **Corresponding Author**: Jie Jia (贾洁)
- **Email**: 411844292@163.com
- **ORCID**: 0009-0002-9431-6016

---

## Data Availability Statement / 数据可用性声明
The data supporting the findings of this study are available from the corresponding author upon reasonable request.

本研究的支持数据可根据合理要求从通讯作者处获取。

---

**Package Version**: 1.0
**Created**: November 2025
