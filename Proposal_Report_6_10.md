# PREDICTING NATIONAL LIFE EXPECTANCY AND IDENTIFYING KEY DRIVERS
## A Machine Learning Approach

**Team Members:**
- Mr. Htut Ko Ko (st126010)
- Mr. Kaung Hein Htet (st126477)  
- Mr. Michael R. Lacar (st126161)

**Course:** AT82.01 – Computer Programming for Data Science and AI  
**Instructor:** Dr. Chantri Polprasert  
**Date:** October 6, 2025

---

## ABSTRACT

This project develops a comprehensive machine learning pipeline to predict national life expectancy using multi-source datasets from World Bank, UNDP, and World Happiness Reports. Our approach addresses critical methodological challenges including temporal data leakage prevention and class imbalance handling while identifying key socioeconomic, health, and governance drivers. The integrated dataset spans 217 countries from 1960-2024 with 15,000+ observations. Preliminary analysis reveals strong correlations: infant mortality (r=-0.85), GDP per capita (r=0.75), and health expenditure (r=0.55). Our target performance is RMSE <3.5 years and R²>0.85, surpassing existing baseline models through advanced feature engineering and ensemble methods.

---

## 1. INTRODUCTION

### Background and Motivation

Life expectancy serves as a fundamental indicator of national health, development, and societal well-being. It reflects the complex interplay between healthcare systems, economic prosperity, environmental conditions, and social structures. The COVID-19 pandemic highlighted the importance of understanding factors that contribute to population resilience and health system effectiveness.

### Business Understanding

Accurate life expectancy prediction has critical applications across multiple sectors:

- **Government Policy:** Resource allocation, healthcare planning, pension system design
- **Insurance Industry:** Risk assessment, premium calculation, actuarial modeling
- **International Development:** Aid prioritization, SDG monitoring, intervention targeting
- **Healthcare Organizations:** Strategic planning, capacity building, population health management

### Potential Impact

This research will provide:
- Data-driven insights for evidence-based policy making
- Early warning systems for declining health trends
- Actionable recommendations for improving population health outcomes
- Novel integration of happiness metrics with traditional health indicators

---

## 2. PROBLEM STATEMENT

**Primary Objective:** Develop a robust machine learning pipeline that accurately predicts national life expectancy while identifying the most significant factors influencing population longevity.

**Specific Research Questions:**
1. Can we achieve superior prediction performance (RMSE <3.5 years) using multi-source data integration?
2. Which factors most strongly predict life expectancy across different regions and time periods?
3. How do temporal modeling techniques improve prediction accuracy while preventing data leakage?
4. What actionable insights can guide evidence-based policy interventions?

**Success Criteria:**
- Quantitative: RMSE <3.5 years, R²>0.85 on held-out test data
- Qualitative: Identify top 5 most influential factors with statistical significance
- Practical: Provide actionable recommendations for policy makers

---

## 3. RELATED WORKS

### Foundational Research
- **Preston Curve (1975):** Established GDP-life expectancy relationship
- **WHO Social Determinants (Marmot et al., 2008):** Social conditions impact on health
- **Environmental Health (Pope et al., 2009):** Air pollution effects on mortality

### Machine Learning Applications
- Traditional approaches: Linear regression, demographic life tables
- Advanced methods: Random Forest, Gradient Boosting, Neural Networks
- Limitations: Single-source data, inadequate temporal handling, limited interpretability

### Research Gaps
Our study addresses:
- Multi-source data integration (World Bank + UNDP + Happiness)
- Proper temporal dependency handling with lag features
- Systematic data leakage prevention
- Class imbalance handling for robust classification

---

## 4. DATASETS

### Primary Data Sources

**World Bank Development Indicators**
- Coverage: 217 countries, 1960-2024
- Key Variables: GDP per capita, health expenditure, infant mortality, PM2.5 pollution, education expenditure, fertility rate, age dependency ratio

**UNDP Human Development Reports**
- Coverage: 191 countries, 1990-2024
- Key Variables: Human Development Index, education indicators, income measures

**World Happiness Report**
- Coverage: 150+ countries, 2015-2024
- Key Variables: Life satisfaction, social support, freedom, generosity, corruption perception

### Data Integration
- Temporal alignment using annual observations
- Geographic matching via ISO country codes
- Quality assessment with systematic missing value analysis
- Final dataset: 15,000+ country-year observations with 70-85% completeness

---

## 5. METHODOLOGY

### Exploratory Data Analysis
- Comprehensive correlation analysis with life expectancy
- Missing value pattern identification
- Temporal trend visualization
- Regional comparison analysis

### Data Preprocessing Pipeline

**Temporal Feature Engineering:**
```python
# Lag features to prevent data leakage
gdp_per_capita_lag1, health_exp_lag1, infant_mortality_lag1

# Moving averages for trend capture
gdp_per_capita_ma3, health_exp_ma3
```

**Data Leakage Prevention:**
- Temporal splitting: training data precedes test data chronologically
- Time-series aware cross-validation
- Lag features using only historical information

**Class Imbalance Handling:**
- Life expectancy categories: Low (<60), Medium (60-75), High (>75 years)
- SMOTE oversampling for balanced training
- Class weights adjustment in models

### Model Development
1. **Baseline Models:** Linear regression with economic indicators
2. **Advanced Models:** Random Forest, XGBoost, Neural Networks
3. **Ensemble Approach:** Stacking for optimal performance
4. **Validation Strategy:** Time-series CV + geographic validation

---

## 6. PRELIMINARY RESULTS

### Dataset Characteristics
- Total observations: 15,247 country-year records
- Countries: 217 unique countries
- Time range: 1960-2024 (varying by indicator)
- Completeness: 70-85% for key indicators post-1990

### Key Correlations with Life Expectancy
| Predictor | Correlation | Significance |
|-----------|-------------|--------------|
| Infant Mortality | -0.85 | p < 0.001 |
| GDP per Capita | +0.75 | p < 0.001 |
| Health Expenditure % GDP | +0.55 | p < 0.001 |
| PM2.5 Pollution | -0.45 | p < 0.001 |
| Education Expenditure | +0.42 | p < 0.001 |

### Regional Patterns
- Western Europe: 80.2 ± 2.1 years (highest)
- East Asia & Pacific: 72.8 ± 8.4 years (most improved: +25.3 since 1960)
- Latin America: 71.5 ± 4.2 years
- Sub-Saharan Africa: 58.3 ± 7.8 years (needs intervention)

### Baseline Performance
- Simple Linear Regression: RMSE = 4.2 years, R² = 0.78
- Target Performance: RMSE <3.5 years, R² >0.85

---

## 7. EXPECTED IMPACT AND APPLICATIONS

### Academic Contributions
- First comprehensive study integrating happiness metrics with health indicators
- Methodological framework for temporal health data analysis
- Quantitative evidence on relative importance of life expectancy drivers

### Practical Applications
- **Government:** Evidence-based resource allocation and policy development
- **International Development:** Aid prioritization and program effectiveness evaluation
- **Private Sector:** Insurance product development and healthcare investment decisions
- **Research:** Enhanced understanding of population health determinants

### Societal Benefits
- Improved health outcomes through better policy targeting
- Reduced health inequalities via evidence-based interventions
- Enhanced preparedness for health crises
- More effective use of public resources

---

## 8. PROJECT TIMELINE AND FEASIBILITY

### 6-Week Implementation Plan
**Weeks 1-2:** Complete data integration and comprehensive EDA
**Weeks 3-4:** Implement advanced models and hyperparameter optimization
**Weeks 5-6:** Comprehensive evaluation and documentation

### Risk Assessment
- **Data Quality:** Mitigated through multiple imputation and sensitivity analysis
- **Model Overfitting:** Addressed via robust cross-validation and regularization
- **Timeline Constraints:** Managed through agile development and prioritized deliverables

### Resource Requirements
- Human: 3 team members × 20 hours/week
- Technical: Python ecosystem, standard computing resources
- Data: Publicly available datasets (~500MB storage)

---

## 9. CONCLUSION

This project addresses a critical global health challenge through a comprehensive machine learning approach that combines methodological rigor with practical relevance. Our multi-source data integration, proper temporal modeling, and focus on interpretability distinguish this work from existing approaches.

The preliminary results demonstrate strong potential for achieving superior predictive performance while providing actionable insights for policy makers. The integration of happiness and social indicators represents a novel contribution to life expectancy research.

Upon completion, this research will provide valuable tools for evidence-based decision making in global health, contributing to improved population health outcomes worldwide.

---

## REFERENCES

1. Avendano, M., & Kawachi, I. (2014). Why do Americans have shorter life expectancy and worse health? *Annual Review of Public Health*, 35, 307–325.

2. Marmot, M., Friel, S., Bell, R., Houweling, T. A., & Taylor, S. (2008). Closing the gap in a generation: health equity through action on the social determinants of health. *The Lancet*, 372(9650), 1661-1669.

3. Pope III, C. A., Ezzati, M., & Dockery, D. W. (2009). Fine-particulate air pollution and life expectancy in the United States. *New England Journal of Medicine*, 360(4), 376-386.

4. Preston, S. H. (1975). The changing relation between mortality and level of economic development. *Population Studies*, 29(2), 231-248.

5. World Bank. (2024). World Development Indicators. Retrieved from https://databank.worldbank.org/

6. UNDP. (2024). Human Development Reports. Retrieved from http://hdr.undp.org/

7. Helliwell, J., Layard, R., & Sachs, J. (2024). World Happiness Report 2024. UN Sustainable Development Solutions Network.
