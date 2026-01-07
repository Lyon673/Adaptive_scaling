# Statistical Analysis Summary: Speed Group Comparison

## Study Design

### Groups
- **Slow Group**: Data 45-54 (n=10)
- **Medium Group**: Data 0-29 (n=30)
- **Fast Group**: Data 55-64 (n=10)

### Metrics Analyzed
1. **Total Time**: Total duration of the surgical task
2. **Total Distance (Left)**: Total distance traveled by left PSM
3. **Clutch Times (Left/Right)**: Number of clutch activations
4. **Gracefulness**: Movement quality metric (lower is better)
5. **Smoothness**: Jerk-based smoothness metric (lower is better)

---

## Key Findings

### 1. Total Time ⏱️
**Highly Significant Difference (p < 0.001)***

- **Slow**: 26.47 ± 4.71 seconds
- **Medium**: 23.94 ± 2.66 seconds  
- **Fast**: 19.32 ± 2.49 seconds

**Post-hoc Results:**
- Slow vs Fast: **p=0.0002*** (large effect, r=-1.000)
- Medium vs Fast: **p=0.0001*** (large effect, r=-0.827)
- Slow vs Medium: p=0.1464 (not significant)

**Interpretation:** Fast group completed tasks significantly faster than both other groups. This validates the speed manipulation.

---

### 2. Total Distance (Left PSM) 📏
**Not Significant (p = 0.0676)**

- **Slow**: 0.211 ± 0.021 m
- **Medium**: 0.221 ± 0.014 m
- **Fast**: 0.229 ± 0.021 m

**Interpretation:** All groups followed similar trajectory paths despite different speeds. This suggests speed changes didn't significantly alter the surgical approach or path efficiency.

---

### 3. Clutch Times 🖱️

#### Left Clutch Times (p < 0.01)**
- **Slow**: 1.10 ± 0.32 times
- **Medium**: 1.60 ± 0.72 times
- **Fast**: 1.00 ± 0.00 times

**Post-hoc:** Medium vs Fast: **p=0.0099*** (medium effect, r=-0.467)

#### Right Clutch Times (p < 0.001)***
- **Slow**: 2.20 ± 0.42 times
- **Medium**: 1.53 ± 0.94 times
- **Fast**: 0.80 ± 0.42 times

**Post-hoc:** 
- Slow vs Fast: **p=0.0001*** (perfect separation, r=-1.000)
- Medium vs Fast: **p=0.0167*** (medium effect, r=-0.487)

**Interpretation:** Faster speeds correlated with fewer clutch activations, especially for the right hand. This suggests faster speeds may encourage more continuous movements or reflect better operator confidence.

---

### 4. Gracefulness 🎯
**Not Significant (p = 0.4817)**

- **Slow**: 4.22 ± 0.41
- **Medium**: 3.73 ± 1.41
- **Fast**: 3.95 ± 0.17

**Interpretation:** Movement gracefulness (curvature-based metric) was similar across all speed groups. Speed manipulation did not significantly affect movement quality in terms of trajectory curvature.

---

### 5. Smoothness 🌊
**Highly Significant Difference (p < 0.001)***

- **Slow**: 5.21 ± 0.61
- **Medium**: 4.96 ± 0.33
- **Fast**: 4.43 ± 0.37

**Post-hoc:**
- Slow vs Fast: **p=0.0022*** (large effect, r=-0.820)
- Medium vs Fast: **p=0.0003*** (large effect, r=-0.773)
- Slow vs Medium: p=0.4078 (not significant)

**Interpretation:** Fast group showed significantly smoother movements (lower jerk). This counterintuitive finding suggests that higher speeds may facilitate more natural, less hesitant movements, reducing jerky corrections.

---

## Overall Conclusions

### ✅ Validated Findings
1. **Speed Manipulation Successful**: Total time clearly differentiated groups
2. **Improved Smoothness at Higher Speeds**: Fast group had lower jerk values
3. **Reduced Clutching at Higher Speeds**: Particularly for right hand

### 🤔 Unexpected Results
1. **No Path Length Difference**: Similar distances despite speed differences
2. **No Gracefulness Difference**: Curvature-based quality unaffected by speed
3. **Better Smoothness at Higher Speeds**: Contradicts typical speed-accuracy tradeoff

### 💡 Implications
- Higher speeds may promote more confident, continuous movements
- Speed manipulation primarily affects temporal aspects, not spatial planning
- Jerk-based smoothness improves with speed, possibly due to reduced hesitation
- Clutch usage decreases with faster speeds, indicating more efficient workspace usage

---

## Statistical Methods

### Tests Performed
1. **Shapiro-Wilk Test**: Normality assessment
2. **Levene's Test**: Homogeneity of variance
3. **One-way ANOVA**: Parametric group comparison
4. **Kruskal-Wallis Test**: Non-parametric alternative
5. **Mann-Whitney U Test**: Pairwise post-hoc comparisons (Bonferroni corrected α=0.0167)

### Effect Size
- Rank-biserial correlation (r):
  - |r| < 0.3: Small effect
  - 0.3 ≤ |r| < 0.5: Medium effect
  - |r| ≥ 0.5: Large effect

---

## Recommendations

1. **For Training**: Consider starting with medium speeds to balance learning and performance
2. **For Performance**: Higher speeds appear beneficial for smoothness and efficiency
3. **For Research**: Investigate the mechanism behind improved smoothness at higher speeds
4. **For Interface Design**: Optimize for reduced clutching at operational speeds

---

## Data Files
- Full report: `statistical_report.txt`
- Visualizations: `metrics_comparison.png`
- Raw data: `../data/[group_id]_data_12-01/`

---

*Analysis performed: December 28, 2025*
*Statistical significance: * p<0.05, ** p<0.01, *** p<0.001*

