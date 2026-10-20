# Final runs — cross-dataset pruning (mean ± std over 3 seeds, final-epoch checkpoint)

Metrics: Acc = accuracy, BalAcc = balanced accuracy, mF1 = macro-F1, AUC.
lr = fine-tune learning rate. Dense = unpruned baseline.

## FloodNet (optical UAV)

| Arch | Method | Ratio | lr | Acc | BalAcc | macro-F1 | AUC |
|------|--------|------:|----:|------|--------|----------|-----|
| ResNet18 | dense        |  0% | 0.01  | 0.970±0.004 | 0.914±0.016 | 0.929±0.011 | 0.991±0.003 |
| ResNet18 | unstructured | 50% | 0.01  | 0.965±0.004 | 0.900±0.010 | 0.917±0.010 | 0.995±0.002 |
| ResNet18 | unstructured | 90% | 0.01  | 0.960±0.004 | 0.920±0.010 | 0.911±0.009 | 0.989±0.002 |
| ResNet18 | structured   | 30% | 0.01  | 0.968±0.005 | 0.913±0.016 | 0.925±0.012 | 0.986±0.012 |
| ResNet18 | structured   | 50% | 0.01  | 0.970±0.007 | 0.909±0.011 | 0.928±0.016 | 0.989±0.004 |
| ResNet18 | structured   | 70% | 0.01  | 0.938±0.026 | 0.856±0.076 | 0.857±0.063 | 0.930±0.052 |
| ResNet18 | structured   | 90% | 0.01  | 0.893±0.059 | 0.836±0.047 | 0.798±0.064 | 0.930±0.018 |
| ResNet18 | structured   | 50% | 0.001 | 0.960±0.005 | 0.943±0.003 | 0.914±0.009 | 0.981±0.003 |
| ResNet18 | structured   | 90% | 0.001 | 0.807±0.093 | 0.778±0.045 | 0.699±0.093 | 0.881±0.051 |
| VGG-16 | dense        |  0% | 0.01  | 0.967±0.002 | 0.924±0.009 | 0.924±0.006 | 0.990±0.000 |
| VGG-16 | unstructured | 50% | 0.01  | 0.967±0.002 | 0.924±0.009 | 0.924±0.006 | 0.990±0.001 |
| VGG-16 | unstructured | 90% | 0.01  | 0.975±0.007 | 0.940±0.023 | 0.943±0.015 | 0.980±0.020 |
| VGG-16 | structured   | 30% | 0.01  | 0.603±0.339 | 0.528±0.039 | 0.386±0.199 | 0.737±0.035 |
| VGG-16 | structured   | 50% | 0.01  | 0.883±0.058 | 0.887±0.032 | 0.803±0.076 | 0.940±0.027 |
| VGG-16 | structured   | 70% | 0.01  | 0.812±0.041 | 0.664±0.124 | 0.621±0.064 | 0.675±0.187 |
| VGG-16 | structured   | 90% | 0.01  | 0.375±0.353 | 0.500±0.000 | 0.230±0.167 | 0.500±0.000 |
| VGG-16 | structured   | 70% | 0.001 | 0.500±0.374 | 0.500±0.000 | 0.289±0.177 | 0.922±0.003 |

## SEN1FLOODS11 (Sentinel-1 SAR)

| Arch | Method | Ratio | lr | Acc | BalAcc | macro-F1 | AUC |
|------|--------|------:|----:|------|--------|----------|-----|
| ResNet18 | dense        |  0% | 0.01  | 0.840±0.009 | 0.840±0.007 | 0.827±0.009 | 0.908±0.013 |
| ResNet18 | unstructured | 50% | 0.01  | 0.845±0.006 | 0.850±0.007 | 0.833±0.006 | 0.912±0.011 |
| ResNet18 | unstructured | 90% | 0.01  | 0.815±0.013 | 0.811±0.013 | 0.799±0.013 | 0.888±0.017 |
| ResNet18 | structured   | 30% | 0.01  | 0.783±0.032 | 0.786±0.026 | 0.768±0.030 | 0.858±0.015 |
| ResNet18 | structured   | 50% | 0.01  | 0.780±0.006 | 0.767±0.008 | 0.759±0.006 | 0.841±0.010 |
| ResNet18 | structured   | 70% | 0.01  | 0.748±0.033 | 0.758±0.020 | 0.734±0.028 | 0.834±0.013 |
| ResNet18 | structured   | 90% | 0.01  | 0.578±0.038 | 0.656±0.036 | 0.577±0.038 | 0.763±0.026 |
| ResNet18 | structured   | 50% | 0.001 | 0.731±0.016 | 0.744±0.002 | 0.718±0.010 | 0.833±0.015 |
| ResNet18 | structured   | 90% | 0.001 | 0.618±0.020 | 0.646±0.005 | 0.609±0.013 | 0.725±0.006 |
| VGG-16 | dense        |  0% | 0.01  | 0.661±0.230 | 0.709±0.148 | 0.621±0.262 | 0.858±0.003 |
| VGG-16 | unstructured | 50% | 0.01  | 0.641±0.217 | 0.691±0.135 | 0.601±0.248 | 0.849±0.005 |
| VGG-16 | unstructured | 90% | 0.01  | 0.638±0.215 | 0.687±0.132 | 0.597±0.245 | 0.846±0.014 |
| VGG-16 | structured   | 30% | 0.01  | 0.632±0.211 | 0.669±0.120 | 0.586±0.237 | 0.834±0.005 |
| VGG-16 | structured   | 50% | 0.01  | 0.548±0.151 | 0.643±0.101 | 0.519±0.190 | 0.788±0.007 |
| VGG-16 | structured   | 70% | 0.01  | 0.445±0.156 | 0.500±0.000 | 0.300±0.070 | 0.502±0.225 |
| VGG-16 | structured   | 90% | 0.01  | 0.445±0.156 | 0.500±0.000 | 0.300±0.070 | 0.500±0.000 |
| VGG-16 | structured   | 70% | 0.001 | 0.500±0.165 | 0.500±0.000 | 0.325±0.074 | 0.700±0.000 |

## One-line reading
- Unstructured pruning transfers across both modalities (ResNet18 holds to 90% on both; macro-F1 within ~0.03 of dense).
- Structured pruning degrades SAR earlier and more steadily than optical on ResNet18 (SAR drops from 30%; optical stable to 50%).
- Lower fine-tune lr (0.001) does NOT rescue the SAR structured drop → the drop is not a fine-tuning-recipe artifact.
- VGG-16 is unstable across seeds (dense SAR 0.66±0.23; structured collapses on BOTH datasets) → VGG-16 structured pruning is unreliable regardless of sensor.
