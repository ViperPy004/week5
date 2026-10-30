# Model Comparison — Week 5

## Setup
- Dataset: Ames Housing (cleaned, Week 1) + 5 engineered features
- Target: log(SalePrice)
- 5-fold cross-validation (KFold, shuffle=True, random_state=42)
- Preprocessing (median imputer + scaler for numeric; "None" imputer + one-hot for categorical) inside Pipeline — no leakage
- Hyperparameter tuning: RandomizedSearchCV (n_iter=20, 5-fold CV, scoring='neg_root_mean_squared_error')

## Model Comparison Table (5-Fold CV)

| Model | Mean CV RMSE (log) | Mean CV R² | Training Time | Note |
|---|---|---|---|---|
| Week 3 baseline (Ridge + log, α=0.1) | 0.1505 ± 0.0453 | 0.8379 ± 0.1214 | 3s | Simple, fast, reference |
| Random Forest (default) | 0.1426 ± 0.0180 | 0.8681 ± 0.0414 | 100s | Solid, but slow |
| Random Forest (tuned) | 0.1384 ± 0.0172 | 0.8767 ± 0.0336 | 421s | Marginal gain over default; expensive |
| XGBoost (default) | 0.1435 ± 0.0164 | 0.8667 ± 0.0393 | 11s | Comparable to RF, 10x faster |
| **XGBoost (tuned)** | **0.1319 ± 0.0192** | **0.8864 ± 0.0404** | **35s** | Best CV score, fastest of the tuned models |

## Tuned Hyperparameters

**Random Forest:** `n_estimators=478, max_depth=17, max_features=0.3, min_samples_leaf=3, min_samples_split=2`

**XGBoost:** `n_estimators=343, max_depth=5, learning_rate=0.115, subsample=0.84, colsample_bytree=0.60, min_child_weight=7, gamma=0.012`

## Hold-out Test Set (XGBoost tuned, untouched 20% from Week 3 split)

- MAE: **$15,660**
- RMSE: **$27,122**
- R²: **0.898**

The CV estimate (R² 0.886) and hold-out estimate (R² 0.898) are consistent — the model is not overfitting the CV folds.

## Verdict

**Ship XGBoost (tuned).**

- **Best CV score:** XGBoost tuned (RMSE 0.1319, R² 0.886) beats both the Week 3 baseline (RMSE 0.1505, R² 0.838) and both Random Forest variants.
- **Fastest tuned model:** 35s vs 421s for the tuned RF — a 12x speed advantage for better performance.
- **Stable:** Std of CV RMSE (0.019) is much tighter than baseline's (0.045), meaning XGBoost performs consistently across folds.

**Is the improvement meaningful?**
- RMSE improves from 0.1505 → 0.1319 (~12% reduction on log scale).
- R² improves from 0.838 → 0.886 (≈ 0.05 absolute, meaningful).
- On the hold-out test, R² = 0.898 — a real, repeatable gain over the baseline.

Yes, the improvement is meaningful. The engineered features (especially `TotalSF` and `HouseAge`) and XGBoost's boosting give a real accuracy lift — not just noise.

**Tradeoff vs Week 3 baseline:**
- Week 3 baseline: 3s, R² 0.838 (CV)
- XGBoost tuned: 35s, R² 0.886 (CV)
- 12x training cost for +0.05 R² — worth it because training is still fast (35s) and this is a one-time cost per model refresh.

**Why not Random Forest?**
Random Forest tuned reaches R² 0.877 but takes 421s to tune and is only marginally better than default RF. XGBoost gives a higher score in less time. RF's bagging approach doesn't benefit as much from the engineered features here.

## Recommendation
Deploy **XGBoost (tuned)** as the production model. It offers the best accuracy-to-cost ratio, is stable across folds, and the feature engineering paid off — proving that better features plus a proper ensemble beat a simple linear model by a meaningful margin.
