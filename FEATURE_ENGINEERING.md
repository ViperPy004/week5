# Feature Engineering — Week 5

## Overview
Five new features engineered from the Ames Housing dataset, all derived from information available at or before the time of sale (no target leakage).

## Engineered Features

### 1. TotalSF
- **Formula:** `TotalBsmtSF + 1stFlrSF + 2ndFlrSF`
- **Justification:** Combines all living area (basement + both floors) into a single measure of usable square footage. This is the strongest signal for house price. Keeping three separate area columns makes the model split the signal; combining them gives it one clear number.

### 2. TotalBath
- **Formula:** `FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath`
- **Justification:** Total bathroom count (full = 1, half = 0.5) including basement. Bathrooms are a value driver, and a single numeric summary is cleaner than four separate columns that individually have weak signal.

### 3. HouseAge
- **Formula:** `YrSold - YearBuilt`
- **Justification:** Age of the house at the time of sale. Older homes typically sell for less (depreciation), newer ones command a premium. Direct measure that captures this effect better than raw build year.

### 4. HasRemodeled
- **Formula:** `1 if YearRemodAdd != YearBuilt else 0`
- **Justification:** Binary flag distinguishing remodeled homes from original builds. Remodeled houses usually carry a price premium. A binary flag lets the model learn this pattern without needing to parse two year columns.

### 5. TotalPorchSF
- **Formula:** `OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch`
- **Justification:** Total porch / outdoor-covered area. Outdoor living space adds value. Summing the four porch types gives a single measure rather than four sparse columns (many houses have 0 for some types).

## Leakage Check
All five features are computable from pre-sale information. None uses `SalePrice` or anything derived from it. No target leakage introduced.

## Engineered Dataset
Saved as `clean_data_engineered.csv` — original columns plus the 5 new features.
