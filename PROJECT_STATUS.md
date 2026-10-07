# Project Status

## Ready for GitHub

- [x] Recruiter-friendly README
- [x] Training script based on the supplied LightGBM workflow
- [x] Original notebook preserved
- [x] Requirements file
- [x] Documentation
- [x] License and gitignore
- [x] Reported metrics documented with source caveat

## Not included

- Original dataset: not supplied with the project files
- Trained model binary: not supplied
- Fully reproducible future-forecast pipeline: the supplied notebook references `future_df` and `future_clean` without defining them

## Recommended future improvement

Add a clean recursive multi-step forecasting pipeline that generates future rows and lag features from the most recent observations, then save model artifacts and forecast outputs in `results/`.
