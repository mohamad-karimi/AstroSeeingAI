# AstroSeeingAI
Machine learning framework for astronomical seeing prediction using historical observatory and meteorological data.

*[\u0641\u0627\u0631\u0633\u06cc \u0631\u0627 \u0628\u062e\u0648\u0627\u0646\u06cc\u062f\u061f \u0628\u0647 README_FA.md \u0646\u06af\u0627\u0647 \u06a9\u0646\u06cc\u062f \u2192](README_FA.md)*

Predicting real, instrument-measured astronomical seeing at the European
Southern Observatory's Paranal site, using real multi-instrument site
monitoring data (2016-2026) rather than generic weather-forecast features.

**Final model: XGBoost, R\u00B2 = 0.696 on a strict, chronologically held-out
1.5-year future test period.**

---

## Overview

Astronomical "seeing" quantifies the blurring of star images caused by
atmospheric turbulence, and directly limits the effective resolution of a
ground-based telescope on a given night. This project builds a supervised
regression model that predicts seeing from real, site-specific atmospheric
conditions, using public historical data from ESO's Paranal Observatory:

| Instrument | Measures | Role |
|---|---|---|
| DIMM | Seeing (arcsec), airmass | Target variable |
| LHATPRO (microwave radiometer) | Sky IR temperature, liquid water path, precipitable water vapour | Feature source |
| 30 m meteorological tower | Temperature, wind (incl. vertical component), pressure, rain | Feature source |

A fourth instrument, MASS (which measures free-atmosphere turbulence
separately), was deliberately **excluded**: its outputs are themselves
real-time turbulence measurements co-derived with DIMM seeing, so using them
as inputs would be target leakage, not genuine forecasting signal.

## Results

| Model | R\u00B2 (test) | MAE (arcsec) | RMSE (arcsec) |
|---|---|---|---|
| Linear / Ridge / Polynomial regression | < 0.10 | \u2014 | \u2014 |
| Random forest | ~ 0.45 | \u2014 | \u2014 |
| Deep neural network | 0.674 | 0.167 | 0.234 |
| **XGBoost (Optuna-tuned + engineered features) \u2014 final model** | **0.696** | **0.160** | **0.225** |

An ensemble of XGBoost and the DNN was also tested (`src/ensemble_eval.py`)
across a full blend-weight sweep. The best blend reached R\u00B2 = 0.6988, only
0.0025 above XGBoost alone \u2014 within the noise level of this evaluation \u2014
and was not adopted, in favour of the simpler single-model solution.

![Model iteration progress](assets/r2_progress.png)
![DNN vs XGBoost](assets/model_comparison.png)
![Ensemble weight sweep](assets/ensemble_sweep.png)
![Feature importance](assets/feature_importance.png)

## Key Findings

- Wind direction (encoded as sine/cosine) and near-surface (2 m) temperature
  are the strongest predictors.
- A 3-hour rolling mean of wind speed materially improved accuracy \u2014 recent
  atmospheric *trend*, not only the instantaneous state, carries predictive
  signal.
- XGBoost consistently outperformed a comparable deep neural network on this
  tabular dataset; repeated rounds of hyperparameter tuning and feature
  engineering showed diminishing returns (~0.001\u20130.003 R\u00B2 per iteration
  after the first few passes), suggesting the model is close to the
  practical ceiling supported by this feature set.
- A **chronologically blocked**, leakage-free train/validation/test split was
  essential: an earlier random split materially overstated model skill.

## Repository Structure

```
.
├── src/
│   ├── build_dataset.py     # Merges DIMM + LHATPRO + meteo tower into an hourly dataset
│   ├── train_dnn.py         # Regularised neural network baseline
│   ├── train_xgboost.py     # Optuna-tuned XGBoost (final model)
│   └── ensemble_eval.py     # Ensemble exploration (not adopted - see Results)
├── docs/
│   ├── report_EN.docx       # Full technical report (English)
│   └── report_FA.docx       # Full technical report (Persian)
├── assets/                  # Figures used in this README / report
├── requirements.txt
└── LICENSE
```

## Data

Raw instrument data are **not included** in this repository (large files,
and redistribution terms follow ESO's archive policy). They are freely
available for research/educational use from the public ESO Science Archive:

- DIMM seeing: `http://archive.eso.org/wdb/wdb/asm/dimm_paranal/form`
- LHATPRO: available via the same ESO ambient-conditions archive interface
- Meteo tower: `http://archive.eso.org/wdb/wdb/asm/meteo_paranal/form`

Download the desired date range for Paranal, place the exported CSVs
(`seeing.csv`, `lhatpro.csv`, `meteo_<year>.csv` for each year) in the
project root, then run:

```bash
pip install -r requirements.txt
python src/build_dataset.py      # -> paranal_specific_dataset.csv
python src/train_dnn.py          # -> dnn_model.keras
python src/train_xgboost.py      # -> xgb_model.json (final model)
python src/ensemble_eval.py       # optional: reproduces the ensemble comparison
```

Meteo tower exports are split by year because ESO's query form times out on
multi-year requests at 1-minute resolution; `build_dataset.py` accepts a list
of yearly files.

## Limitations & Future Work

- **Forecast availability**: several of the strongest features are
  hyper-local to Paranal's own instrumentation and are not produced by any
  general weather-forecast model; Paranal's own ambient database itself
  updates once per local night rather than in real time. As built, this
  model is best suited to retrospective/statistical analysis rather than a
  live forecasting product.
- **Historical depth**: the DIMM archive used here begins April 2016; ESO's
  earlier-generation DIMM extends back to 1998.
- **Site generalisation**: this model is intentionally specific to Paranal.
  Extending the methodology to other observatories requires an equivalent
  multi-instrument dataset for each site \u2014 collaboration inquiries welcome
  (see contact below).

## License

MIT \u2014 see [LICENSE](LICENSE).

## Contact / Collaboration

I'm interested in extending this methodology to other observatories with
equivalent site-monitoring instrumentation. If your observatory can share
historical seeing-monitor and local meteorological data, I'd welcome the
opportunity to collaborate on a site-specific model. Get in touch:
**mohamadkarimi.dev@gmail.com**.
