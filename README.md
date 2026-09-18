# Air Quality Prediction (PM2.5)

Predict the PM2.5 concentration one hour ahead from recent pollution and weather
readings, and translate the number into an AQI category a person can act on.

A self-study project: every stage runs from the command line, and the notebook
explains the reasoning behind each choice.

## Quick start

```bash
pip install -r requirements.txt

python src/train.py       # loads data, compares models, saves the best one
python src/evaluate.py    # error plots -> reports/
python src/predict.py --hours 6
```

Then open `notebooks/air_quality_prediction.ipynb` for the walkthrough.

## Data

Uses the UCI *Beijing PM2.5* dataset (hourly pollution + weather, 2010–2014),
downloaded automatically on first run. **If there's no internet, the project
falls back to a synthetic dataset** with the same columns and realistic daily
and seasonal patterns, so everything still runs. The console tells you which
one you got.

To use Indian data instead, edit `src/data.py` — the CPCB CAAQMS portal and the
OpenAQ API both publish station-level readings. Nothing downstream changes as
long as the column names match.

## Structure

```
src/config.py      paths, target, horizon, split size
src/data.py        download / synthetic fallback / cleaning
src/features.py    lags, rolling stats, cyclic time features, chronological split
src/train.py       baseline + 3 models, metrics, saves best_model.joblib
src/evaluate.py    time series, residual and importance plots
src/predict.py     recursive multi-hour forecast from the saved model
src/aqi.py         PM2.5 -> AQI (CPCB breakpoints) + health advice
notebooks/         annotated end-to-end walkthrough
reports/           metrics.json, test_predictions.csv, plots
```

## How the problem is framed

A time series becomes a supervised learning table: features are built from
**past** values, the target is shifted from the **future**.

- **Target** — PM2.5 at t+1 (change `HORIZON` in `config.py`)
- **Features** — lags of PM2.5 (1–48h), rolling mean/std (3, 12, 24h), first
  difference, current weather, lagged temperature and wind, cyclic encodings of
  hour and month, wind direction one-hot
- **Split** — chronological, last 20% held out. Never shuffled: random splits on
  time series leak future information and produce scores that don't survive
  contact with reality.

## The baseline matters more than the model

The comparison includes **persistence** — predict that the next hour equals the
current hour. PM2.5 is highly autocorrelated, so this naive rule is already
strong, and it's the number every model must be judged against.

Reported metrics: MAE (average error in ug/m3), RMSE (penalises big misses),
R² (variance explained).

## Known limitations

Worth stating plainly rather than hiding:

- Error grows with pollution level, and predictions **lag sudden spikes** — the
  model is weakest exactly when a forecast would be most useful.
- Multi-hour forecasts feed predictions back as inputs, so errors compound.
  Past ~3 hours the output is indicative, not reliable.
- The model uses *current* weather as a feature. A real deployment would need a
  weather forecast for that column, and would inherit its errors.
- CPCB breakpoints are defined on 24-hour averages; applying them to an hourly
  value is an approximation.

## Next steps

See the last section of the notebook — start by setting `HORIZON = 24` and
watching what happens.
