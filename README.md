# House Edge: joint Bayesian inversion of the simulator

## Entry
- Team ID: beta-dancers
- Round: main
- Season ID: 5fffdc4d0c700bbf61c3f911
- Forecast file: submission.csv (uploaded separately)
- Forecast SHA-256: 608859f7bb53c3f0388f4a066f592a1233105317cf3823bd0f4d103f2ef64669
- Environment: `python` image; Python standard library only, so there is no requirements.txt
- Tested on: Python 3.13, macOS; CPU only, one core, no GPU
- Measured runtime: about 70 seconds to predict the test split, about 2 minutes to refit

## Recreate the forecast
Run from this folder:

    python predict.py --data /data --out /output/submission.csv

This loads params.json and predicts on /data/test without training. The code uses no random numbers, so no seed is needed.

## Recreate params.json (optional)

    python fit_train.py --data /data --out /tmp/params.json

This reads train/queries.csv and train/history.csv only. It discovers the lag pairs, the size lags and the cross-machine lag, then fits the parameters by hard EM. It does not read train labels, condition labels or any validation file.

## Method
Each machine's win follows a signed XNOR of past wins at lag pair (1,3), (2,5) or (4,7). Win size follows whether the payout at lag 1, 2 or 4 was at least 2x; that size lag is normally the first lag of the machine's win pair, so the two are tied by a strong prior. Episodes share a hidden two-state hot/cold chain. Win odds also rise with the number of other machines in the episode that won 3 ticks earlier (the cross-machine lag is discovered on train; lag 3 gains 38 nats of train log-likelihood, every other lag under 1).

The model averages over rule hypotheses and three explanations: no flip, per-machine flip (drift, modelled as a gradual reversal over about 20 ticks) and a common flip tick (shared change). Episodes whose histories are sparse (short history) use a near-certain no-flip prior; this is detected from the observed tick count, not from condition labels. Histories are grouped by episode_id and machine_id, only ticks before each cutoff are used, and information is shared only within an episode.

Validation points with the official checker: 59.031 / 60 (stable 0.0068, short history 0.0392, drift 0.0121, shared change 0.0075 loss ratios).

## How each data split was used
- Train histories: all fitted parameters and the discovered structure (lag pairs, size lags, cross-machine lag), through fit_train.py.
- Train condition labels: analysis only (for example, checking that short-history episodes show no decisive flips). Not used for fitting or prediction.
- Validation answers: exploring the data, which is how the rule form, the lag tie and the cross-machine effect were first noticed; choosing the settings in SETTINGS in fit_train.py; and scoring. No value in params.json is fitted to validation answers; the cross-machine lag and strength are re-discovered and fitted on train.
- Train labels: not used.
- Test: model input only.
