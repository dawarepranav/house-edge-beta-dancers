# House Edge: joint Bayesian inversion of the simulator (adaptation)

## Entry
- Team ID: beta-dancers
- Round: adaptation
- Season ID: 5fffdc4d0c700bbf61c3f911
- Forecast file: submission.csv (uploaded separately)
- Forecast SHA-256: b82a49fd5f8aafe660e4202513a17f8c860474cf6e5c03545e75a2e7e3719033
- Code version: round 1 (main) code, unchanged, refitted on adaptation train (GitHub tag r2)
- Libraries: Python standard library only, so there is no requirements.txt
- Tested on: Python 3.15 and 3.9, macOS; CPU only, one core, no GPU
- Measured runtime: about 1 minute to predict the test split, about 2.5 minutes to refit

## Recreate the forecast
Run from this folder:

    python predict.py --data /data --out /output/submission.csv

This loads params.json and predicts on /data/test without training. The code uses no random numbers, so no seed is needed.

## Recreate params.json (optional)

    python fit_train.py --data /data --out /tmp/params.json

This reads the adaptation train/queries.csv and train/history.csv only. It rediscovers the lag pairs, size lags and cross-machine lag, then fits the parameters by hard EM. It does not read train labels, condition labels or any validation file. Main-round data and parameters are not used.

## What changed and how we adapted
The code is identical to our main-round entry; only params.json was refitted on adaptation train. The rule discovery found that the interacting win lags moved from (1,3), (2,5), (4,7) in main to (3,8), (1,4), (2,6). The payout-size lags moved to 3, 1 and 2 (still the first lag of each win pair). The cross-machine effect stays at lag 3.

Each machine's win follows a signed XNOR of two past wins; win size depends on whether the payout at the size lag was at least 2x. Episodes share a hidden two-state hot/cold chain, and win odds rise with how many other machines in the episode won 3 ticks earlier. The model averages over rule hypotheses and three explanations: no flip, per-machine flip (gradual over about 20 ticks) and a common flip tick. Short-history episodes, detected from the observed tick count, use a near-certain no-flip prior.

Validation with the official checker (adaptation kit): loss ratios stable 0.0120, short history 0.0172, drift 0.0058, shared change 0.0074. The main-round parameters, unchanged, scored far worse (loss ratios 1.39 to 1.79), which is how the change was confirmed.

## How each data split was used
- Adaptation train histories: all fitted parameters and the discovered structure, through fit_train.py.
- Adaptation validation answers: scoring only. Settings in SETTINGS in fit_train.py are carried over unchanged from the main round.
- Train labels and condition labels: not used.
- Adaptation test: model input only.
- Main-round data: not used by this entry.
