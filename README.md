# Independent score normalization for weak-return detection

Research software and saved experiment outputs for **Independent score
normalization for weak-return detection in single-photon lidar**,
Vecdi Emre Levent, Fenerbahçe University.

The method is a statistical signal detector. Its input is an event-time record,
a known pulse template, acquisition settings and an independently trained null
normalization bank. It reports a secondary-return decision and estimated locations.

## What the experiments establish

- The fixed-recovery, two-surface detection study is a simulation study.
- The main cohort contains 256 two-return and 384 single-return queries.
- At the nominal 1% level the proposed detector gives 203 correct localizations,
  one wrong location and four null alarms. The matched own-null censored score
  gives 186, one and five. In the bright-hidden regime the correct counts are
  30/64 versus 14/64 and both have 2/96 null alarms.
- The extended-start full-likelihood-ratio reference gives 196 correct,
  one wrong location and two null alarms. Different observed false-alarm counts
  must be retained when interpreting that comparison.
- Measured timestamps evaluate both the observation model and single-surface detector false alarms. A separate, explicitly semi-synthetic study completes latent photon paths from measured records and injects a known weak return before sensor filtering. It does not provide physically measured two-surface accuracy.

## Install

Use Python 3.11. From the repository root:

```sh
python -m venv .venv
# Activate .venv using the command appropriate to your operating system.
python -m pip install -r requirements.txt
```

Set OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS and NUMBA_NUM_THREADS
to 1 before recomputing simulations. The original numerical studies use up to
eight worker processes. The first Numba invocation includes JIT compilation.

## Reproduce reported results from saved outputs

```sh
python reproduce.py --check
python reproduce.py --figures
```

`--check` independently recounts all main operating points and regimes,
checks the 199-draw calibration ranks, verifies input hashes, checks the
amplitude/separation and mismatch outputs, and verifies held-out measured
log-score summaries. `--figures` rebuilds the six plots, tables and result
fragments in `manuscript/ded_style_article_v2/` from saved experiment outputs.
Plots use an installed Times New Roman font; on systems without it, Matplotlib
may substitute another font. This affects appearance, not numerical counts.

No expensive detector optimization is needed for these two commands. The
original query-level SQLite files and fitted model arrays are included as
individual files, not in an archive. `MANIFEST.json` records their hashes.

## Recompute simulations

The following commands use the original frozen algorithms, seeds and protocols:

```sh
python src/score_peak_localization_development.py run
python tools/revision_censored_bank.py run
python tools/revision_censored_bank_common.py run
python tools/revision_exposure_bank.py run
python tools/revision_sensitivity.py power
python tools/revision_sensitivity.py robust
python tools/revision_safe_robustness.py run
python src/localization_postfailure_full_joint_diagnosis.py run
```

These runners resume their own SQLite outputs. A completed output is not
silently replaced. To recompute a study from scratch, retain the original
repository and use a separate copy; move only the corresponding output SQLite
and runner state files aside in that copy before running. Protocol, source,
trained-bank and upstream audit inputs are hash-pinned: do not edit them.
`freeze` commands are historical protocol creation commands and must not be
used to regenerate a different protocol under an existing result name.
Full reruns are much more expensive than saved-output verification.

## Measured inputs

Original data: J. Rapp et al., *High-flux single-photon lidar*, Optica 8,
30–39 (2021), https://doi.org/10.1364/OPTICA.403190.
The source authors provide their data-download link in
https://github.com/Goyal-STIR-Group/HighFluxSPL.

Download the source archive from that repository and extract these files to
`data/highflux_raw/2019_01_28/`:

- `FGS_td048_2019_01_28_acq1.mat` (OD3, low flux)
- `FGS_td048_2019_01_28_acq2.mat` (OD0, high flux)
- `laser_calib_100.mat`

`MEASURED_RECOVERY_PROTOCOL_20261107.json` gives the exact SHA-256 digests and
calibration/development segment selection.
`MEASURED_CLUSTER_PROTOCOL_20261107.json` gives the fresh evaluation segment.
Raw third-party MAT files are not redistributed. The saved fitted parameters,
gap/prediction arrays and diagnostics needed to regenerate the reported
measurement figure and table are included.

After obtaining and verifying the inputs, the measurement entrypoints are:

```sh
python tools/measurement_recovery_study.py run
python tools/measurement_recovery_predictive.py
python tools/measured_cluster_extension.py run
python tools/audit_measurement_recovery.py
python tools/audit_measured_results.py
python tools/audit_measured_cluster.py
```

The recovery-only development fit motivated the clustering component; only the
fresh disjoint evaluation segment is used for its reported predictive result.

## Source map

| Component | Main files |
| --- | --- |
| Serial-recovery likelihood and derivatives | `src/ct_dual_reverse.py`, `src/ct_dual_fast.py` |
| Known-active exposure and pulse fitting | `src/deadtime_exposure.py`, `src/fast_two_return_fit.py` |
| Independent score normalization | `src/null_information_bank.py`, `src/null_bank_score.py` |
| Main detection/localization cohort | `src/score_peak_localization_development.py` |
| Matched independent censored banks | `tools/revision_censored_bank.py`, `tools/revision_exposure_bank.py` |
| Full-likelihood-ratio reference | `src/localization_postfailure_full_joint_diagnosis.py` |
| Power and model mismatch | `tools/revision_sensitivity.py` |
| Measured recovery and clustering | `tools/measurement_recovery_model.py`, `tools/measured_cluster_extension.py` |
| Figures and numerical tables | `tools/build_revision_article.py`, `tools/build_measurement_assets.py` |

Frozen source and protocol names retain their original development identifiers
to preserve provenance and hash checks. Only the results selected in the
article's data files are used for the reported comparisons; other preserved
modules document the numerical dependency graph.

## Boundaries and availability

Calibration uses 199 null draws, not 999. The data do not establish performance
at exactly matched population false-alarm rates. The original fixed-recovery bank remains unchanged. A separate calibrated extension trains sensor-specific banks and thresholds with variable recovery and timestamp jitter.
No physical two-surface experiment or superiority over CASPI is claimed.

This repository accompanies a manuscript prepared for submission; it is not
a claim of journal acceptance. Contact: emre.levent@fbu.edu.tr.

## Calibrated recovery and measured-path injection

The extension integrates a truncated-normal electronics recovery distribution
inside each transition density. With jitter it uses an integrated pair criterion,
not the exact joint likelihood of the noisy record. Complete noisy paths are
simulated for calibration so adjacent gaps retain their shared endpoint errors.
Independent 3000-period single-surface records fit recovery in each mismatch
condition; detector recovery stays nominal and jitter is an instrument input.

Each sensor has 64 independent matrix-training acquisitions and 499 separate
null calibration acquisitions. These provide cohort-level calibration over the
specified nuisance population, not the original query-specific 199-draw bootstrap.
The nine mismatch conditions reuse all original 864 acquisitions. Both score
methods share the calibration paths and use their own thresholds.

The high-flux measured study uses the measured pulse and independent recovery
fit. It tests 192 simulated positives,128 simulated nulls,and128 disjoint measured
single-surface blocks. A prospectively extended measured-null evaluation uses1024
blocks,including the first128,and gives 6/1024 calibrated-detector alarms.
The semi-synthetic study completes unrecorded avalanches and suppressed incident
photons conditional on64 measured blocks,including the terminal no-record interval.
All64 source traces are reproduced exactly at zero injection. An independent
secondary photon stream is then added before both recovery filters. At a12.5%
amplitude ratio the injected-secondary correct counts are 26/64 and
7/64 for the calibrated and censored methods. Ratios reuse the same
source blocks; their results are not independent pooled observations.

The terminal-complete version2 is the reported injection study. Version1 and its
protocol are retained as an earlier boundary-handling diagnostic. The amendment
changes only completion after the last measured record; all source blocks,
injection strengths,seeds and estimators stay fixed. Raw third-party MAT files
are still obtained from the source authors and are not redistributed here.

Saved-output checks and the extension figure/tables:

```sh
python tools/audit_calibrated_recovery.py
python tools/audit_semisynthetic_recovery.py
python tools/audit_extended_measured_nulls.py
python tools/build_calibrated_recovery_assets.py
```

After obtaining the measured inputs,the original Windows study entrypoints are:

```sh
python tools/calibrated_recovery_study.py calibrate
python tools/calibrated_recovery_study.py train
python tools/calibrated_recovery_study.py threshold
python tools/calibrated_recovery_study.py evaluate
python tools/calibrated_recovery_study.py public
python tools/analyze_calibrated_recovery_complete.py
python tools/semisynthetic_recovery_tail.py run
python tools/semisynthetic_recovery_tail.py analyze
python tools/extended_measured_nulls.py run
python tools/extended_measured_nulls.py analyze
```

Retain the supplied frozen protocols. Do not run `freeze` again. Completed
sensor fits are retained; simulation runners resume missing records. A fresh
recomputation requires a separate checkout with the relevant outputs moved
aside. The new source hashes are checked byte-for-byte; `.gitattributes` preserves
them. Full study reruns are expensive; saved-output verification needs no raw
measurements. Recovery-fit uncertainty,unknown jitter,and physical two-surface
validation remain distinct from these calibrated experiments.
