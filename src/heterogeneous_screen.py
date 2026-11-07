"""Single-report DEVELOPMENT audit on heterogeneous scenes.

All scene families and seeds in this script remain development data. This
script does not create a held-out test merely by assigning new random seeds.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
from pilot import simulate
from detector_suite import suite, evaluate_report, METHOD_NAMES

ROOT = Path(__file__).resolve().parents[1]
SCENES = ('slope', 'secondary_step', 'crossing', 'pulse_mismatch',
          'hole', 'high_background', 'close', 'pileup')


def scene_spec(name, size=24):
    y, x = np.mgrid[:size, :size]
    primary = np.full((size, size), 80.)
    target = primary + 12.
    mask = (x >= size//4) & (x < 3*size//4) & (y >= size//4) & (y < 3*size//4)
    pars = dict(size=size, primary=30., secondary=3., background=20., pulse_fwhm=5.)
    if name == 'slope':
        primary = 77.3 + .22*x + .12*y
        target = primary + 12.4
    elif name == 'secondary_step':
        target = primary + np.where(x < size//2, 11.3, 18.7)
    elif name == 'crossing':
        target = primary + 4. + .7*x
    elif name == 'pulse_mismatch':
        pars['pulse_fwhm'] = 6.5
    elif name == 'hole':
        mask[:] = True
        mask[size//2-2:size//2+2, size//2-2:size//2+2] = False
    elif name == 'high_background':
        pars['background'] = 200.
    elif name == 'close':
        target = primary + 6.
    elif name == 'pileup':
        pars.update(primary=600., secondary=30., background=100.)
    else:
        raise ValueError(name)
    pars.update(primary_delay_map=primary, secondary_delay_map=target, secondary_mask=mask)
    # Radius four proposal support is entirely inside the image for scored pixels.
    interior = np.zeros((size, size), bool)
    interior[5:-5, 5:-5] = True
    return pars, mask, target, interior


def source_hashes():
    names = ('heterogeneous_screen.py', 'detector_suite.py', 'pilot.py',
             'methods.py', 'crossfit.py', 'joint_detector.py', 'screen.py')
    return {n: hashlib.sha256((ROOT/'src'/n).read_bytes()).hexdigest() for n in names}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--calibration-count', type=int, default=20)
    p.add_argument('--evaluation-count', type=int, default=40)
    p.add_argument('--scenes', nargs='+', choices=SCENES, default=list(SCENES))
    args = p.parse_args()
    outdir = ROOT/'results'/'heterogeneous_cache'
    outdir.mkdir(exist_ok=True)
    result = dict(status='development only; single report per pixel',
                  calibration_rule='max of separate null and mixed-context 99th percentiles',
                  calibration_seeds=[8000, 8000+args.calibration_count-1],
                  evaluation_seeds=[9000, 9000+args.evaluation_count-1],
                  code_sha256=source_hashes(), details={})
    start = time.perf_counter()
    for scene in args.scenes:
        pars, mask, target, interior = scene_spec(scene)
        calibration = {m: {'null': [], 'mixed': []} for m in METHOD_NAMES}
        records = []
        timings = []
        for split, first, count in [('calibration', 8000, args.calibration_count),
                                    ('evaluation', 9000, args.evaluation_count)]:
            for seed in range(first, first+count):
                for null in (True, False):
                    cfg = dict(pars)
                    if null:
                        cfg['secondary'] = 0.
                    h, _, _ = simulate(seed, **cfg)
                    begin = time.perf_counter()
                    outputs = suite(h)
                    timings.append(time.perf_counter()-begin)
                    path = outdir/f'{scene}_{seed}_{"null" if null else "mixed"}.npz'
                    arrays = {f'{m}_score': outputs[m][0] for m in METHOD_NAMES}
                    arrays.update({f'{m}_delay': outputs[m][1] for m in METHOD_NAMES})
                    np.savez_compressed(path, **arrays)
                    if split == 'calibration':
                        context = 'null' if null else 'mixed'
                        negative = interior if null else (~mask & interior)
                        for m in METHOD_NAMES:
                            calibration[m][context].extend(outputs[m][0][negative].ravel())
                    else:
                        records.append((seed, null, path))
            print(scene, split, 'complete', flush=True)
        scene_result = dict(primary_centers=pars['primary_delay_map'].tolist(),
                            secondary_centers=target.tolist(), mask=mask.tolist(),
                            interior=interior.tolist(), methods={},
                            suite_seconds_mean=float(np.mean(timings)))
        for m in METHOD_NAMES:
            quantiles = {}
            for context, values in calibration[m].items():
                vals = np.asarray(values)
                # The expected tail order statistic can be -inf when no proposal
                # is valid. In that case a threshold of -inf still accepts all
                # finite valid proposals, while invalid scores remain rejected.
                idx = int(np.ceil(.99*(len(vals)-1)))
                quantiles[context] = float(np.sort(vals)[idx])
            threshold = max(quantiles.values())
            entries = []
            for seed, null, path in records:
                with np.load(path) as d:
                    metrics = evaluate_report(d[f'{m}_score'], d[f'{m}_delay'], threshold,
                                              np.zeros_like(mask) if null else mask,
                                              target, interior)
                entries.append(dict(seed=seed, null=null, **metrics))
            keys = ('recall', 'wrong_depth_fp', 'absent_pixel_fp')
            mixed = {k: float(np.mean([e[k] for e in entries if not e['null'] and k in e]))
                     for k in keys if any(not e['null'] and k in e for e in entries)}
            mixed['null_fp'] = float(np.mean([e['absent_pixel_fp'] for e in entries if e['null']]))
            scene_result['methods'][m] = dict(threshold=threshold, context_quantiles=quantiles,
                                             metrics=mixed, scene_metrics=entries)
        result['details'][scene] = scene_result
        result['elapsed_seconds'] = time.perf_counter()-start
        (ROOT/'results'/'heterogeneous_screen_v1.json').write_text(json.dumps(result, indent=2))
        print(scene, {m: r['metrics'] for m, r in scene_result['methods'].items()}, flush=True)


if __name__ == '__main__':
    main()
