"""Trace actual input reads, copy their closure, and verify it in isolation.

No efficacy study is rerun and no live database is copied. The isolated root
contains owned source/protocol/trained-bank inputs needed by the frozen graph.
Numerical fixtures check the real compiled backend in that isolated root.
"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'output/isolated_localization_graph_v1'
seen = set()
capture = True


def audit(event, args):
    if event != 'open' or not capture or not args or not isinstance(args[0], (str, bytes)):
        return
    try:
        path = Path(args[0]).resolve()
        relative = path.relative_to(ROOT)
    except (ValueError, TypeError, OSError):
        return
    if any(part.startswith('.venv') or part == '__pycache__' for part in relative.parts):
        return
    if path.suffix.lower() in ['.py', '.json', '.npz', '.txt', '.tex'] and path.is_file():
        seen.add(path)


def main():
    global capture
    if DEST.exists():
        raise FileExistsError('Portable-closure fixture already exists; preserve it')
    sys.addaudithook(audit)
    sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
    import localization_full_joint_control as control
    import localization_full_joint_integrity_v2
    import localization_full_joint_failure_audit_v2
    import score_peak_localization_integrity
    import score_peak_localization_parent_failure_audit
    import score_peak_localization_failure_audit
    import benchmark_localization_test_cost as cost
    control.check()
    cost.check()
    for module in list(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            try:
                path = Path(filename).resolve()
                relative = path.relative_to(ROOT)
                if path.suffix == '.py' and not any(p.startswith('.venv') for p in relative.parts):
                    seen.add(path)
            except (ValueError, OSError):
                pass
    fixtures = ['test_score_peak_localization.py', 'test_null_bank_score.py',
                'test_full_information_joint_reference_v2.py',
                'test_null_bank_equal_information_audits_v2.py', 'test_ct_dual_reverse.py',
                'null_bank_control_reference.py']
    seen.update(ROOT / 'src' / name for name in fixtures)
    capture = False
    assert not any(p.suffix == '.sqlite3' for p in seen)
    digests = {}
    for path in sorted(seen):
        relative = path.relative_to(ROOT)
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
        digests[relative.as_posix()] = digest
    (DEST / 'output').mkdir(exist_ok=True)
    manifest = dict(status='COPIED_ACTUAL_SOURCE_PROTOCOL_BANK_INPUT_CLOSURE', files=digests,
                    scope='No live database or upstream raw photon data. Frozen graph and real '
                          'numerical fixtures only, not all efficacy computations or submission readiness.')
    (DEST / 'portable_input_manifest.json').write_text(json.dumps(manifest, indent=2))
    graph_code = ('import sys,pathlib; sys.path.insert(0,"src"); '
                  'import localization_full_joint_control as c; c.check(); '
                  'assert c.parent.ROOT.resolve()==pathlib.Path.cwd().resolve(); '
                  'print("Isolated frozen source/protocol/trained-bank graph PASS")')
    graph = subprocess.run([sys.executable, '-c', graph_code], cwd=DEST, capture_output=True, text=True,
                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    (DEST / 'output/graph_check.log').write_text(graph.stdout + graph.stderr)
    assert graph.returncode == 0, graph.stderr
    print(graph.stdout, end='', flush=True)
    tests = subprocess.run([sys.executable, '-m', 'pytest', *['src/' + name for name in fixtures if name.startswith('test_')], '-q'],
                           cwd=DEST, capture_output=True, text=True,
                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    (DEST / 'output/numerical_fixture_tests.log').write_text(tests.stdout + tests.stderr)
    assert tests.returncode == 0, tests.stdout + tests.stderr
    print(tests.stdout, flush=True)
    result = dict(status='PASS', isolated_files=len(digests), isolated_input_bytes=sum((DEST / n).stat().st_size for n in digests),
                  graph_check_exit=graph.returncode, numerical_fixture_exit=tests.returncode,
                  original_root_mutated=False, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  manifest_sha256=hashlib.sha256((DEST / 'portable_input_manifest.json').read_bytes()).hexdigest(),
                  scope=manifest['scope'])
    (ROOT / 'output/localization_portable_closure_qa.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
