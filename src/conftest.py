"""Route the backend-specific runner integration test to its real environment.

The general environment intentionally lacks Numba;scientific kernels have a
tested Python fallback. The frozen high-flux runner itself requires Numba,
and its integration test is run in the dedicated .venv_ded test invocation.
No scientific backend is stubbed and no outcome-dependent test is excluded.
"""
from importlib.util import find_spec

_MISSING_NUMBA=find_spec('numba') is None


def pytest_ignore_collect(collection_path,config):
    if _MISSING_NUMBA and collection_path.name=='test_one_step_location_development.py':return True
    return None


def pytest_terminal_summary(terminalreporter,exitstatus,config):
    if _MISSING_NUMBA:
        terminalreporter.write_line('Backend routing: one frozen runner integration module requires Numba; run it in .venv_ded (included in the compiled-path suite).')
