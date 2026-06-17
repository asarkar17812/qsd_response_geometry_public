"""Minimal standalone shim.

The analysis scripts in this folder were extracted from a larger project where `config` provided I/O
helpers and the calibrated reduced-core generator. This shim reproduces only what those scripts use,
so the folder runs without the parent project. Outputs are written under code/output/.
"""
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

# Reduced-core killed generator (5x5) at the calibrated operating point, used by
# infinite_dim_qsd_kernel.py for the finite-truncation cross-check of the group-inverse identity.
# Pinned here as data so this folder is self-contained.
_QT_5x5 = np.array([
    [-0.27803991933073646, 0.27803991933073646, 0.0, 0.0, 0.0],
    [0.0, -0.9769027638697216, 0.6013995361044441, 0.3753021245652775, 0.0],
    [1.05017776332411, 0.476991533444471, -1.527169296768581, 0.0, 0.0],
    [0.7001185088827402, 0.23568993417256212, 0.0, -0.9358084430553023, 0.0],
    [0.04, 0.0, 0.0, 0.0, -0.04],
])


class _QSD:
    QT = _QT_5x5


def reduced_core_params():
    return None


def qsd_at(params=None, **kwargs):
    return _QSD()


def output_path(*parts, kind="output"):
    base = os.path.join(_HERE, kind)
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, *parts) if parts else base
