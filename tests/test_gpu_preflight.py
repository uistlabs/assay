"""assay.job.assert_gpu_available - the fail-fast GPU guard.

Added after the CUDA-804 metal RCA: on the nvidia/cuda base, cuda-compat raised CUDA
error 804 on a consumer 5090, torch reported zero devices, and llm-compressor
SILENTLY fell back to CPU - 35 min of paid CPU calibration on a rented pod. The
guard must fail LOUD instead. torch is faked here so the paths are deterministic on
any host (CI has no GPU; the dev box's 1070 can't run the Blackwell wheel)."""

import sys

import pytest

from assay.job import assert_gpu_available


class _FakeTensor:
    """Faithful enough for `(torch.ones(8, device=...) * 2).sum().item()`."""
    def __init__(self, data):
        self.data = data

    def __mul__(self, k):
        return _FakeTensor([x * k for x in self.data])

    def sum(self):
        return _FakeTensor([sum(self.data)])

    def item(self):
        return self.data[0]


class _FakeCuda:
    def __init__(self, *, init_exc=None, count=1, name="FakeGPU"):
        self._init_exc = init_exc
        self._count = count
        self._name = name

    def init(self):
        if self._init_exc is not None:
            raise self._init_exc

    def device_count(self):
        return self._count

    def get_device_name(self, _i):
        return self._name


class _FakeTorch:
    def __init__(self, cuda, ones_value=1.0):
        self.cuda = cuda
        self._ones_value = ones_value

    def ones(self, n, device=None):
        return _FakeTensor([self._ones_value] * n)


def _install_torch(monkeypatch, fake):
    # assert_gpu_available does `import torch` internally -> resolves via sys.modules.
    monkeypatch.setitem(sys.modules, "torch", fake)


def test_passes_when_gpu_present_and_probe_correct(monkeypatch, capsys):
    _install_torch(monkeypatch, _FakeTorch(_FakeCuda(count=1)))
    assert_gpu_available()  # ones(8)*2 -> sum 16.0 == expected; must not raise
    assert "GPU preflight OK" in capsys.readouterr().out


def test_raises_when_cuda_init_errors(monkeypatch):
    # This is the exact CUDA-804 failure: init raises Error 804; is_available()
    # would have swallowed it, so we call init() to re-raise and die loud.
    boom = RuntimeError("CUDA error 804: forward compatibility was attempted")
    _install_torch(monkeypatch, _FakeTorch(_FakeCuda(init_exc=boom)))
    with pytest.raises(SystemExit) as ei:
        assert_gpu_available()
    assert "no usable CUDA GPU" in str(ei.value)
    assert "804" in str(ei.value)  # the real cause is surfaced, not masked


def test_raises_when_zero_devices(monkeypatch):
    _install_torch(monkeypatch, _FakeTorch(_FakeCuda(count=0)))
    with pytest.raises(SystemExit) as ei:
        assert_gpu_available()
    assert "no usable CUDA GPU" in str(ei.value)


def test_raises_when_probe_kernel_wrong(monkeypatch):
    # Device enumerates but the wheel ships no kernel for this SM -> the probe
    # returns a wrong value (here forced via ones_value) -> must fail, not proceed.
    _install_torch(monkeypatch, _FakeTorch(_FakeCuda(count=1), ones_value=0.0))
    with pytest.raises(SystemExit) as ei:
        assert_gpu_available()
    assert "no usable CUDA GPU" in str(ei.value)
