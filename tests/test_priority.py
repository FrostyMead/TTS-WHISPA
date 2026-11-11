import sys

import pytest


@pytest.mark.skipif(sys.platform != "win32", reason="Windows priority classes")
def test_raise_priority_takes_effect():
    import ctypes
    from ctypes import wintypes

    from whispa.app import raise_priority

    raise_priority()
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.GetPriorityClass.argtypes = [wintypes.HANDLE]
    assert k32.GetPriorityClass(k32.GetCurrentProcess()) == 0x8000  # ABOVE_NORMAL
