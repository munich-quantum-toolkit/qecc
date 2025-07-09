# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Site customization shim to enable multiprocess coverage collection in tests.

See https://coverage.readthedocs.io/en/latest/subprocess.html.
"""

from __future__ import annotations

try:
    import coverage

    coverage.process_startup()
except ImportError:
    # The 'coverage' module is optional
    # If it is not installed, we do not enable multiprocess coverage collection
    pass
