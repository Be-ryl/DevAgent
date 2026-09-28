"""Smoke tests for the DevAgent package."""

import unittest

import devagent


class PackageImportTest(unittest.TestCase):
    """Verify that the package is importable."""

    def test_import_devagent(self) -> None:
        self.assertEqual(devagent.__name__, "devagent")


if __name__ == "__main__":
    unittest.main()
