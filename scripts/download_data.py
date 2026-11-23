"""Thin wrapper: ``python scripts/download_data.py --data-dir sitesight_data``."""

from sitesight.data.download import main

if __name__ == "__main__":
    raise SystemExit(main())
