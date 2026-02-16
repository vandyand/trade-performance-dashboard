#!/usr/bin/env python3
"""Refresh Parquet cache from JSONL trading logs.

Intended to be run via systemd timer every 5 minutes.

Usage:
    python update_cache.py           # incremental update
    python update_cache.py --full    # full rebuild from scratch
"""

import argparse
import logging
import sys

from cache_manager import refresh_all


def main():
    parser = argparse.ArgumentParser(
        description="Refresh Parquet cache from JSONL trading logs")
    parser.add_argument(
        "--full", action="store_true",
        help="Full rebuild (ignore saved byte offsets)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    try:
        refresh_all(full_rebuild=args.full)
        logging.info("Cache refresh completed successfully")
    except Exception:
        logging.exception("Cache refresh failed")
        sys.exit(1)

    # After cache refresh, write JSON data files and push to git
    try:
        from push_to_vercel import push_system, _git_push
        for system in ["oanda", "alpaca", "solana"]:
            push_system(system)
        _git_push()
        logging.info("Data push completed successfully")
    except Exception as e:
        logging.warning("Data push failed (non-fatal): %s", e)


if __name__ == "__main__":
    main()
