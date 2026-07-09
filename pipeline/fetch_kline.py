"""Compatibility wrapper for market data fetching."""

from market.fetch_kline import *  # noqa: F401,F403

if __name__ == "__main__":
    from market.fetch_kline import main

    main()
