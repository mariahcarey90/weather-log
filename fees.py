"""Polymarket taker fees on a buy (29 Oct 2026 audit, finding 11).

fee = shares * rate * (p * (1 - p)) ** exponent, taken in shares on a buy, so a stake buys
fewer shares than stake / p. A market's own schedule comes from fetch.py (bucket "fee");
when a market doesn't report one, the documented weather rate is assumed, which errs
toward costing paper trades too much rather than too little.
"""
DEFAULT_RATE = 0.05   # Polymarket's documented taker rate for the Weather category


def _rate_exp(fee):
    if not fee:
        return DEFAULT_RATE, 1.0
    return float(fee.get("rate") or 0.0), float(fee.get("exponent") or 1.0)


def net_shares(stake, price, fee=None):
    """Shares received for `stake` dollars at average price `price`, after the fee."""
    rate, e = _rate_exp(fee)
    shares = stake / price
    return shares - shares * rate * (price * (1 - price)) ** e / price


def effective_price(price, fee=None):
    """What one share really costs after the fee: stake / net shares."""
    return 1.0 / net_shares(1.0, price, fee)
