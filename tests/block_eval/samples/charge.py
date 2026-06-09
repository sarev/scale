import time


def charge(card, amount_pennies):
    """Charge a card via the gateway."""
    if amount_pennies <= 0:
        raise ValueError("amount must be positive")
    token = gateway.tokenise(card)
    resp = gateway.charge(token, amount_pennies)
    delay = 0.5
    for attempt in range(3):
        if resp.ok:
            break
        time.sleep(delay)
        delay *= 2
        resp = gateway.charge(token, amount_pennies)
    return resp.receipt_id
