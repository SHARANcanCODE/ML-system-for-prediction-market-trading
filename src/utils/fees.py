import numpy as np

def polymarket_fee(price: float, rate: float = 0.0175) -> float:
    p = np.clip(price, 0.01, 0.99)
    return p * (1 - p) * rate
