def cumulative_returns(returns: list) -> list:
    """
    Returns the compounded cumulative return after every period.
    """
    W_prev = 1.0
    W_t = 0.0
    R_t = []

    for i, r in enumerate(returns):
        W_t = W_prev*(1 + r) 
        R_t.append( W_t - 1 )
        W_prev = W_t

    return R_t
        