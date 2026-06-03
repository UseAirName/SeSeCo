def lambda_t(t, lambda_ts_val):
    for ts, alpha_ts, lambda_ts in lambda_ts_val:
        if t <= ts:
            return lambda_ts


def mean_shift_latent(t, mean_shift_val):
    for ts, mean_shift_ts in mean_shift_val:
        if t <= ts:
            return mean_shift_ts


def std_shift_latent(t, std_shift_val):
    for ts, std_shift_ts in std_shift_val:
        if t <= ts:
            return std_shift_ts
