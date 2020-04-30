from scipy.special import comb


def get_bernstein_polynomial(n, v, x):
    coeff = comb(n, v)
    weights = coeff * ((1 - x) ** (n - v)) * (x ** v)
    return weights
