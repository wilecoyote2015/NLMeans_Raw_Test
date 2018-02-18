import numpy as np

def ascombe_transform(value):
    return 2 * np.sqrt(value + 3/8)

def inverse_ascombe_transform(value):
    term_1 = value / 4
    term_2 = 1 / 8
    term_3 = np.sqrt(3/2) / (value * 4)
    term_4 = 11 / 8 * np.power(value, -2)
    term_5 = 5/8 * np.sqrt(3/2) * np.power(value, -3)

    return term_1 - term_2 + term_3 - term_4 + term_5

def ascombe_transform_scale(value, alpha, beta):
    return ascombe_transform((value - beta) / alpha)

def inverse_ascombe_transform_scale(value, alpha, beta):
    return alpha * inverse_ascombe_transform(value) + beta