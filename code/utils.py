import math

from code.variable_node import E


def coef_two_gaussian_multiplication(c_new, c_old, m_new, m_old, v_new, v_old):
    c_k = c_old * c_new * (E ** (-1 * ((m_old - m_new) ** 2) / (2 * v_old + 2 * v_new)))
    return c_k / math.sqrt(2 * math.pi * (v_old + v_new))


def mean_two_gaussian_multiplication(m_new, m_old, v_k, v_new, v_old):
    return v_k * (m_old / v_old + m_new / v_new)


def variance_two_gaussian_multiplication(v_new, v_old):
    return 1 / (1 / v_old + 1 / v_new)