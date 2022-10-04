import unittest
from belief_propagation.variable_node import variable_node_function


class TestVariableNodeFunction(unittest.TestCase):
    def test_varaible_node_function_(self):
        y = 4
        sigma = 0.05
        means = [-1.6, 6.6]
        vars = [0.08, 0.16]
        m, v = variable_node_function(y, sigma, [1, 1/2], means, vars)
        print(m, v)




if __name__ == '__main__':
    unittest.main()