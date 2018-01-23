# test the line search (and the gradient computation) 
import unittest
from gradient import get_gradient
import linesearch
from GradientDescent import gradient_descent
import numpy as np 
import sympy

class TestGradient(unittest.TestCase):
	def test_get_gradient(self):
		x, y = sympy.symbols('x y')
		expr = 100 * (y - x**2)**2 + (1 - x)**2
		# whose gradient is (-400xy+400x^3+2x-2, 200(y-x^2))
		position1 = np.array([0., 0.])
		# the value should be (-2, 0) 
		self.assertTrue(np.allclose(np.array([-2,0]), get_gradient(expr, position1)))
		position2 = np.array([1., 1.])
		# the value should be (0, 0)
		self.assertTrue(np.allclose(np.array([0.,0.]), get_gradient(expr, position2)))

class TestLinesearch(unittest.TestCase):
	def test_linesearch(self):
		x, y = sympy.symbols('x y')
		expr = 100 * (y - x**2)**2 + (1 - x)**2
		start_pos = np.array([-100., -30.])
		direction = np.array([1, 1])
		alpha_bar = 10.
		rho = 0.4
		sigma = 0.6
		alpha = linesearch.line_search(expr, start_pos, direction, alpha_bar, rho, sigma)
		print 'line search, step length', alpha

class TestGradientSearch(unittest.TestCase):
	def test_gradient_descent(self):
		x, y = sympy.symbols('x y')
		expr = 100 * (y - x**2)**2 + (1 - x)**2		
		start_pos = np.array([-100, -30])
		epsilon = 10e-4
		result = gradient_descent(expr, start_pos, epsilon, alpha_bar=10, rho=0.4, sigma=0.7)
		print 'gradient descent, result: ', result


if __name__ == '__main__':
	unittest.main()