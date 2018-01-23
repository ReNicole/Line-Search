# main loop of the gradient descent

from linesearch import line_search
from gradient import get_gradient
import sympy
import numpy as np 
from numpy.linalg import norm as npnorm

def gradient_descent(expression, position, epsilon, alpha_bar=10, rho=0.4, sigma=0.7, 
	show_iter=True):
	"""
	carry out the gradient descent method to find the minimal point of the given function
	para::expression: the expression of the function
	para::position: the initial search point
	para::epsilon: the termination error
	para::alpha_bar: the initial length of the line search
	para::rho: parameter in line search, in (0, 1/2)
	para::sigma: parameter in line search, in (rho, 1)
	para::show_iter: if True, output the iteration times
	return::position: the minimal point for the given function
	return::iter_time(optional): the total times of iteration
	..note: to use this method, you'd better construct the expression in this way:
		claim all the variables
		write the expression

		for example:
		>>> import sympy 
		>>> import numpy as np
		>>> x, y, z = sympy.symbols('x y z')
		>>> expr = sympy.cos(x) + sympy.exp(y*z)
	"""
	iter_time = 0
	while True:
		expr_gradient = get_gradient(expression, position)
		if npnorm(expr_gradient) < epsilon:
			if show_iter:
				return position, iter_time
			else:
				return position
		direction = -expr_gradient
		# get the step length by line search
		alpha = line_search(expression, position, direction, alpha_bar, rho, sigma)
		position = position + alpha * direction
		iter_time += 1