# this is for computing the gradient
# use sympy for symbol computation
# and use numpy to express the gradient as a vector
import sympy
import numpy as np 

def get_gradient(expression, position):
	"""
	given an expression, return the gradient at the given position 
	para::expression: the expression to evaluate the gradient
	para::position: array like object(python list, numpy 1d array, etc)
	return::gradient: numpy array
	..note: to use this method, you'd better construct the expression in this way:
		claim all the variables
		write the expression

		for example:
		>>> import sympy 
		>>> import numpy as np
		>>> x, y, z = sympy.symbols('x y z')
		>>> expr = sympy.cos(x) + sympy.exp(y*z)
		>>> position = np.array([1., 2., 3.])
		>>> get_gradient(expr, position)
		array([ -8.41470985e-01,   1.21028638e+03,   8.06857587e+02])

	"""
	# initialize the output
	length = len(position)
	gradient = np.zeros(length)
	# get the variables
	variables = list(expression.free_symbols)
	# construct the evaluation list
	value = [(variables[k], position[k]) for k in range(length)]
	for k in range(length):
		gradient[k] = expression.diff(variables[k]).subs(value)
	return gradient