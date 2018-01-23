# visulize the test function with the contour and the line search 
# currently for 2d case

from linesearch import line_search
from gradient import get_gradient
import sympy
import numpy as np 
from numpy.linalg import norm as npnorm
import matplotlib.pyplot as plt 
import matplotlib.collections 
import itertools

def view_gradient_descent(expression, position, epsilon, alpha_bar=3, rho=0.4, sigma=0.7):
	"""
	carry out the gradient descent method to find the minimal point of the given function
	para::expression: the expression of the function
	para::position: the initial search point
	para::epsilon: the termination error
	para::alpha_bar: the initial length of the line search
	para::rho: parameter in line search, in (0, 1/2)
	para::sigma: parameter in line search, in (rho, 1)
	..note: to use this method, you'd better construct the expression in this way:
		claim all the variables
		write the expression

		for example:
		>>> import sympy 
		>>> import numpy as np
		>>> x, y, z = sympy.symbols('x y z')
		>>> expr = sympy.cos(x) + sympy.exp(y*z)
	"""
	# convert the function for show
	show_func = sympy.lambdify(expression.free_symbols, expression, 'numpy')
	# the new points in ech iteration
	show_points = [position.tolist()]
	iter_time = 0
	while True:
		expr_gradient = get_gradient(expression, position)
		if npnorm(expr_gradient) < epsilon:
			break
		direction = -expr_gradient
		# get the step length by line search
		alpha = line_search(expression, position, direction, alpha_bar, rho, sigma)
		position = position + alpha * direction
		show_points.append(position.tolist())
		iter_time += 1
	print 'iterations: ', iter_time
	print 'point: ', show_points[-1]
	# draw the contour and line segments between the iterate points
	fig = plt.figure()
	ccx = np.linspace(-50, 50, 1000)
	ccy = np.linspace(-50, 50, 1000)
	X,Y = np.meshgrid(ccx, ccy)
	Z = show_func(X,Y)
	plt.contour(X,Y,Z, colors='black')
	"""
	for k in range(len(show_points) -1):
		plt.plot((show_points[k], show_points[k+1]),
			color='brown', marker='o')
	"""
	# draw the lines
	segs = [[k, k+1] for k in range(len(show_points)-1)]
	lines = [[tuple(show_points[j]) for j in i] for i in segs]
	lc = matplotlib.collections.LineCollection(lines)
	ax = fig.add_subplot(111)
	ax.add_collection(lc)
	plt.xlim([-50,50])
	plt.ylim([-50,50])
	plt.show()



x, y = sympy.symbols('x y')
expr = 100 * (y - x**2)**2 + (1 - x)**2
start_pos = np.array([-50,-50])
epsilon = 10e-5
view_gradient_descent(expr, start_pos, epsilon)