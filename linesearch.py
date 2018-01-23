# for line search
# based on Wolfe-Powell condition, inexact method
# main reference: the lecture notes of Operation Research, 2017, by Professor Yang 
import numpy as np 
import sympy
from gradient import get_gradient

def derivate_in_search(expression, direction, position):
	"""
	handle the derivate operation in the line search
	para::expression: the expression of a function
	para::direction: the direction of the line search, array like object(python list, numpy 1d array, etc)
	para::position: array like object(python list, numpy 1d array, etc)
	return::derivate: the numerical value 
	background:
		we hope the find the minimum of function
		phi(alpha) = f(x_k + alpha * direction_k)
		which derivate is 
		phi'(alpha) = gradient(f)(x_k) * direction_k 
	"""
	expr_gradient = get_gradient(expression, position)
	derivate = np.dot(expr_gradient, direction)
	return derivate

def evaluate(expression, position):
	"""
	evaluate the expression using the given position
	"""
	variables = list(expression.free_symbols)
	value = [(variables[k], position[k]) for k in range(len(position))]
	return expression.subs(value)

def step1(expression, position, direction, para):
	"""
	to get/reset a suitable value of alpha for step 2
	return::para: the modified parameters 
	"""
	phi = evaluate(expression, position + direction * para['alpha'])
	while phi > para['phi0'] + para['rho'] * para['alpha'] * para['phi0_prime']:
		# update a2
		# construct a interpolation and reset alpha
		para['a2'] = para['alpha']
		para['alpha'] = para['a1'] + 0.5 * (para['a1'] - para['alpha'])**2 * para['phi1_prime'] / ((para['phi1'] -
			phi) - (para['a1'] - para['alpha']) * para['phi1_prime'])
		# recompute phi
		phi = evaluate(expression, position + direction * para['alpha'])
	# update the parameters
	para['phi'] = phi
	return para

 
def line_search(expression, position, direction, alpha_bar, rho, sigma):
	"""
	carry out the line search(based on Wolf-Powell condition, inexact)
	and return the step length under the given direction
	para::expression: the expression of a function
	para::position: the start point
	para::direction: the direction of the line search, array like object(python list, numpy 1d array, etc)
	para::alpha_bar: the initial value of right endpoint of the search interval
	..note: the initial value of left endpoint is set as 0
	para::rho: parameter, in (0, 1/2)
	para::sigma: parameter, in (rho, 1)
	return::step_length: the step length under the direction
	"""
	# check the validity of the given parameter
	assert alpha_bar > 0
	assert 0. < rho < 0.5
	assert rho < sigma < 1.
	# initialize the parameters
	phi0 = evaluate(expression, position)
	phi0_prime = derivate_in_search(expression, direction, position)
	# use the dict to record the parameters; here just set alpha as the init right endpoint value
	parameters = {'a1': 0, 'a2': alpha_bar, 'phi0': phi0, 'phi0_prime': phi0_prime,
	 'phi1': phi0, 'phi1_prime': phi0_prime, 'alpha': alpha_bar, 'rho': rho, 'sigma': sigma, 'phi': 0}
	# carry out step 1 at first to find a suitable value for alpha to start the loop of step 2
	parameters = step1(expression, position, direction, parameters)
	# the main loop for step 2
	while True:
		phi_prime = derivate_in_search(expression, direction, position + parameters['alpha'] * direction)
		# meet the stop condition
		if phi_prime >= parameters['sigma'] * parameters['phi0_prime']:
			return parameters['alpha']
		# if not meet the stop condition, reset values
		alpha_temp = parameters['alpha'] - (parameters['a1'] - parameters['alpha']) * phi_prime / (
			parameters['phi1_prime'] - phi_prime)
		parameters['a1'] = parameters['alpha']
		parameters['alpha'] = alpha_temp
		parameters['phi1'] = parameters['phi']
		parameters['phi1_prime'] = phi_prime
		parameters = step1(expression, position, direction, parameters)
