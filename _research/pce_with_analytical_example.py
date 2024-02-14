import numpy as np
import sys
import os
import math
import matplotlib
import matplotlib.pyplot as plt
import scipy
import scipy.stats as stats
import sklearn.linear_model as lm

print(sys.path)
matplotlib.use("Qt5Agg")
import time
import pickle


from numpy.polynomial.polynomial import polyval

from analytical_function import *


# Analytical function: 
def analytical_function(params, t=None):
    """
    Function evaluates a set of parameters (params) in the non-linear, non-Gaussian equation for each location
    (t).
    Source:
    Oladyshkin, S., Mohammadi, F., Kroeker I. and Nowak, W. Bayesian active learning for Gaussian process emulator using
        information theory. Entropy, X(X), X, 2020,
    Oladyshkin, S. and Nowak, W. The Connection between Bayesian Inference and Information Theory for Model Selection,
        Information Gain and Experimental Design. Entropy, 21(11), 1081, 2019,

    For function details and reference information, see:
    https://doi.org/10.3390/e21111081

    :param params: <np.array[n_mc, n_params]
                   with n_mc parameter sets
    :param t: <np.array[n_obs, 1]>
                locations "t" where the function is to be evaluates
    :return: <Dict>
            with an <array[n_obs,1] and <np.array[n_mc, n_obs] with results of evaluating function on the n_mc
            parameter sets
    """

    # If vector with parameters is passed, reshape it to matrix
    if params.ndim == 1:
        params = np.reshape(params, (1, params.shape[0]))
    # If only one parameter is sent
    if params.shape[1] == 1:
        params = np.hstack((params, params))

    n_params = params.shape[1]  # number of parameters

    param_sets = params.shape[0]  # number of parameter sets

    term1 = (params[:, 0] ** 2 + params[:, 1] - 1) ** 2
    term2 = params[:, 0] ** 2
    term3 = 0.1 * params[:, 0] * np.exp(params[:, 1])

    # Term that all models have in common:
    term5 = 0
    if n_params > 2:
        for i in range(2, n_params):
            term5 = term5 + np.power(params[:, i], 3) / (i + 1)

    # Sum all non-time-related terms: gives one value per row, and one row for each parameter set
    const_per_set = term1 + term2 + term3 + term5 + 1  # All non-time-related terms

    # Calculate time term: gives one value per row for each time interval
    term4 = np.full((param_sets, t.shape[0]), 0.0)
    for i in range(0, param_sets):
        term4[i, :] = -2 * params[i, 0] * np.sqrt(0.5 * t)

    output = term4 + np.repeat(const_per_set[:, None], t.shape[0], axis=1)

    return output

# 1.
def compute_orthonomal_basis(data, d):
    """
   Construction of Data-driven Orthonormal Polynomial Basis

    Parameters
    ----------
    data : array [MC]
        Raw data.
    d : int
        degree of polynomial expansion

    Returns
    -------
    Polynomial : array [d+2, d+2]
        The coefficients of the orthonormal polynomials.

     Notes:
    * MC is the total number of MonteCarlo samples
    """

    # Initialization
    dd = d+1  # Degree of polynomial for roots definition
    nsamples = len(data)

    # Forward linear transformation (Avoiding numerical issues)
    MeanOfData = np.mean(data)
    VarOfData = np.var(data)
    data = data/MeanOfData

    # Compute raw moments for input data
    M = [np.sum(np.power(data, p)) / nsamples for p in range(2 * dd + 2)]

    # Step-wise
    # for p in range(2 * dd + 2):
    #     print(p)
    #     step1 = np.sum(np.power(data, p))
    #     step2 = step1/nsamples

    # --------------------------------------------------------------------------
    # ----- Main Loop for Polynomial with degree up to dd
    # --------------------------------------------------------------------------
    PolyCoeff_NonNorm = np.empty((0, 1))
    Polynomial = np.zeros((dd+1, dd+1))

    for degree in range(dd+1):
        Mm = np.zeros((degree+1,degree+1))
        Vc = np.zeros((degree+1))

        # Define Moments Matrix Mm
        for i in range(degree+1):
            for j in range(degree+1):
                if i < degree:
                    Mm[i, j] = M[i+j]

                elif (i == degree) and (j == degree):
                    Mm[i, j] = 1

            # Numerical Optimization for Matrix Solver
            Mm[i, :] = Mm[i, :]/max(abs(Mm[i,:]))

        # Definition of Right Hand side orthogonality conditions: Vc
        for i in range(degree+1):
            Vc[i] = 1 if i == degree else 0

        # Solution: Coefficients of Non-Normal Orthogonal Polynomial: Vp Eq.(4)
        try:
            Vp = np.linalg.solve(Mm, Vc)
        except:
            inv_Mm = np.linalg.pinv(Mm)
            Vp = np.dot(inv_Mm,Vc.T)

        # PolyCoeff_NonNorm[degree,0:degree]=Vp #PolyCoeff_NonNorm(degree+1,1:degree+1)=Vp'
        if degree == 0:
            PolyCoeff_NonNorm = np.append(PolyCoeff_NonNorm,Vp)

        if degree !=0:
            if degree == 1:
                zero=[0]
            else:
                zero=np.zeros((degree,1))
            PolyCoeff_NonNorm = np.hstack((PolyCoeff_NonNorm , zero))

            PolyCoeff_NonNorm = np.vstack((PolyCoeff_NonNorm, Vp))

        if 100*abs(sum(abs(np.dot(Mm,Vp)) - abs(Vc))) > 0.5:
            print('\n---> Attention: Computational Error too high !')
            print('\n---> Problem: Convergence of Linear Solver')

        # Original Numerical Normalization of Coefficients with Norm and Ortho-normal Basis computation
        # Matrix Storage Note: Polynomial(i,j) correspond to coefficient number "j-1" of polynomial degree "i-1"
        P_norm = 0
        for i in range(nsamples):
            Poly = 0
            for k in range(degree+1):
                if degree == 0:
                    Poly += PolyCoeff_NonNorm[k] * (data[i] ** k)
                else:
                    Poly += PolyCoeff_NonNorm[degree, k] * (data[i] ** k)

            P_norm += Poly**2 / nsamples

        P_norm = np.sqrt(P_norm)

        for k in range(degree+1):
            if degree == 0:
                Polynomial[degree, k] = PolyCoeff_NonNorm[k]/P_norm
            else:
                Polynomial[degree, k] = PolyCoeff_NonNorm[degree,k]/P_norm

    # Backward linear transformation to the real data space
    data_mod = data * MeanOfData
    for k in range(len(Polynomial)):
        Polynomial[:, k] = Polynomial[:, k]/(MeanOfData**(k))

    return Polynomial


# 2.
def sort_basis_indices(keys, graded=True, reverse=False):
    """
    Sort keys using graded lexicographical ordering. It gives the first dimension a higher importance than subsequent
    dimensions.
    Same as ``numpy.lexsort``, but also support graded and reverse lexicographical ordering.
    Args:
        keys:
            Values to sort.
        graded:
            Graded sorting, meaning the indices are always sorted by the index
            sum. E.g. ``(2, 2, 2)`` has a sum of 6, and will therefore be
            considered larger than both ``(3, 1, 1)`` and ``(1, 1, 3)``.
        reverse:
            Reverse lexicographical sorting meaning that ``(1, 3)`` is
            considered smaller than ``(3, 1)``, instead of the opposite.
    Returns:
        Array of indices that sort the keys along the specified axis.
    Examples:
        >>> indices = np.array([[0, 0, 0, 1, 2, 1],
        ...                        [1, 2, 0, 0, 0, 1]])
        >>> indices[:, np.lexsort(indices)]
        array([[0, 1, 2, 0, 1, 0],
               [0, 0, 0, 1, 1, 2]])
        >>> indices[:, numpoly.glexsort(indices)]
        array([[0, 1, 2, 0, 1, 0],
               [0, 0, 0, 1, 1, 2]])
        >>> indices[:, numpoly.glexsort(indices, reverse=True)]
        array([[0, 0, 0, 1, 1, 2],
               [0, 1, 2, 0, 1, 0]])
        >>> indices[:, numpoly.glexsort(indices, graded=True)]
        array([[0, 1, 0, 2, 1, 0],
               [0, 0, 1, 0, 1, 2]])
        >>> indices[:, numpoly.glexsort(indices, graded=True, reverse=True)]
        array([[0, 0, 1, 0, 1, 2],
               [0, 1, 0, 2, 1, 0]])
        >>> indices = numpy.array([4, 5, 6, 3, 2, 1])
        >>> indices[numpoly.glexsort(indices)]
        array([1, 2, 3, 4, 5, 6])
    """
    keys_ = np.atleast_2d(keys)    # convert to a 2D array
    if reverse:
        keys_ = keys_[::-1]
    # get indices from smallest to largest, giving the 1st row a higher importance
    indices = np.array(np.lexsort(keys_))
    if graded:
        indices = indices[np.argsort(
            np.sum(keys_[:, indices], axis=0))].T
    return indices


def get_polynomial_basis(max_degree, ndim):
    """
    Get the order combinations for all dimensions
    :param max_degree: int
        Max polynomial degree
    :param ndim: int
        Number of parameters
    :return:
    """
    # Arrays with smallest order (0) to the max degree + 1
    start = np.zeros(ndim, dtype=int)
    stop = np.full(ndim, max_degree+1, dtype=int)   # Add +1 so np.arange(0, stop) fills up to degree "d"
    bound = stop.max()

    # To control the size of the arrays:
    dtype = np.uint8 if bound < 256 else np.uint16
    range_ = np.arange(bound, dtype=dtype)           # vector with values of "d" to consider
    # Initialize the indices for the first parameter (row-wise), based on the order range
    indices = range_[:, np.newaxis]  # list of orders, in order

    # Fill the combinatorics array, one dimension at a time:
    for idx in range(ndim - 1):

        # Repeats the current set of indices ndim times
        # e.g. [0,1,2] -> [0,1,2,0,1,2,...,0,1,2]
        indices = np.tile(indices, (bound, 1))

        # Stretches ranges over the new dimension.
        # e.g. [0,1,2] -> [0,0,...,0,1,1,...,1,2,2,...,2]
        front = range_.repeat(len(indices) // bound)[:, np.newaxis]

        # Put the array "front" in front of the previous "indices" array, to do the combinations of dimensions <= idx
        indices = np.column_stack((front, indices))

        # Truncate at each step to keep memory usage low, dor idx > 0
        idx_to_keep = np.sum(indices, axis=-1) <= max_degree
        indices = indices[idx_to_keep]

    # Order in descending norm value (sum of all orders), giving priority to the first dimensions
    new_order = sort_basis_indices(keys=indices.T, reverse=False, graded=True)
    indices = indices[new_order]

    return indices


def get_pnorm_polynomial_basis(max_degree, ndim, p=0.85):
    """
    Get the order combinations for all dimensions, truncating the basis orders using the L_p norm.
    math:
        L_p(x) = \sum_i |x_i/b_i|^p ^{1/p} \leq 1
    where :math:`b_i` are bounds that each :math:`x_i` should follow
    :param max_degree: int
        Max polynomial degree
    :param ndim: int
        Number of parameters
    :param p: float
        The `p` in the `L_p`-norm.
    :return:
    """
    # Arrays with smallest order (0) to the max degree + 1
    start = np.zeros(ndim, dtype=int)
    stop = np.full(ndim, max_degree+1, dtype=int)   # Add +1 so np.arange(0, stop) fills up to degree "d"
    bound = stop.max()

    # To control the size of the arrays:
    dtype = np.uint8 if bound < 256 else np.uint16
    range_ = np.arange(bound, dtype=dtype)           # vector with values of "d" to consider
    # Initialize the indices for the first parameter (row-wise), based on the order range
    indices = range_[:, np.newaxis]  # list of orders, in order

    # Fill the combinatorics array, one dimension at a time:
    for idx in range(ndim - 1):

        # Repeats the current set of indices ndim times
        # e.g. [0,1,2] -> [0,1,2,0,1,2,...,0,1,2]
        indices = np.tile(indices, (bound, 1))

        # Stretches ranges over the new dimension.
        # e.g. [0,1,2] -> [0,0,...,0,1,1,...,1,2,2,...,2]
        front = range_.repeat(len(indices) // bound)[:, np.newaxis]

        # Put the array "front" in front of the previous "indices" array, to do the combinations of dimensions <= idx
        indices = np.column_stack((front, indices))

        # Truncate at each step to keep memory usage low, dor idx > 0,using p-norm sparsity
        idx_to_keep = np.sum((indices/bound)**p, axis=-1)**(1./p) <= 1
        indices = indices[idx_to_keep]

    # Order in descending norm value (sum of all orders), giving priority to the first dimensions
    new_order = sort_basis_indices(keys=indices.T, reverse=False, graded=True)
    indices = indices[new_order]

    return indices


def gen_multi_idx(n_o, dim):
    """
    Ilja's code: generates mulit-indices of multi-variate polynomial base
    uses graded lexicographic ordering (p. 156, Sullivan)
    """
    a = np.arange(n_o+1, dtype=np.int32)
    alphas = a
    for d in range(1, dim):
        a_tmp = np.vstack((np.repeat(alphas, n_o +1, axis=0).T, np.tile(a, alphas.shape[0]))).T
        alphas = a_tmp[a_tmp.sum(axis=1) < n_o+1]
    return alphas.reshape((-1, dim))


# 3. Training steps:
# 3.1 Evaluate univariate orthonormal basis
def evaluate_univariate_basis(samples, max_degree, poly_coeff):
    """
    Evaluate the aPCE univariate basis functions in each "samples" sample.
    :param samples: np.array, [n_samples, n_dim]
        With parameter sets to evaluate in the univariate basis functions, to build a PSI
    :param max_degree: int
        maximum polynomial degree being considered
    :param poly_coeff: np.array, [n_dim, max_degree+2, max_degree+2]
        With the coefficients for each univariate basis function. One array for each dimension
    :return: np.array [n_params, n_samples, max_degree + 1]
        array with the evaluated univariate, orthonormal basis functions, evaluated in "samples"
    """
    n_samples, n_params = samples.shape

    univ_basis = np.zeros((n_params, n_samples, max_degree + 1))

    for i in range(n_params):  # Evaluate one parameter (dimension) at a time.
        values = np.zeros((n_samples, max_degree + 1))
        for deg in range(max_degree + 1):
            step = poly_coeff[i, deg]
            step2 = polyval(samples[:, i], poly_coeff[i, deg])
            values[:, deg] = polyval(samples[:, i], poly_coeff[i, deg]).T
        univ_basis[i, :, :] = values

    return univ_basis


# 3.2 Generate PSI matrix
def create_psi(samples, basis_indices, poly_coeff, max_degree):
    """
    Author: Farid Mohammadi (BayesValidRox)
    This function assemble the design matrix Psi from the given basis index set INDICES and the univariate
    polynomial evaluations univ_p_val
    :param samples: np.array, [n_samples, n_dim]
        With parameter sets to evaluate in the univariate basis functions, to build a PSI
    :param basis_indices: np.array [n_terms, n_params]
        with combination of univariate polynomials to use
    :param poly_coeff: np.array, [n_dim, max_degree+2, max_degree+2]
        With the coefficients for each univariate basis function. One array for each dimension
    :param max_degree: int
        max degree of polynomial to consider
    :return:
    """
    # Evaluate univariate basis functions:
    univ_basis_func = evaluate_univariate_basis(samples=samples, max_degree=max_degree, poly_coeff=poly_coeff)
    # Initialization and consistency checks
    # number of input variables
    n_params = univ_basis_func.shape[0]
    n_samples = univ_basis_func.shape[1]
    n_terms = basis_indices.shape[0]

    # Preallocate the Psi matrix for performance: all ones, since P^0 = 1 (already accounted for)
    psi = np.ones((n_samples, n_terms))

    # Assemble the Psi matrix
    for m in range(n_params):   # one parameter at a time - each array in 3D array in univ_basis_func
        aa = np.where(basis_indices[:, m] > 0)[0]   # get idx of rows where parameter "m" has order > 0
        try:
            basisIdx = basis_indices[aa, m]   # get order for parameter "m" for each aa index
            # Copy the evaluated basis functions of orders 'basisIdx' for parameter m
            bb = univ_basis_func[m, :, basisIdx].T.reshape(psi[:, aa].shape)
            # Multiply the values to get the multivariate orthonormal basis functions, add to PSI matrix
            psi[:, aa] = np.multiply(psi[:, aa], bb)
        except ValueError as err:
            raise err
    return psi


# 3.3
def train(psi, Y):
    n_obs_ = Y.shape[1]
    pce_list = []

    for i in range(n_obs_):
        # method 1: Direct estimation
        y = Y[:, i]
        coeff_1 = (scipy.linalg.pinv(np.dot(psi.T, psi)).dot(psi.T)).dot(y)

        # Method 2: Scikit-Learn's linear model regression
        clf_poly = lm.LinearRegression(fit_intercept=False)
        clf_poly.fit(psi, y)

        # method 3: Analogous to pinv() in Matlab
        coeff_3 = scipy.linalg.pinv(psi).dot(y.reshape(-1, 1))

        # method 4: Scikit-Learn's Bayesian Ridge regression
        clf_poly_4 = lm.BayesianRidge(n_iter=1000, tol=1e-7,
                                      fit_intercept=False,
                                      compute_score=True,
                                      alpha_1=1e-04, alpha_2=1e-04,
                                      lambda_1=1 / np.var(y), lambda_2=1 / np.var(y))
        clf_poly_4.fit(psi, y)

        stop = 1


# ...........................................................................................................
# Input ....................................................................................................
N = 3   # 2, 10 --> number of uncertain parameters
d = 3   # maximum polynomial degree
n_obs = 1
mc_size = 100_000
n_init = 6

if n_obs > 1:
    loc = np.arange(0, n_obs, 1.) / 9          # loc observations
else:
    loc = np.array([0.1111])
observations = np.repeat([2.], n_obs)      # observation values, for each loc
error = np.repeat([2.], n_obs)


# 1. We need to othonormal basis functions for each input paramter (distribution) ----------------------------------
# Build priors:
# prior_distribution = stats.uniform.rvs(loc=-3, scale=6, size=(mc_size, N))
# pickle.dump(prior_distribution, open('Z://users//ac136123//Codes//prior.pickle', "wb"))

prior_distribution = pickle.load(open('Z://users//ac136123//Codes//prior.pickle', 'rb'))

# Max number of terms
P = int(math.factorial(N+d)/(math.factorial(N)*math.factorial(d)))

orthonormal_basis = np.zeros((N, d+2, d+2))  # d+2 because it goes from 0 to d+1
dict_polycoeff = dict()
for i in range(N):
    orthonormal_basis[i, :, :] = compute_orthonomal_basis(prior_distribution[:, i], d)
    dict_polycoeff[f'{i+1}'] = orthonormal_basis[i, :, :]

# 2. Computation of the polynomial degrees
basis_indices = get_polynomial_basis(max_degree=d, ndim=N)


# t2 = time.time()
# idx2 = gen_multi_idx(d, N)
# print(f'Ilja: {time.time()-t2}')


# 3. Collocation points:
collocation_points = np.array([
    [0, 0, 0],
    [2.5, -2.5, 2.5],
    [-2.5, 2.5, -2.5],
    [-1.25, -1.25, 1.25],
    [3.75, 3.75, -3.75],
    [1.25, -3.75, -1.25]
])

# collocation_points = stats.uniform.rvs(loc=-3, scale=6, size=(n_init, N), random_state=42)
model_evaluations = analytical_function(collocation_points, t=loc)

# 4. Train
# 4.1
psi = create_psi(samples=collocation_points, poly_coeff=orthonormal_basis, basis_indices=basis_indices, max_degree=d)

# 4.2
train(Y=model_evaluations, psi=psi)


stp = 1