# packages
import numpy as np
from scipy.stats import dirichlet


# methods

def importance_sampling(proposal_dist, objective_func, alpha_proposal, eta, num_samples):
    """
    Perform Importance Sampling with a given proposal and objective function.

    Parameters:
    -----------
    proposal_dist : function
        Function that generates samples from the proposal distribution. 
        It takes the number of samples and the proposal alpha as inputs.

    objective_func : function
        Objective function to be evaluated for the target distribution. 
        It takes the sample points, target alpha, and sigma as inputs.

    alpha_proposal : ndarray
        Parameters of the Dirichlet proposal distribution.

    alpha_target : ndarray
        Parameters of the target Dirichlet distribution.

    sigma : float
        Scaling parameter for the objective function.

    num_samples : int
        Number of samples to draw.

    Returns:
    --------
    resampled_points : ndarray
        The points resampled based on importance weights.
        
    weights : ndarray
        The computed importance weights for each sample.

    Notes:
    ------
    - Samples from the proposal distribution and computes the objective function value for each.
    - Computes the importance weights and resamples the points based on those weights.
    """

    # Step 1: generating samples from the proposal distribution
    samples = proposal_dist(num_samples, alpha_proposal)

    # Step 2: computing the importance weights
    weights = objective_func(samples, eta) / np.array([dirichlet.pdf(i,alpha_proposal) for i in samples])

    # normalizing weights to sum to 1
    weights /= np.sum(weights)

    # Step 3: resampling based on the weights
    resampled_indices = np.random.choice(np.arange(num_samples), size=num_samples, p=weights)
    resampled_samples = samples[resampled_indices]

    return resampled_samples, weights

def log_importance_sampling(proposal_dist, objective_func, alpha_proposal, eta, num_samples):
    """
    Perform Importance Sampling with a given proposal and objective function.

    Parameters:
    -----------
    objective_func : function
        The objective function to evaluate the target distribution (log-probabilities).

    proposal_dist : function
        Function that generates samples from the proposal distribution.

    alpha_proposal : ndarray
        Parameters of the proposal distribution.

    alpha_target : ndarray
        Parameters of the target distribution.

    sigma : float
        Scaling parameter for the objective function.

    num_samples : int
        Number of samples to draw from the proposal distribution.

    Returns:
    --------
    resampled_points : ndarray
        Resampled points based on the importance weights.
    """
    # Step 1: generating samples from the proposal distribution
    samples = proposal_dist(num_samples, alpha_proposal)  # Ensure this returns an array of samples

    # Step 2: computing the log of the target distribution (log-probabilities)
    log_p_target = objective_func(samples, eta)

    # Step 3: computing the log of the proposal distribution (Dirichlet log-probabilities)
    log_p_proposal = np.array([dirichlet._logpdf(i.T,alpha_proposal) for i in samples])  # Note: dirichlet.logpdf expects transposed input

    # Step 4: computing log-weights
    log_weights = log_p_target - log_p_proposal

    # Step 5: normalzing the log-weights to avoid numerical issues
    max_log_weight = np.max(log_weights)
    weights = np.exp(log_weights - max_log_weight)  # Subtract max to prevent overflow
    weights /= np.sum(weights)  # Normalize to sum to 1

    # Step 6: resampling based on the normalized weights
    resampled_indices = np.random.choice(np.arange(num_samples), size=num_samples, p=weights)
    resampled_samples = samples[resampled_indices]

    return resampled_samples, weights



def met(x, lpr, eta, num_samples, alpha_proposal, burn_in=100, thinning=1):
    """
    Perform Metropolis updates with a Gaussian proposal distribution and projection to the simplex.

    This function implements a Random-Walk Metropolis algorithm where a new state is proposed
    by adding Gaussian noise to the current state, then projecting the proposed state back
    onto the simplex. The proposal distribution has a mean equal to the current state and a 
    standard deviation scaled by `s`. The function includes options for burn-in and thinning 
    to ensure effective sampling.

    Parameters:
    -----------
    x : ndarray
        Initial state of the Markov chain. This is the starting point for the Metropolis algorithm.
        It is assumed that `x` lies within the simplex.

    lpr : function
        Function that takes a state `x` and a parameter `eta`, and returns the log probability (or
        log density) of `x` under the distribution defined by `eta`.

    eta : float
        Parameter that influences the distribution from which the log probability is computed.

    s : float
        Standard deviation for the Gaussian noise used in the proposal distribution.

    num_samples : int
        Number of samples to collect after the burn-in phase.

    burn_in : int, optional
        Number of initial steps to discard (burn-in) to allow the Markov chain to reach a stationary
        distribution. Default is 0.

    thinning : int, optional
        Interval at which samples are collected. If thinning > 1, only every `thinning`-th sample is
        stored, reducing autocorrelation. Default is 1 (no thinning).

    Returns:
    --------
    samples : ndarray
        Array of sampled states from the Markov chain after the burn-in phase, with thinning applied
        if specified. Each row corresponds to a sample.

    Notes:
    ------

    Burn-in steps are executed first, followed by the sampling phase where thinning can be applied.
    The function returns an array of samples after the burn-in phase.
    """
    met_calls = 0
    met_rejections = 0
    samples = []

    # burn-in phase
    for _ in range(burn_in):
        met_calls += 1
        # proposing a new state with Gaussian noise scaled by s
        px = dirichlet.rvs(alpha_proposal, size=1)
        # computing the difference in log probability
        d = lpr(px, eta) - lpr(x, eta)
        # accepting or rejecting the new state
        if not np.isnan(d) and np.log(np.random.rand()) <d:
            x = px
        else:
            met_rejections += 1

    # sampling phase with thinning
    for i in range(num_samples * thinning):
        met_calls += 1
        # proposing a new state with Gaussian noise scaled by s
        px = dirichlet.rvs(alpha_proposal, size=1)
        # computing the difference in log probability
        d = lpr(px, eta) - lpr(x, eta)
        # accepting or rejecting the new state
        if not np.isnan(d) and np.log(np.random.rand()) < d:
            x = px
        else:
            met_rejections += 1
        # collecting the sample only if it's not a thinning step
        if (i + 1) % thinning == 0:
            samples.append(x.copy())

    return np.array(samples)

def log_sum_exp(x):
    """
    Compute the logarithm of the sum of exponentials of input elements in a numerically stable way.

    This function is useful when adding probabilities in log-space to prevent underflow or overflow 
    issues that can arise with direct exponentiation, especially when the input values are large 
    or very different in magnitude.

    Parameters:
    -----------
    x : array-like
        An array or list of values for which the log of the sum of exponentials is to be computed.

    Returns:
    --------
    float
        The logarithm of the sum of exponentials of the input elements.

    Notes:
    ------
    The function uses the identity:
    
    log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
    
    This method is numerically stable because it subtracts the maximum value before exponentiation, 
    preventing very large values from causing overflow and very small values from underflowing 
    to zero.
    """
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))

def log_add_exp(a, b):
    """
    Compute the logarithm of the sum of exponentials of two numbers in a numerically stable way.

    This function is used to add two probabilities in log-space without directly exponentiating 
    them, which can lead to numerical instability when dealing with very large or very small 
    numbers.

    Parameters:
    -----------
    a : float
        The logarithm of the first value.
        
    b : float
        The logarithm of the second value.

    Returns:
    --------
    float
        The logarithm of the sum of the exponentials of the input values.

    Notes:
    ------
    The function uses the identity:
    
    log(exp(a) + exp(b)) = max(a, b) + log(1 + exp(-|a - b|))
    
    This method is numerically stable because it factors out the larger of the two exponentials, 
    preventing overflow and ensuring that the computation remains within the range of representable 
    numbers.
    """
    m = np.maximum(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))  

def ais(lpr, alpha_proposal, eta, trans, initial=None, final=None, num_samples=1, burn_in=100, thinning=10):
    """
    Perform Annealed Importance Sampling (AIS) to estimate the ratio of normalizing constants 
    between a sequence of probability distributions defined by different parameter values.
    
    """
    

    n = len(eta)
    ratios = np.empty(n)

    # checking if performing a forward or reverse run
    if initial is None and final is None:
        raise ValueError("Must specify a state from either the initial or final distribution")
    
    if final is None:
        d = len(initial)
        states = np.empty((n, d))
        states[0, :] = initial
        prev_eta = 0
        order = np.arange(n)
    elif initial is None:
        d = len(final)
        states = np.empty((n, d))
        states[0, :] = final
        prev_eta = eta[-1]
        order = np.arange(n - 1, -1, -1)
    else:
        raise ValueError("Must not specify states from both initial and final distributions")

    # looping through the sequence of distributions
    for i in range(0, n):
        if order[i] == 0:
            this_eta = np.zeros(len(alpha_proposal))
        else:
            this_eta = eta[order[i]]

        # computing the log of the next ratio
        ratios[i] = lpr(states[i, :], this_eta).item() - lpr(states[i, :], prev_eta).item()
       
        if i < (n - 1):
            # simulating the next transition
            next_state = trans(states[i, :], lpr, this_eta, num_samples, alpha_proposal, burn_in, thinning)
            next_state.flatten()
            states[i + 1, :] = next_state  
            
            # updating the previous eta
            prev_eta = this_eta

    return {'ratios': ratios, 'states': states}

def lis(lpr, eta, trans, stepsize, alpha_proposal, n_trans, initial=None, final=None, rho0=None, rho1=None, iratios=None, burn_in=1, num_samples=1, thinning =10 ):
    """
    Perform a single run of Linked Importance Sampling (LIS).

    This function simulates a sequence of Markov chains to estimate the ratios
    of normalizing constants between distributions defined by different values 
    of the eta parameter. The sequence can be run either in the `forward` 
    direction (from initial to final distribution) or in the `reverse` 
    direction (from final to initial distribution). Markov chain updates for 
    the intermediate distributions are performed by the specified transition 
    function.

    Parameters:
    -----------
    lpr : function
        Function with arguments (x, eta) that returns the log probability (or density) 
        of `x` under the distribution defined by `eta`, with the normalizing constant omitted.

    eta : array-like
        Vector of parameter values for the distributions in the sequence, except for the 
        initial distribution, which is assumed to have eta=0.

    trans : function
        Function with arguments (x, lpr, eta, stepsize) that returns a new state produced 
        by a Markov chain transition from `x`. This function should satisfy detailed balance 
        with respect to the distribution defined by `lpr` and `eta`. The `stepsize` argument 
        may modify the transition.

    stepsize : array-like
        Vector of stepsize arguments for `trans`, to be used for each distribution in the sequence.
        This vector will be extended or truncated to match the length of `eta` plus one by 
        repetition if necessary. The first element is not used for forward runs, and the last 
        is not used for reverse runs.

    n_trans : array-like
        Vector containing the number of transitions in the chains for each distribution. 
        This vector will be extended or truncated to the length of `eta` plus one by repetition 
        if necessary. The first element is not used for forward runs, and the last is not used 
        for reverse runs.

    initial : ndarray, optional
        A matrix where rows contain states sampled from the initial distribution (eta=0), either 
        independently or via a Markov chain that leaves the initial distribution invariant. 
        This argument is required for a forward run.

    final : ndarray, optional
        A matrix where rows contain states sampled from the final distribution, either independently 
        or via a Markov chain that leaves the final distribution invariant. This argument is required 
        for a reverse run.

    rho0 : float, optional
        Power to be used for the power form of the bridge distribution between p0 and p1. Default is 0.5.

    rho1 : float, optional
        Power to be used for the power form of the bridge distribution between p0 and p1. Default is 0.5.

    iratios : array-like, optional
        Vector of guesses at log ratios to be used for the optimal form of the bridge. This vector 
        will be extended or truncated to the length of `eta` by repetition if necessary. This 
        argument cannot be specified if `rho0` or `rho1` is specified.

    Returns:
    --------
    result : dict
        A dictionary with the following keys:
        
        - `ratios`: A vector of the logs of the ratios whose product is an unbiased estimate of the 
                    ratio of the normalizing constant for the last distribution to the normalizing 
                    constant for the first. For a forward run, `first=initial (eta=0)` and `last=final`, 
                    whereas for a reverse run, this (and the order in `ratios`) is reversed.
        
        - `chains`: A list where the i-th element is a matrix whose rows are the states of the i-th 
                    chain, satisfying detailed balance with respect to the i-th distribution, 
                    not counting the first chain (whose states were passed as the `initial` or `final` argument).
        
        - `links`: A vector where the i-th element is the index (starting from 1) of the link state in the 
                   i-th chain stored in `chains`.

    Notes:
    ------
    To perform both a forward and a reverse run, with reversal being the only difference, 
    the `lis` function should be called twice with the same values for `lpr`, `eta`, `trans`, 
    `stepsize`, `n_trans`, `rho0`, `rho1`, and `iratios`, but with one call specifying `initial` 
    and the other specifying `final`.
    """
      
    # checking arguments for validity
    if np.any(n_trans < 0):
        raise ValueError("An element of n_trans is less than zero")

    if (rho0 is not None or rho1 is not None) and iratios is not None:
        raise ValueError("Can't specify rho0 or rho1 and also iratios")

    # finding number of distributions
    n = len(eta)

    # determining whether we are doing a forward or reverse run
    if initial is None and final is None:
        raise ValueError("Must specify a state from either the initial or final distribution")
    
    if final is None:
        d = initial.shape[1]
        prev_n_trans = initial.shape[0] - 1
        prev_chain = initial
        prev_eta = 0
        order = np.arange(n)
    elif initial is None:
        d = final.shape[1]
        prev_n_trans = final.shape[0] - 1
        prev_chain = final
        prev_eta = eta[-1]
        order = np.arange(n-1, -1, -1)
    else:
        raise ValueError("Must not specify states from both initial and final distributions")

    # setting up vectors of stepsizes and numbers of transitions
    stepsize = np.resize(stepsize, n + 1)
    n_trans = np.resize(n_trans, n + 1)

    # setting default values for rho0 and rho1 if needed
    if iratios is None:
        if rho0 is None:
            rho0 = 0.5
        if rho1 is None:
            rho1 = 0.5
    else:
        iratios = np.resize(iratios, n)
        if initial is None:
            iratios = -iratios

    # creating arrays and lists to store the results
    ratios = np.full(n, np.nan)
    links = np.full(n, np.nan)
    chains = []

    # looping through the sequence of distributions
    for i in range(n):
        if order[i] == 0:
            this_eta = 0  
        else: 
            this_eta = eta[order[i]]  
        this_n_trans = n_trans[1 + order[i]]

        # allocating space for new chain
        chain = np.full((this_n_trans + 1, d), np.nan)

        # randomly deciding how many transitions come before the link state
        n_before = np.random.randint(0, this_n_trans + 1)

        # selecting the link state from the previous chain, and compute the numerator of the ratio
        lr = np.full(prev_chain.shape[0], np.nan)
        for j in range(prev_chain.shape[0]):
            lpr0 = lpr(prev_chain[j], prev_eta)
            lpr1 = lpr(prev_chain[j], this_eta)
            if iratios is None:
                lr[j] = rho1 * lpr1.item() + (rho0 - 1) * lpr0.item()
            else:
                lr[j] = lpr1.item() - log_add_exp(iratios[i] + np.log(prev_n_trans / this_n_trans) + lpr0, lpr1)
        lr_sum = log_sum_exp(lr)
        k = np.random.choice(prev_chain.shape[0], p=np.exp(lr - lr_sum))
        lnk = prev_chain[k]

        # computing the numerator of the ratio
        chain[n_before] = lnk
        links[i] = n_before
        log_numerator = lr_sum - np.log(prev_chain.shape[0])

        # simulating the states before the link state
        if n_before > 0:
            for j in range(n_before - 1, -1, -1):
                chain[j] = np.squeeze(trans(chain[j + 1], lpr, this_eta, alpha_proposal = alpha_proposal, num_samples=1, burn_in=burn_in, thinning = thinning))

        # simulating the states after the link state
        if n_before < this_n_trans:
            for j in range(n_before + 1, this_n_trans + 1):
                chain[j] = np.squeeze(trans(chain[j - 1], lpr, this_eta, alpha_proposal = alpha_proposal, num_samples=1, burn_in= burn_in, thinning = thinning))

        # computing the denominator of the ratio
        lr = np.full(chain.shape[0], np.nan)
        for j in range(chain.shape[0]):
            lpr0 = lpr(chain[j], prev_eta)
            lpr1 = lpr(chain[j], this_eta)
            if iratios is None:
                lr[j] = rho0 * lpr0.item() + (rho1 - 1) * lpr1.item()
            else:
                lr[j] = lpr0.item() - log_add_exp(iratios[i] + np.log(prev_n_trans / this_n_trans) + lpr0, lpr1)
        lr_sum = log_sum_exp(lr)
        log_denominator = lr_sum - np.log(chain.shape[0])

        ratios[i] = log_numerator - log_denominator
        # moving onto the next distribution
        chains.append(chain)
        prev_chain = chain
        prev_eta = this_eta
        
    return {'ratios': ratios}

import numpy as np
import time

def run_lis_experiment(
    eta_ranges, 
    alpha, 
    proposal_distribution, 
    wrapped_log_objective_function, 
    alpha_proposal, 
    met_obj, 
    stepsize, 
    bigm_checkpoints, 
    eta_part
):
    results_dict_lis_for_geom_t = {}

    for i, eta_range in enumerate(eta_ranges):
        # Creating eta array for the current range
        eta = np.array([np.full(len(alpha), eta_val) for eta_val in eta_range])

        # Initializing result and time storage
        results = []
        times = []
        
        # Fix BigM 
        BigM = 10

        # Run LIS
        for m in range(1, BigM + 1):
            # Starting the timer
            start_time = time.time()

            # Updating initial state using met_dir
            initial, _ = log_importance_sampling(proposal_distribution, wrapped_log_objective_function, alpha_proposal, eta[0], 1)

            # Running LIS with the wrapped log objective function
            result = lis_obj(
                wrapped_log_objective_function, 
                eta, 
                met_obj, 
                stepsize, 
                alpha_proposal, 
                10, 
                initial=initial, 
                rho1=0.5, 
                rho0=0.5, 
                burn_in=10, 
                num_samples=10, 
                thinning=1
            )

            # Computing and storing the time for this iteration
            elapsed_time = time.time() - start_time
            times.append(elapsed_time)

            # Storing the result
            results.append(np.prod(np.exp(result['ratios'])))

            # Checking if we've reached a checkpoint
            if m in bigm_checkpoints:
                # Computing and storing the mean ratio and times for the current checkpoint
                results_dict_lis_for_geom_t[(m, eta_part[i])] = {
                    'mean_ratio': np.mean(results),
                    'times': times[:m]
                }

                # Printing output for the current combination of eta and BigM checkpoint
                print(f"Checkpoint reached: eta = {eta_part[i]}, BigM = {m}, mean_ratio = {results_dict_lis_for_geom_t[(m, eta_part[i])]['mean_ratio']}")

    return results_dict_lis_for_geom_t

# estimation 

# function to find final estimates of ratios of normalizing constants.
def loge(results):
    """
    Calculate the log estimates of the ratio of normalizing constants
    from a list of LIS or AIS results.
    
    Parameters:
    -----------
    results : list of dict
        A list of dictionaries, where each dictionary contains the results
        from an LIS or AIS run, including a 'ratios' key with log ratio values.

    Returns:
    --------
    s : ndarray
        A vector containing the logs of the estimates for the ratio of normalizing 
        constants found by each of these LIS or AIS runs.
    """
    n_runs = len(results)
    s = np.full(n_runs, np.nan)

    for i in range(n_runs):
        s[i] = np.sum(results[i]['ratios'])
        
    return s


# function to find simple LIS/AIS estimates for the ratio of normalizing constants.
def simple_est(e):
    """
    Calculate simple LIS/AIS estimates for the ratio of normalizing constants.

    Parameters:
    -----------
    e : ndarray
        A vector of log estimated ratios from LIS or AIS.

    Returns:
    --------
    result : ndarray
        A 2x2 matrix containing two estimates (first column) and their standard
        errors (second column) in its two rows. The estimates are for the ratio 
        of normalizing constants and for the log of this ratio.
    """
    n = len(e)
    m = np.max(e)
    e = np.exp(e - m)

    ratio_estimate = np.mean(e) * np.exp(m)
    log_ratio_estimate = np.log(np.mean(e)) + m
    ratio_std_err = np.std(e) * np.exp(m) / np.sqrt(n)
    log_ratio_std_err = np.std(e) / np.mean(e) / np.sqrt(n)

    return np.array([[ratio_estimate, ratio_std_err],
                     [log_ratio_estimate, log_ratio_std_err]])


# function to find adjusted LIS/AIS estimates for the ratio of normalizing constants.
def adjusted_est(e):
    """
    Calculate adjusted LIS/AIS estimates for the ratio of normalizing constants.

    Parameters:
    -----------
    e : ndarray
        A vector of log estimated ratios from LIS or AIS.

    Returns:
    --------
    result : ndarray
        A 2x2 matrix containing two estimates (first column) and their standard
        errors (second column) in its two rows. The estimates are for the ratio 
        of normalizing constants and for the log of this ratio.
    """
    n = len(e)
    mu = np.mean(e)
    ss = np.var(e)

    m = np.max(e)
    e = np.exp(e - m)

    ratio_estimate = np.exp(mu + ss / 2)
    log_ratio_estimate = mu + ss / 2
    ratio_std_err = np.exp(mu + ss / 2) * np.sqrt(ss / n + 2 * ss**2 / (n - 1))
    log_ratio_std_err = np.sqrt(ss / n + 2 * ss**2 / (n - 1))

    return np.array([[ratio_estimate, ratio_std_err],
                     [log_ratio_estimate, log_ratio_std_err]])


# function to find bridged LIS/AIS estimates for the ratio of normalizing constants.
def bridged_est(fe, re, use=None, log_iest=None, n_iter=0, est_func=simple_est):
    """
    Calculate bridged LIS/AIS estimates for the ratio of normalizing constants.

    Parameters:
    -----------
    fe : ndarray
        A vector of log estimated ratios from a forward call of LIS or AIS.
        
    re : ndarray
        A vector of log estimated ratios from a reverse call of LIS or AIS.
        
    use : int, optional
        If provided, only the first 'use' elements of fe and re are used. Default is None.
        
    log_iest : float, optional
        Initial estimate for the log of the ratio of normalizing constants. If None, geometric bridge is used. Default is None.
        
    n_iter : int, optional
        Number of iterations to improve the estimate when using the optimal form of bridge. Default is 0.
        
    est_func : function, optional
        Function used to find lower-level estimates (either simple_est or adjusted_est). Default is simple_est.

    Returns:
    --------
    br : ndarray
        A 2x2 matrix containing two estimates (first column) and their standard
        errors (second column) in its two rows. The estimates are for the ratio 
        of normalizing constants and for the log of this ratio.
    """
    if use is not None:
        use = int(np.ceil(use))
        fe = fe[:use]
        re = re[:use]

    N_0 = len(fe)
    N_1 = len(re)

    if log_iest is None:    # geometric bridge
        ne = fe / 2
        de = re / 2
        nr = est_func(ne)
        dr = est_func(de)
        br = nr.copy()
        br[1, 0] = nr[1, 0] - dr[1, 0]
        br[1, 1] = np.sqrt(nr[1, 1]**2 + dr[1, 1]**2)

    else:                   # optimal form of bridge
        logr = log_iest
        for _ in range(n_iter + 1):
            ne = -log_add_exp(logr + np.log(N_0 / N_1) - fe, 0)
            de = -log_add_exp(logr + np.log(N_0 / N_1), -re)
            nr = est_func(ne)
            dr = est_func(de)
            br = nr.copy()
            br[1, 0] = nr[1, 0] - dr[1, 0]
            br[1, 1] = np.sqrt(nr[1, 1]**2 + dr[1, 1]**2)
            logr = br[1, 0]

    br[0, 0] = np.exp(br[1, 0])
    br[0, 1] = br[1, 1] * br[0, 0]

    return br


# function to reverse estimates.
def rev_est(e):
    """
    Reverse estimates for 1/r and convert them to estimates for r.

    Parameters:
    -----------
    e : ndarray
        A 2x2 matrix containing estimates and their standard errors.

    Returns:
    --------
    e : ndarray
        The reversed estimates.
    """
    e[1, 0] = -e[1, 0]
    e[0, 0] = np.exp(e[1, 0])
    e[0, 1] = e[0, 0] * e[1, 1]

    return e
