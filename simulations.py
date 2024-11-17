import numpy as np

class simulations:
    def __init__(self):
        """
        Initialize the Simulations class.
        """
        pass

    def generate_simplex_sample(self, dimensions):
        """
        Generate a sample from a simplex in the specified dimensions.

        Parameters:
        dimensions (int): The number of dimensions for the simplex sample.

        Returns:
        np.ndarray: A sample from the simplex.
        """
        # generating d samples from exponential distribution
        samples = -np.log(np.random.rand(dimensions))
        
        # normalizing the samples to make them sum up to 1
        simplex_sample = samples / np.sum(samples)
    
        return simplex_sample
    
    def generate_simplex_sample_gamma(self, dimensions, alpha=1):
        """
        Generate a sample from a simplex in the specified dimensions using gamma distribution.

        Parameters:
        dimensions (int): The number of dimensions for the simplex sample.
        alpha (float): The shape parameter of the Gamma distribution (default is 1 for uniform).

        Returns:
        np.ndarray: A sample from the simplex.
        """
        # Generate gamma-distributed random variables
        gamma_samples = np.random.gamma(alpha, 1, dimensions)
        
        # Normalize the gamma samples to make them sum up to 1
        simplex_sample = gamma_samples / np.sum(gamma_samples)
        
        return simplex_sample

    def sample_generator(self, sample_size, dimensions, draw_function):
        """
        Generate a set of samples using the specified draw function.

        Parameters:
        sample_size (int): The number of samples to generate.
        dimensions (int): The number of dimensions for each sample.
        draw_function (callable): A function that generates a single sample given the dimensions.

        Returns:
        np.ndarray: An array of generated samples.
        """
        # creating the sample vector
        sample = []

        # drawing sample_size samples
        for _ in range(sample_size):
            x = draw_function(dimensions)
            sample.append(x)

        # transforming sample from a list to a np.array
        sample = np.array(sample)
        return sample

    def mc_integration(self, sample_size, dimensions, objective_function, draw_function, space_volume):
        """
        Perform Monte Carlo integration using the specified parameters.

        Parameters:
        sample_size (int): The number of samples to use in the integration.
        dimensions (int): The number of dimensions for each sample.
        objective_function (callable): The function to integrate.
        draw_function (callable): The function to generate samples.
        space_volume (float): The volume of the integration space.

        Returns:
        float: The approximated integral value.
        """
        # generating the sample
        sample = self.sample_generator(sample_size=sample_size, dimensions=dimensions, draw_function=draw_function)

        # computing the sum of the drawn value evaluated on the objective function
        total_sum = np.sum(objective_function(sample))

        # computing the approximated integral
        approx_integral = (total_sum * space_volume) / sample_size

        return approx_integral