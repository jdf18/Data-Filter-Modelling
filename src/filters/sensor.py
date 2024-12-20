import numpy as np

class Sensor:
	def __init__(self,
				observation_matrix: np.array,
				measurement_noise_covariance_matrix: np.array,
				sample_function
				):
		self.observation_matrix = observation_matrix
		self.measurement_noise_covariance_matrix = measurement_noise_covariance_matrix
		self.sample_function = sample_function

	def sample(self, *args, **kwargs):
		return_value = self.sample_function(*args, **kwargs)
		return return_value, self.observation_matrix, self.measurement_noise_covariance_matrix