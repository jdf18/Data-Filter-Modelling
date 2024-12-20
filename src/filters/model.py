import numpy as np
import matplotlib.pyplot as plt

from filters.sensor import Sensor

class Logger:
	def __init__(self):
		self.next_step = 0
		self.store = []
		self.state_label_translations = {}
		self.labels = {}

		self.total_plots = 0

	def add_state_translation(self, label, translation_matrix):
		self.state_label_translations.update({
			label: translation_matrix
		})
		
	def step(self, time_val: float=None):
		if time_val is None:
			time_val = self.next_step
			self.next_step += 1
		self.store.append({
			'time': time_val,
			'data':[]
		})

	def log_value(self, label: str, data: str, value: float):
		self.store[-1]['data'].append({
			'label': label,
			'data': data,
			'value': value
		})

	def log_state(self, state):
		for l, m in self.state_label_translations.items():
			self.log_value(l, 'estimates', (m @ state)[0][0])

	def log_covariance(self, covariance):
		for l, m in self.state_label_translations.items():
			self.log_value(l, 'variances', m @ covariance @ m.T)
			

	def plot(self):
		plt.figure(figsize=(12,24))

		for e in self.store:
			for x, l, d, v in map(lambda x: (x, x['label'], x['data'], x['value']), e['data']):
				if not l in self.labels.keys():
					self.labels.update({
						l: {
							'datasets': []
						}
					})
	
				if not l: continue
	
				if not d in self.labels[l]['datasets']:
					self.labels[l]['datasets'].append(d)

		self.total_plots = len(self.labels.keys())

		def get_label(store, label, data_l):
			out = {}
			for state in store:
				for data in state['data']:
					if data['label'] != label: continue
					if data['data'] != data_l: continue
					out.update({state['time']:float(data['value'])})
					break
			return out

		plt.figure(figsize=(12,24))
		for i, (label, plot) in enumerate(self.labels.items()):
			references, measurements, estimates, variances = None, None, None, None
			if 'reference' in plot['datasets']:
				references = get_label(self.store, label, 'reference')
			if 'measurements' in plot['datasets']:
				measurements = get_label(self.store, label, 'measurements')
			if 'estimates' in plot['datasets']:
				estimates = get_label(self.store, label, 'estimates')
			if 'variances' in plot['datasets']:
				variances = get_label(self.store, label, 'variances')

			self.plot_1D(i+1, 'Test Title', label, references, estimates, variances, measurements)
				

	def plot_1D(self, plot_no, title, label, reference, estimates, variances, measurements):
		plt.subplot(self.total_plots, 1, plot_no)
		if not reference is None:
			plt.plot(reference.keys(), reference.values(), label=f"Reference {label}", linestyle="--")
		if not measurements is None:
			plt.scatter(measurements.keys(), measurements.values(), label="Measurements", color="red", s=10)
		if not estimates is None:
			plt.plot(list(estimates.keys()), list(estimates.values()), label="Kalman estimates", color="green")
			if not variances is None:
				plt.fill_between(
				    list(map(lambda x:x[0], zip(estimates.keys(), variances.keys()))),
				    [x - 2 * np.sqrt(y) for x, y in zip(estimates.values(), variances.values())],
				    [x + 2 * np.sqrt(y) for x, y in zip(estimates.values(), variances.values())],
				    color="orange",
				    alpha=0.3,
				    label="95%"
				)
		plt.title(f"{title}: {label}")
		plt.xlabel("Time")
		plt.ylabel(f"{label}")
		plt.legend()

	def plot_2D(self):
		pass

	def plot_3D(self):
		pass

class Filter:
	def __init__(self):
		self.state: tuple = None
	def __repr__(self):
		return f"<Unknown Filter Object {type(self)}>"
	def get_state(self):
		return self.state

class KalmanFilter(Filter):
	def __init__(self, 
                 dimensions: int, 
                 state_transition_matrix: np.array,
                 process_noise_covariance_matrix: np.array,
                 control_input_matrix: np.array = None
				):
		self.logger = Logger()
		
		self.dimensions = dimensions
		super().__init__()
        
		self.state = np.array([[0] for _ in range(self.dimensions)])
		self.state_covariance = np.eye(self.dimensions)
		
		self.state_transition_matrix = state_transition_matrix
		self.process_noise_covariance_matrix = process_noise_covariance_matrix
		
		if control_input_matrix == None:
			self.control_input_matrix = np.array([[0] for _ in range(self.dimensions)])
		else:
			self.control_input_matrix = control_input_matrix

	def __repr__(self):
		return f"<class filters.model.KalmanFilter dimensions:{self.dimensions}>"

	def prediction_step(self, control_input: np.array = np.array([[0]])):
        # u: control_input
        # x: self.state
		# P: self.state_covariance
        # F: self.state_transition_matrix
        # B: self.control_input_matrix
        # Q: self.state_covariance
	
		self.logger.step()
		self.state = self.state_transition_matrix @ self.state + self.control_input_matrix @ control_input
		self.state_covariance = self.state_transition_matrix @ self.state_covariance @ self.state_transition_matrix.T + self.process_noise_covariance_matrix

		return self.state

	def update_step(self, sensor: Sensor, *args, **kwargs):
		# z: measurement
        # H: observation matrix
		# R: measurement covariance
		# y: innovation
		# S: innovation covariance
		# K: kalman gain
        # P: self.state_covariance
		# F: self.state_transition_matrix

		z, H, R = sensor.sample(*args, **kwargs)
		self.logger.log_value('position', 'measurements', z)
		
		y = z - H @ self.state
		S = H @ self.state_covariance @ H.T + R
		K = self.state_covariance @ H.T @ np.linalg.inv(S)
		self.state = self.state + K @ y

		self.state_covariance = (np.eye(self.dimensions) - K @ H) @ self.state_covariance

		return self.state

	def log(self):
		self.logger.log_state(self.state)
		self.logger.log_covariance(self.state_covariance)
        
