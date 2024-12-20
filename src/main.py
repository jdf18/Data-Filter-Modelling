import filters.model
import filters.dataset

data = filters.Dataset('filepath', 'format')

model = filters.model.KalmanFilter(6)
print(model)
evalutation = filters.ModelEvaluator()

for sample in data:
	model.update(sample)
	evaluation.eval(sample, model)

evaluation.plot()
