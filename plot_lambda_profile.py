from matplotlib import pyplot as plt
import numpy as np
from classes_losses.multitask_distillation import *
print('done imports')
fig, axes = plt.subplots(1)
x = np.linspace(0, 49, 50)
y_no_lambda_scheduling_075 = [no_lambda_scheduling(0.75, e) for e in x]
y_no_lambda_scheduling_0 = [no_lambda_scheduling(0., e) for e in x]
y_no_lambda_scheduling_1 = [no_lambda_scheduling(1., e) for e in x]
y_step_pretrainer_lambda_scheduling = [step_pretrainer_lambda_scheduling(15, e) for e in x]
y_asymptotic_pretrainer_lambda_scheduling = [asymptotic_pretrainer_lambda_scheduling(5, e) for e in x]
axes.plot(x+1, y_no_lambda_scheduling_0, label = "Distillation only")
axes.plot(x+1, y_no_lambda_scheduling_1, label = "Acquisition prediction only")
axes.plot(x+1, y_no_lambda_scheduling_075, label = "Previous multitask set up")
axes.plot(x+1, y_step_pretrainer_lambda_scheduling, label = "Acquisition prediction with pretraining")
axes.plot(x+1, y_asymptotic_pretrainer_lambda_scheduling, label = "Gradual task shift")
axes.set_xlabel('Epoch number')
axes.set_ylabel('$\lambda$')
axes.legend()
fig.savefig('test_lambda_profiles.png')
