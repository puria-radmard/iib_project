import torch
import numpy as np
from torch import nn
# from config.losses import *
from config.ensemble import *
from config.audio_lists import *
from config.architectures import *
from config.metric_functions import *

device = (
    torch.device('cuda') if torch.cuda.is_available() 
    else torch.device('cpu')
)

DEFAULT_DTYPE = torch.float32


# DEFAULT_MEANS = np.array([
#     6.5701337 , 6.9947671 , 7.43054389, 7.71279133, 7.77880355,
#     7.71001804, 7.84715879, 7.96562144, 7.97178071, 7.90799967,
#     7.8597201 , 7.84901087, 7.85534653, 7.86995385, 7.90230486,
#     7.9589462 , 8.03425086, 8.15287966, 8.27845208, 8.3697235 ,
#     8.43723236, 8.48500926, 8.55167921, 8.66847626, 8.75753459,
#     8.80738884, 8.8440763 , 8.89638177, 8.96199342, 9.0078516 ,
#     9.006915  , 8.97789231, 8.95423136, 8.88066078, 8.83438973,
#     8.7114148 , 8.55230933, 8.21362381, 7.7309175 , 7.052912  
# ])
# 
# 
# DEFAULT_STDS = np.array([
#     1.97248723, 2.05983263, 2.13193066, 2.17351782, 2.16397947,
#     2.12706346, 2.18262594, 2.20196923, 2.16154033, 2.12601442,
#     2.07945462, 2.0353137 , 1.98933221, 1.96260985, 1.9444271 ,
#     1.93609969, 1.92942288, 1.92087713, 1.91456481, 1.91322875,
#     1.90662833, 1.89707972, 1.89627434, 1.90301773, 1.91477061,
#     1.89912617, 1.89208545, 1.89692786, 1.91262111, 1.90406365,
#     1.90758007, 1.90371552, 1.90550009, 1.8815407 , 1.86886717,
#     1.84525824, 1.82552013, 1.78753352, 1.71215755, 1.54987047
# ])