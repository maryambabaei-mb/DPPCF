#from utils.distance import*
from utils.data.distance import *
#from nice.utils.data import data_NICE
#from utils.data.data import data_NICE
#from utils.optimization.heuristic import best_first
#from nice.utils.optimization.heuristic import best_first
from dpnice.utils.optimization.reward import SparsityReward, ProximityReward, PlausibilityReward,DifferenriallPrivacyAward
#from nice.utils.optimization.reward import SparsityReward, ProximityReward, PlausibilityReward,DifferenriallPrivacyAward
#from typing import Optional
#import numpy as np

# =============================================================================
# Types and constants
# =============================================================================
CRITERIA_DIS = {'HEOM':HEOM}
CRITERIA_NRM = {'std':StandardDistance,
                'minmax':MinMaxDistance}
CRITERIA_REW = {'sparsity':SparsityReward,
                'proximity':ProximityReward,
                'plausibility':PlausibilityReward,
                'differentialprivacy':DifferenriallPrivacyAward}
