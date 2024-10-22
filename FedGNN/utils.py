import numpy as np
import pandas as pd
import random
import Setting


def laplace_mech(v, sensitivity, epsilon):
    v['embeddings'] += np.random.laplace(loc=0, scale=sensitivity/epsilon)
    return v

