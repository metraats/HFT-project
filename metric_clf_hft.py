## metric
import numpy as np
from sklearn.metrics import confusion_matrix

def metric(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    weight_matrix = np.array([
                [1.9, -0.3, -2],
                [0,    0,    0],
                [-2, -0.3, 1.9]
    ])
    
    hit_matrix = conf_matrix * weight_matrix
    hit_matrix_sum = np.sum(hit_matrix)
    action_count = np.sum(conf_matrix[0]) + np.sum(conf_matrix[2])
    
    result = hit_matrix_sum / np.sqrt(action_count)
    return result