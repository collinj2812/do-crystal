import numpy as np

# define some functions

def gaussian(x, mu, sigma):
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*((x-mu)/sigma)**2)



def find_x_at_percentile(x, y, percentile):
    y_normalized = np.cumsum(y) / np.sum(y)
    idx_below = np.max(np.where(y_normalized < percentile)[0])
    idx_above = idx_below + 1

    x0, x1 = x[idx_below], x[idx_above]
    y0, y1 = y_normalized[idx_below], y_normalized[idx_above]

    return x0 + (percentile - y0) * (x1 - x0) / (y1 - y0)