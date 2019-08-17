import numpy as np

def _match(x, y):
    """Returns an array of the positions of (first) matches of y in x
    
    This is similar to R's `match` or Matlab's `[Lia, Locb] = ismember`
    
    See https://stackoverflow.com/a/8251757
    
    This assumes that all values in y are in x, but no check is made
    
    Parameters
    ----------
    x : 1-d array
    y : 1-d array
    
    Returns
    -------
    yindex : 1-d array
        np.all(x[yindex] == y) should be True
    """
    
    index = np.argsort(x)
    sorted_index = np.searchsorted(x, y, sorter=index)
    yindex = index[sorted_index]
    
    return yindex

def _match_hash(x, y, no_match=None):
    """Returns an array of the positions of (first) matches of y in x
    
    This is similar to R's `match` or Matlab's `[Lia, Locb] = ismember`
    
    As in R's `match`, a hash (i.e. dictionary) is built to do the
    mapping.
    """
    
    x_val2ind = {v: i for i, v in enumerate(x)}
    y_ind = [x_val2ind.get(v, no_match) for v in y]
    
    return y_ind
