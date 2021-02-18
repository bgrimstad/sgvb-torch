"""
Created 13 February 2020
Bjarne Grimstad, bjarne.grimstad@gmail.com
"""

import torch





if __name__ == '__main__':

    print('Test prediction summary')
    import numpy as np
    a = np.linspace(0, 100, 100)
    b = np.linspace(0, 50, 100)
    c = np.stack([a, b]).T
    print(c)
    print(c.shape)
    t = torch.tensor(data=c)
    summary = prediction_summary(t)
    print(summary)



