import numpy as np

def quan2rpy(quan):
    rpy = np.zeros(3)
    rpy[0] = np.arctan2(2 * (quan[0] * quan[1] + quan[3] * quan[2]), 1 - 2 * (quan[1]**2 + quan[2]**2))
    rpy[1] = np.arcsin(2 * (quan[0] * quan[2] - quan[3] * quan[1]))
    rpy[2] = np.arctan2(2 * (quan[0] * quan[3] + quan[1] * quan[2]), 1 - 2 * (quan[2]**2 + quan[3]**2))
    return rpy

if __name__ == "__main__":
    x = 0.03450148079436214
    y = -0.004270304990170054
    z = -0.4555866695684873
    w = 0.8895123376489957
    quan = np.array([x, y, z, w])
    rpy = quan2rpy(quan)
    print(rpy)