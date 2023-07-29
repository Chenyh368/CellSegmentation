import numpy as np

def dir_to_class(y_dir, class_num):
    y_dir_class = []
    for i in range(len(y_dir)):
        x, y = y_dir[i]
        if x == -9999:
            y_vec = np.zeros(class_num)
            y_dir_class.append(y_vec)
        else:
            if y == 0 and x > 0:
                deg = np.arctan(float('inf'))
            elif y == 0 and x < 0:
                deg = np.arctan(-float('inf'))
            elif y == 0 and x == 0:
                deg = np.arctan(0)
            else:
                deg = np.arctan((x/y))
            if (x > 0 and y < 0) or (x <= 0 and y < 0):
                deg += np.pi
            elif x < 0 and y >= 0:
                deg += 2 * np.pi
            cla = int(deg / (2 * np.pi / class_num))
            y_vec = np.zeros(class_num)
            y_vec[cla] = 1
            y_dir_class.append(y_vec)
    return np.array(y_dir_class)