import random
import numpy as np
import pandas as pd
import random
import numpy as np
import pandas as pd
import time


def generate_random_dataset_linear_separable(size):
    x = []
    y = []
    target = []
    s_ = 0
    while s_ < size:
        # class zero
        x1 = random.uniform(-7, 7)
        y1 = random.uniform(-7, 7)
        if (y1-2*x1) < -1:
            target.append(1)        # class one
            x.append(x1)
            y.append(y1)
            s_ = s_+1
        elif (y1-2*x1) > 6:
            target.append(0)        # class one
            x.append(x1)
            y.append(y1)
            s_ = s_+1

    df_x = pd.DataFrame(data=x)
    df_y = pd.DataFrame(data=y)
    df_target = pd.DataFrame(data=target)
    data_frame = pd.concat([df_x, df_y], ignore_index=True, axis=1)
    data_frame = pd.concat([data_frame, df_target], ignore_index=True, axis=1)
    data_frame.columns = ['x', 'y', 'target']
    return data_frame


def generate_random_dataset_linear_separable_noise(size):
    x = []
    y = []
    target = []
    s_ = 0
    while s_ < size:
        # class zero
        x1 = random.uniform(-7, 7)
        y1 = random.uniform(-7, 7)
        if (y1-2*x1) < -1:
            target.append(1)        # class one
            x.append(x1)
            y.append(y1)
            s_ = s_+1
        if (y1-2*x1) > 6:
            target.append(0)        # class one
            x.append(x1)
            y.append(y1)
            s_ = s_+1
        elif (y1-2*x1) > -1 and (y1-2*x1) < 6:
            p = random.uniform(0, 1)
            if p < 0.3:
                target.append(1)        # class one
                x.append(x1)
                y.append(y1)
                s_ = s_+1
            else:
                target.append(0)        # class one
                x.append(x1)
                y.append(y1)
                s_ = s_+1

    df_x = pd.DataFrame(data=x)
    df_y = pd.DataFrame(data=y)
    df_target = pd.DataFrame(data=target)
    data_frame = pd.concat([df_x, df_y], ignore_index=True, axis=1)
    data_frame = pd.concat([data_frame, df_target], ignore_index=True, axis=1)
    data_frame.columns = ['x', 'y', 'target']
    return data_frame


def generate_random_dataset_circle(size, cx=0, cy=0):
    """ Generate a random dataset and that follows a quadratic  distribution
    """
    x = []
    y = []
    target = []
    for i in range(size):
        # class zero
        x1 = random.uniform(-6, 6)
        y1 = random.uniform(-6, 6)
        if (x1-cx)**2+(y1-cy)**2 < 9:  # or (x1-15)**2+(y1-15)**2 < 4:
            target.append(0)        # class one
            x.append(x1)
            y.append(y1)
        else:
            target.append(1)        # class one
            x.append(x1)
            y.append(y1)

    df_x = pd.DataFrame(data=x)
    df_y = pd.DataFrame(data=y)
    df_target = pd.DataFrame(data=target)
    data_frame = pd.concat([df_x, df_y], ignore_index=True, axis=1)
    data_frame = pd.concat([data_frame, df_target], ignore_index=True, axis=1)
    data_frame.columns = ['x', 'y', 'target']
    return data_frame


def generate_random_dataset_xor(size):
    X_xor = np.random.randn(size, 2)
    y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
    target = np.where(y_xor, 1, -1).reshape(-1, 1)
    df_target = pd.DataFrame(data=target)
    data_frame = pd.DataFrame(data=X_xor)
    data_frame = pd.concat([data_frame, df_target], ignore_index=True, axis=1)
    data_frame.columns = ['x', 'y', 'target']
    return data_frame


def unbalanced_data(size_1, size_2):
    x = np.random.randn(size_1+size_2, 2)
    target = np.zeros((size_1+size_2, 1))
    target[0:size_1] = 1
    data_frame = pd.DataFrame(data=x)
    df_target = pd.DataFrame(data=target)
    data_frame = pd.concat([data_frame, df_target], ignore_index=True, axis=1)
    data_frame.columns = ['x', 'y', 'target']
    return data_frame


def generate_random_dataset_xor_high_dimension(size):
    X_xor = np.random.randn(size, 20)
    y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
    target = np.where(y_xor, 1, -1).reshape(-1, 1)
    df_target = pd.DataFrame(data=target)
    data_frame = pd.DataFrame(data=X_xor)
    data_frame = pd.concat([data_frame, df_target], ignore_index=True, axis=1)
    columns = []
    for i in range(20):
        columns.append('x'+str(i))
    columns.append('target')
    data_frame.columns = columns

    return data_frame


def generate_random_dataset_high_dimension(size):
    X = np.random.randn(size, 10000)
    target = np.ones((size, 1), dtype=int)
    target[0:int(size/2)] = -1
    df_target = pd.DataFrame(data=target)
    data_frame = pd.DataFrame(data=X)
    data_frame = pd.concat([data_frame, df_target], ignore_index=True, axis=1)
    columns = []
    for i in range(10000):
        columns.append('x'+str(i))
    columns.append('target')
    data_frame.columns = columns

    return data_frame
