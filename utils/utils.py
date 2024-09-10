import random
import numpy as np

def mk_colored(instance_img):
    instance_img = instance_img.astype(np.int32)
    H, W = instance_img.shape[0], instance_img.shape[1]
    pred_colored_instance = np.zeros((H, W, 3))

    nuc_index = list(np.unique(instance_img))
    nuc_index.pop(0)
    # nuc_index.pop(0)


    for k in nuc_index:
        pred_colored_instance[instance_img == k, :] = np.array(get_random_color())

    return pred_colored_instance

def get_random_color():
    ''' generate rgb using a list comprehension '''
    r, g, b = [random.random() for i in range(3)]
    return r, g, b