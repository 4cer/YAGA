def get_size(obj, seen=None):
    import sys
    """Recursively finds size of objects"""

    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0

    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)

    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])

    return size

def linear_gaussian_kernel(length: int, sig = 1.):
    import numpy as np
    ax = np.linspace(-(length - 1) / 2., (length - 1) / 2., length)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))

    return gauss / np.sum(gauss)


def mask_neighborhood(length, x0, size=1):
    import numpy as np
    mask = np.zeros(length, dtype=bool)
    for i in range(x0-size, x0+size+1):
        i = max(0,min(i,length-1))
        try:
            mask[i] = True
        except:
            pass
    return mask

def mask_bounds(length, x1, x2):
    import numpy as np
    mask1 = np.zeros(length, dtype=bool)
    mask2 = np.zeros(length, dtype=bool)
    mask3 = np.zeros(length, dtype=bool)
    for i in range(0,length):
        if i < x1:
            mask1[i] = True
        elif i <= x2:
            mask2[i] = True
        else:
            mask3[i] = True
    return [mask2, np.not_equal(mask1, mask3)]
    

def generate_random_vertices(count, bounds_x = [-1000.0, 1000.0], bounds_y = [-1000.0, 1000.0]):
    import numpy as np
    if count <= 0:
        raise ValueError("Count must be an integer greater than 0")
    if bounds_x[0] > bounds_x[1] or bounds_y[0] > bounds_y[1]:
        raise ValueError("Bounds positive must be greater than negative for each of axis")
    vertex_tuples = []
    for _ in range(count):
        vertex_tuples.append((np.random.uniform(bounds_x[0], bounds_x[1]), np.random.uniform(bounds_y[0], bounds_y[1])))
    return vertex_tuples

def generate_circle_vertices(resolution, radius, center = (0,0)):
    import numpy as np
    vertices = []
    for vert in range(resolution):
        theta = 2*np.pi * vert/resolution
        x = radius * np.cos(theta) + center[0]
        y = radius * np.sin(theta) + center[1]
        vertices.append((x.item(),y.item()))
    return vertices


def main():
    import numpy as np
    print(mask_neighborhood(12,11,2))
    [mask1, mask2, mask3] = mask_bounds(10, 3, 7)
    print(mask1)
    print(mask2)
    print(mask3)
    print(np.not_equal(mask1, mask3))
    pass


if __name__ == "__main__":
    main()