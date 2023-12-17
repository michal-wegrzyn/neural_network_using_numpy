import numpy as np

class DatasetAugmentation:
    @staticmethod
    def shift(dataset, shifts_coordinates, color=0.0, reshape_data=None, reshape_result=None):
        coordinates_count = len(shifts_coordinates[0])
        assert all([len(shift_coords)==coordinates_count for shift_coords in shifts_coordinates])
        
        min_coords = [0]*coordinates_count
        start_coords = [0]*coordinates_count
        for shift_coords in shifts_coordinates:
            for pos in range(coordinates_count):
                min_coords[pos] = min(min_coords[pos], shift_coords[pos])
                start_coords[pos] = - min_coords[pos]

        max_coords = [0]*coordinates_count
        for shift_coords in shifts_coordinates:
            for pos in range(coordinates_count):
                max_coords[pos] = max(max_coords[pos], shift_coords[pos])
        
        res = []

        for data in dataset:
            if not reshape_data is None:
                data[0] = np.reshape(data[0], reshape_data)
            array_shape = list(data[0].shape)
            assert len(array_shape) == coordinates_count
            for i in range(coordinates_count):
                array_shape[i] += max_coords[i] - min_coords[i]
            
            arr = np.full(array_shape, color)
            arr[tuple([slice(start, start+size) for start, size in zip(start_coords, data[0].shape)])] = data[0]
            
            for shift_coords in shifts_coordinates:
                arr_slice = arr[tuple([slice(start+sh, start+size+sh) for start, size, sh in zip(start_coords, data[0].shape, shift_coords)])]
                if not reshape_result is None:
                    arr_slice = arr_slice.reshape(reshape_result)
                res.append([arr_slice, data[1]])
        
        return res
    
    @staticmethod
    def flip(dataset, flips, reshape_data=None, reshape_result=None):
        if type(flips) != list:
            flips = list(flips)
        for i, axises in enumerate(flips):
            if type(axises) not in [tuple, int]:
                flips[i] = tuple(flips[i])

        res = []
        for data in dataset:
            if not reshape_data is None:
                data[0] = np.reshape(data[0], reshape_data)
            for flip in flips:
                res.append([np.flip(data[0], flip).reshape(reshape_result), data[1]])
        
        return res