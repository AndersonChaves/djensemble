import math
import numpy as np
import h5py
import xarray as xr
from .tile import Tile
import os.path

class DatasetManager:
    def resizeDataset(self, x, y, data):
        while not (data.shape[2] % x == 0 and data.shape[3] % y == 0):
            if data.shape[2] % x != 0:
                # The size of the dimension that must be extended
                #  so that the final size is multiple of the models input.
                # Ex. mx=3, tx = 10: ceil(10/3)*3-10=4*3-10=2
                ext_x = abs(math.ceil(data.shape[2] / x) * x - data.shape[2])

                # Duplicates the last column of the data to fit the models input size
                # Ex. x y ===> x y y
                #     z w      z w w
                data = np.concatenate((data, data[:, :, -ext_x:, :, :]), axis=2)

            if data.shape[3] % y != 0:
                ext_y = abs(math.ceil(data.shape[3] / y) * y - data.shape[3])
                data = np.concatenate((data, data[:, :, :, -ext_y:, :]), axis=3)
        return data

    def loadDataset(self, dataPath):
        if os.path.isfile(dataPath):
            ds = xr.load_dataset(dataPath)
            ds = ds.sortby('time')
        else:
            print("Data Path " + dataPath + "is empty.")
            raise()
        return ds

    def filter_by_date(self, ds, lower_date, upper_date):
        filtered_dataset = ds.sel(time=slice(lower_date, upper_date), drop=True)
        filtered_dataset['rain'] = filtered_dataset['rain'].fillna(0)
        return filtered_dataset

    def loadTemperatureDataset(self, dataPath):
        with h5py.File(dataPath) as f:
            dataset = f['real'][...]
        return dataset

    def get_data_from_tile(self, dataset: np.array, tile: Tile):
        sx, sy = tile.get_start_coordinate()
        ex, ey = tile.get_end_coordinate()
        return dataset[:, sx:ex+1, sy:ey+1]

    def synthetize_dataset(self, shape, noise=0):
        time, lat, long = shape

        series = []
        for i in range(lat // 3 * long):
            series.append([k for k in range(time)])
        for i in range(lat // 3 * long):
            series.append([3 - (k % 3) for k in range(time)])
        for i in range((lat // 3 + lat % 3) * long):
            series.append([10 for _ in range(time)])

        array = np.reshape(np.array(series), (lat, long, time))
        array = np.swapaxes(array, 0, 2)

        if noise > 0:
            noise = np.random.normal(0, noise, array.shape)
            array = array + noise
        return array