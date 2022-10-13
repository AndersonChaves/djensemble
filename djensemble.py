from functools import reduce
import pandas as pd
import numpy as np
import time
import copy

import core.categorization as categorization
from core.dataset_manager import DatasetManager
from core.models_manager import ModelsManager
from core.series_generator import SeriesGenerator
from core.tile import Tile
from core.convolutional_model_invoker import ConvolutionalModelInvoker

MAXIMUM_COST = 999999

def normalize_data(model_tiles):
    tile_identifier_list = ["Tile" + str(i+1) for i in range(model_tiles.shape[1] - 1)]
    normalized_model_tiles = copy.deepcopy(model_tiles)
    max_value_rmse = normalized_model_tiles[tile_identifier_list].max().max()
    normalized_model_tiles[tile_identifier_list] = normalized_model_tiles[tile_identifier_list].div(max_value_rmse)
    return normalized_model_tiles

class DJEnsemble:
    def __init__(self, configuration_manager):
        self.configuration_manager = configuration_manager

        self.convolutional_models_path =          self.get_parameter_value("convolutional_models_path") + "/"
        self.temporal_length           =      int(self.get_parameter_value("temporal_length"))
        self.threshold                 =    float(self.get_parameter_value("threshold"))
        self.temporal_models_path      =          self.get_parameter_value("temporal_models_path")
        self.region                    =     eval(self.get_parameter_value("max_tile_area"))
        self.min_tile_length           =      int(self.get_parameter_value("min_tile_length"))
        self.number_of_samples         =      int(self.get_parameter_value("number_of_samples"))
        self.offset                    =      int(self.get_parameter_value("offset"))
        self.output_file_name          =          self.get_parameter_value("output_file_name")

        self.multiply_number_of_executions = self.get_parameter_value("multiply_number_of_executions") == 'S'
        self.normalize_values = self.get_parameter_value("normalize_values") == 'S'
        self.dataset_manager = DatasetManager()
        self.models_manager  = ModelsManager()
        self.models_manager.include_convolutional_models_from_directory(self.convolutional_models_path)
        self.models_manager.include_temporal_models_from_directory(self.temporal_models_path)
        self.models_manager.load_models()

    def run_offline_step(self, dataset):
        print("Executing offline stage... ")
        self.start_offline = time.time()
        self.dataset = dataset
        self.tiles = self.categorize_dataset(dataset)
        print("End of Categorization, time: ", time.time() - self.start_offline)
        self.update_cost_estimation_function()
        self.initialized = True
        print("End of offline stage, total time: ", time.time() - self.start_offline)

    def initialize_mock(self):
        dataset = DatasetManager().synthetize_dataset((10, 10, 10))
        self.tiles = {}

    def get_parameter_value(self, parameter):
        return self.configuration_manager.get_configuration_value(parameter)

    def categorize_dataset(self, dataset: np.array):
        # Input: Numpy Array
        # Output: Tiles Dictionary {id: (lat, long)}
        return categorization.categorize_dataset(dataset)


    def update_cost_estimation_function(self):
        print("Updating Cost Estimation Function")
        self.models_manager.update_eef()

    def run_online_step(self, validation_dataset, query: dict):
        print("Measuring core step ... ")
        self.start_online = time.time()
        result_by_tile = []

        print("--------------------- GET TILES INTERSECTING WITH QUERY ------------------------")
        query_tiles = self.get_tiles_intersecting_with_query(query, self.tiles)
        print(query_tiles)
        print("--------------------- GETTING ERROR ESTIMATION FOR EACH TILE -------------------")
        error_estimative = self.get_error_estimative(validation_dataset, query_tiles)

        print("--------------------- CALCULATE ALLOCATION COSTS -------------------------------")
        ensemble = self.get_lower_cost_combination(error_estimative)
        print('Ensemble Result: ', ensemble)

        print("--------------------------------- EVALUATE ERROR ----------------------------")
        for tile_id, i in zip(ensemble.keys(), range(len(query_tiles))):
            model_name = ensemble[tile_id][0]
            print("==>Evaluating error for tile ", tile_id, ": ", i+1 , "of ", len(query_tiles))
            print("Model: ", model_name)
            start = time.time()
            learner = self.models_manager.get_model_from_name(model_name)
            average_error = learner.evaluate(validation_dataset)

            print("Total time for tile evaluation: ", round(time.time() - start, 2), " seconds")
            result_by_tile.append(average_error)

        # Returns the total average rmses and the tiles rmses
        results = result_by_tile
        total_rmse = sum(results) / len(query_tiles)

        print("Total time (core): " + str(time.time() - self.start_online) + '\n')
        self.write_results(self.output_file_name, ensemble, result_by_tile, total_rmse)
        return total_rmse, results

    def get_tiles_intersecting_with_query(self, query: dict, tiles_list: dict):
        # Query ex. {"lat": (2, 5), "long": (3, 5)}
        intersecting_tiles = list(tiles_list.keys())
        for dim, value in query.items():
            q_min, q_max = value
            for tile_id in tiles_list.keys():
                tile = tiles_list[tile_id]
                tile_lower_bound = tile[dim][0]
                tile_upper_bound = tile[dim][1]
                if q_min > tile_upper_bound or \
                    q_max < tile_lower_bound:
                    if tile_id in intersecting_tiles:
                        intersecting_tiles.remove(tile_id)
                    break

        return intersecting_tiles

    def get_error_estimative(self, dataset, target_tiles):
        error_estimative = {}
        for tile_id in target_tiles:
            print("Estimating error for tile ", tile_id)
            t = Tile(tile_id, self.tiles[tile_id])
            if not type(dataset) is np.ndarray: # todo: Replace to check if is xarray # <class 'xarray.core.dataarray.DataArray'>
                dataset = dataset.to_numpy()
            if len(dataset.shape) == 2:
                dataset = np.reshape(dataset, dataset.shape + (1,))
            elif len(dataset.shape) == 4:
                dataset = np.reshape(dataset, dataset.shape[1:])
            error_estimative[tile_id] = self.models_manager.get_error_estimative_ranking(dataset, t)
        return error_estimative

    def get_lower_cost_combination(self, error_estimative):
        ensemble = {}
        for tile_id in error_estimative.keys():
            best_model, best_error = 'x', float('inf')
            for model_name, error in error_estimative[tile_id].items():
                if error < best_error:
                    best_model = model_name
                    best_error = error
            ensemble[tile_id] = (best_model,  best_error)
        return ensemble

    def write_results(self, output_file_name, best_allocation, result_by_tile, total_rmse):
        end = time.time()
        file = open(output_file_name +'.out', 'w')
        file.write("Best Allocation: \n")
        for model in best_allocation:
            file.write(str(model) + '\n')
        file.write("Best cost by tile: \n")
        file.write("Total RMSE: " + str(total_rmse) + '\n')
        file.write("Total time (core): " + str(end - self.start_online) + '\n')
        file.write("Results by tile: \n")
        for result in result_by_tile:
            file.write(str(result).replace('.', ',') + '\n')
