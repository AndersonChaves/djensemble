import sys
from djensemble import DJEnsemble
from datetime import datetime
from core.configuration_manager import ConfigurationManager
from core.dataset_manager import DatasetManager
import time

def print_time():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    return time.time()

def run(query_configurations_path):
    start = print_time()
    config = ConfigurationManager(query_configurations_path)
    djensemble = DJEnsemble(config)
    ds = DatasetManager().loadDataset(config.get_configuration_value("dataset_path"))

    clustering_period = eval(config.get_configuration_value("clustering_period"))
    online_period     = eval(config.get_configuration_value("online_period"))
    query_region      = eval(config.get_configuration_value("query_region"))
    target_attribute       = config.get_configuration_value("target_attribute")

    clustering_dataset = DatasetManager().filter_by_date(ds, *clustering_period)
    validation_dataset = DatasetManager().filter_by_date(ds, *online_period)
    djensemble.run_offline_step(clustering_dataset.to_array(dim='target_attribute')[0, ...])
    result, result_by_tile = djensemble.run_online_step(validation_dataset.to_array(dim=target_attribute)[0, ...],
                                                        query_region)
    print("DJEnsemble:", result, result_by_tile )
    end = print_time()
    print("Total time: ", end - start, " seconds")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please inform the query configurations file path.\n")
        exit(1)
    query_configurations_path = sys.argv[1]
    #query_configurations_path = "queries/query-alerta-rio.config"
    run(query_configurations_path)
