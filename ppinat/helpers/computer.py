from ppinot4py.computers import measure_computer, LogConfiguration
import logging

logger = logging.getLogger(__name__)

class MeasureComputer:
    def __init__(self, log):
        self.dataframe = log.as_dataframe()
        self.log_configuration = LogConfiguration(id_case=log.id_case, time_column=log.time_column)

    def compute(self, metric, time_grouper=None):
        return measure_computer(
            metric,
            self.dataframe, 
            log_configuration=self.log_configuration,
            time_grouper=time_grouper
        )