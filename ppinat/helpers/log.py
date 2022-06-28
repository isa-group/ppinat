import pandas as pd
import pm4py
from pm4py.objects.conversion.log import converter as log_converter
import logging

logger = logging.getLogger(__name__)



def attribute_options(attrib_list, df):
    result_map = { k: values_limit(df[k].unique()) for k in attrib_list }
    return "\n ".join([f"- {k} ({result_map[k]})" for k in result_map])


def values_limit(values, limit = 3):
    values_size = len(values)
    return ", ".join(list(map(str,values))) if values_size <= limit else ", ".join(list(map(str, values[:limit]))) + f", and {values_size-limit} more"


class Log:
    def __init__(self, log, id_case = 'case:concept:name', time_column = 'time:timestamp', activity_column='concept:name'):
        if isinstance(log, pd.DataFrame):
            self.dataframe = log
            self.log = None
        elif isinstance(log, pm4py.objects.log.log.EventLog):
            self.log: pm4py.objects.log.log.EventLog = log
            self.dataframe = None
        else:
            raise RuntimeError("Invalid log")
        
        self.id_case = id_case
        self.time_column = time_column
        self.activity_column = activity_column

    def as_dataframe(self):
        if self.dataframe is None:
            self.dataframe = log_converter.apply(self.log, variant=log_converter.Variants.TO_DATA_FRAME).rename(columns={
                'case:concept:name': self.id_case,
                'time:timestamp': self.time_column,
                'concept:name': self.activity_column
            })
            self.dataframe[self.time_column] = pd.to_datetime(self.dataframe[self.time_column], utc=True)

        return self.dataframe
    
    def as_eventlog(self):
        if self.log is None:
            parameters = {
                log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: self.id_case
            }
            self.log = log_converter.apply(self.dataframe, 
                                           parameters=parameters, 
                                           variant=log_converter.Variants.TO_EVENT_LOG)

        return self.log


def load_log(file_name, id_case = 'case:concept:name', time_column = 'time:timestamp', activity_column = 'concept:name'):
    logger.info("Loading log...")
    
    if file_name.endswith('csv'):
        dataframe = pm4py.read_csv(file_name)
        log = Log(dataframe, id_case=id_case, time_column=time_column, activity_column=activity_column)
    else:
        log_xes = pm4py.read_xes(file_name)
        log = Log(log_xes, id_case=id_case, time_column=time_column, activity_column=activity_column)

    logger.info("Log successfully loaded")
    return log
