import datetime
from dateutil.relativedelta import relativedelta

# general timeframe for the data to use
START_DATE = datetime.datetime(2014, 1, 1)
END_DATE = datetime.datetime(2020, 12, 31)

# constant batch size for all experiments
BATCH_SIZE = relativedelta(months=1)

# classification constant
CLASSES_TO_USE = ['Bulk carrier', 'General cargo/multipurpose',
                  'Oil tanker', 'Container', 'Chemical tanker',
                  'Fishing vessel', 'Other special activities',
                  'Tug', 'Offshore supply', 'Ro-Ro cargo', 'Gas carrier']


# general constants
PORTCALLS_FILEPATH = " " 

#constants for feature engineering
BINS = 10
CUTOFF_NEGATIVE_DURATION_PERCENTAGE = 9.84403128933072

NODE_CENTRALITIES = ['degree', 'in_degree', 'out_degree',
                        'strength', 'in_strength', 'out_strength',
                        'closeness', 'closeness_weighted',
                        'betweenness', 'betweenness_weighted',
                        'eigenvector', 'eigenvector_weighted'
                        ]