import os
base= os.getcwd()
data_route = 'datasets/'
class Flags:
    def __init__(self, directory=None):

        if directory is None:
            base_route=os.path.join(base, data_route)
        else:
            base_route=directory

        self.logdir= '/tmp/bandits/'#, 'Base directory to save output')
        self.mushroom_data=os.path.join(base_route,'mushroom.data')
            #,'Directory where Mushroom data is stored.')
        self.financial_data=os.path.join(base_route,'raw_stock_contexts')
            #,'Directory where Financial data is stored.')
        self.jester_data=os.path.join(base_route,'jester_data_40jokes_19181users.npy')
        self.statlog_data=os.path.join(base_route, 'shuttle.trn')
        self.adult_data=os.path.join(base_route,'adult.full')
        self.covertype_data=os.path.join(base_route, 'covtype.data')
        self.census_data=os.path.join(base_route,'USCensus1990.data.txt')
