#############################
# main file for running GMQHE
#############################

# import modules
from utils.data_prep import TB_model
from utils.build import create_specimens, lead_attaching

# work flow settings
regenerate_tb_models = True
rebuild_specimens = True

# paths settings
raw_data_folder = './data/raw'
tb_models_folder = './data/TB_models'
specimens_folder = './data/specimens'

# load data
TB_models = TB_model.from_folder(raw_data_folder)

# specify orders
orders = {}
lead_orders = {}

# build specimens
specimens = create_specimens(orders)

# attach leads
specimens = lead_attaching(specimens, lead_orders)

# simulation

# visualization