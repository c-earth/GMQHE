#############################
# main file for running GMQHE
#############################

# import modules
from utils.data_prep import TB_model

# work flow settings
regenerate_tb_models = True
rebuild_speciments = True

# paths settings
raw_data_folder = './data/raw'
tb_models_folder = './data/TB_models'
speciments_folder = './data/speciments'

# load data
TB_models = TB_model.from_folder(raw_data_folder)

# generate plain structures

# etch holes

# attach leads

# measure conductances