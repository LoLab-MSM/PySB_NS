
from PySB_NS import NS
from model_3 import model
from likelihood import likelihood_function as lh

NS(model, lh, 'earm_data.csv')