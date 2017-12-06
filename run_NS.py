
from PySB_NS import NS
# from test_model import model
from likelihood import likelihood_function as lh
import datetime
import sys
import imp

a = datetime.datetime.now()

model = imp.load_source('model', sys.argv[1]).model

outfile = sys.argv[1]
# outfile = '/home/mak/PycharmProjects/PySB_NS/'
NS(model, lh, 'earm_data.csv', outfile)

print a
print datetime.datetime.now()