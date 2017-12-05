
from os import walk, getcwd, listdir
import re

models = []
current_path = getcwd()
for each in listdir(current_path):
    if re.match(r'model.*.py\Z', each):
        models.append(each)

for each in models:
    filename = 'run_' + each
    f = open(filename, 'w+')
    f.write('\nfrom PySB_NS import NS\n')
    f.write('from ' + each[:-3] + ' import model\n')
    f.write('from likelihood import likelihood_function as lh\n\n')
    f.write('import datetime\n')
    f.write('a = datetime.datetime.now()\n\n')
    f.write('NS(model, lh, \'earm_data.csv\')\n\n')
    f.write('print \'start time\', a\n')
    f.write('print \'end time\', datetime.datetime.now()')
    f.close()
