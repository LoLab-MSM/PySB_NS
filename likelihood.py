
# Likelihood functions will be problem specific
# and will depend on the available data.

def likelihood_function(ex_data, simulation):

    # setup: find indices for [bidM, bidU, bidT, smacA, smacC, smacM, parpC, parpU]
    names = ['BidM_obs', 'BidU_obs', 'BidT_obs', 'SmacA_obs', 'SmacC_obs', 'SmacM_obs', 'PARPC_obs', 'PARPU_obs']
    indices = [None for _ in range(len(names))]
    for i,each in enumerate(simulation[0]):
        for j,item in enumerate(names):
            if each == item:
                indices[j] = i

    # calculate likelihood in the form of negative mean squared error
    mse = 0.0
    n = 0
    for i,each in enumerate(ex_data):
        if i > 0:
            mse += (ex_data[i][1] - simulation[i][indices[0]]/(simulation[i][indices[0]] + simulation[i][indices[1]] + simulation[i][indices[2]]))**2
            mse += (ex_data[i][2] - simulation[i][indices[3]]/(simulation[i][indices[3]] + simulation[i][indices[4]] + simulation[i][indices[5]]))**2
            mse += (ex_data[i][3] - simulation[i][indices[6]]/(simulation[i][indices[6]] + simulation[i][indices[7]]))**2
            n += 3

    return -mse/n