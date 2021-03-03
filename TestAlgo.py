import numpy as np
from numpy.lib import type_check

def perceptronWithOffset(feature_matrix, labels, T):
    theta = [-1,1]
    theta_0 = -1
    for t in range(T):
        for i in range(5):
            x = feature_matrix[i]
            loss = (np.matmul ( x , np.transpose(theta)) + theta_0) * labels[i]
            if (loss <= 0):
                theta = theta + labels[i] * feature_matrix[i]
                theta_0 = theta_0 + labels[i]
                print(theta)
                print(theta_0)


def perceptron(feature_matrix, labels, T):
    theta = [0,0,0]    
    for t in range(T):
        for i in range(3):
            x = feature_matrix[i]
            loss = (np.matmul ( x , np.transpose(theta))) * labels[i]
            if (loss <= 0):
                theta = theta + labels[i] * feature_matrix[i]                
                print(theta)


def empiricalRiskHingeLoss (feature_matrix , labels , theta):
    r = 0
    i = 0
    for row in feature_matrix:
        loss = 0
        z = labels[i] - np.matmul(row , theta)
        if (z < 1):
            loss = 1 - z
        r = r + loss
        i += 1
    
    return r / 4


def empiricalRiskSquaredError (feature_matrix , labels , theta):
    r = 0
    i = 0
    for row in feature_matrix:
        loss = 0
        z = labels[i] - np.matmul(row , theta)
        loss = np.math.pow(z , 2)/2
        r = r + loss
        i += 1
    
    return r / 4                   


def main ():
    data = np.array([[1,0,1],[1,1,1],[1,1,-1],[-1,1,1]])
    labels = [2,2.7,-0.7,2]
    theta = [0,1,2]
    er_hingeLoss = empiricalRiskHingeLoss(data , labels , theta)
    er_squaredError = empiricalRiskSquaredError (data , labels , theta)

    print ("Empirical Risk hinge loss " + str(er_hingeLoss) )
    print ("Empirical Risk square error " + str(er_squaredError) )


main()                


