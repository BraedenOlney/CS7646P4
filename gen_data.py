""""""  		  	   		 	   		  		  		    	 		 		   		 		  
"""  		  	   		 	   		  		  		    	 		 		   		 		  
template for generating data to fool learners (c) 2016 Tucker Balch  		  	   		 	   		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	   		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		 	   		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	   		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		 	   		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		 	   		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		 	   		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	   		  		  		    	 		 		   		 		  
or edited.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		 	   		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		 	   		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	   		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
Student Name: Braeden Olney  		  	   		 	   		  		  		    	 		 		   		 		  
GT User ID: bolney3		  	   		 	   		  		  		    	 		 		   		 		  
GT ID: 904056796			  	   		 	   		  		  		    	 		 		   		 		  
"""  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
import math  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
import numpy as np  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  

def best_lin_eq1(x):
    """
    Param: X values
    Returns: Y value
    """
    y = x * 100
    return y

def best_lin_eq2(x):
    """
    Param: X values
    Returns: Y value
    """
    y = x * 200
    return y

# this function should return a dataset (X and Y) that will work  		  	   		 	   		  		  		    	 		 		   		 		  
# better for linear regression than decision trees  		  	   		 	   		  		  		    	 		 		   		 		  
def best_4_lin_reg(seed=1489683273):  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    Returns data that performs significantly better with LinRegLearner than DTLearner.  		  	   		 	   		  		  		    	 		 		   		 		  
    The data set should include from 2 to 10 columns in X, and one column in Y.  		  	   		 	   		  		  		    	 		 		   		 		  
    The data should contain from 10 (minimum) to 1000 (maximum) rows.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    :param seed: The random seed for your data generation.  		  	   		 	   		  		  		    	 		 		   		 		  
    :type seed: int  		  	   		 	   		  		  		    	 		 		   		 		  
    :return: Returns data that performs significantly better with LinRegLearner than DTLearner.  		  	   		 	   		  		  		    	 		 		   		 		  
    :rtype: numpy.ndarray  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    np.random.seed(seed)
    seed_num = np.random.randint(10000, 100000)

    seed_num2 = np.random.randint(5, 500)

    x = np.zeros((100, 2))
    y = np.zeros((100,))
    for i in range(0,100):
        x[i, 0] = best_lin_eq1(seed_num+i*100)
        x[i, 1] = best_lin_eq2(seed_num2+i*100)
        y[i] = np.mean(x[i,:])

    np.savetxt("best4lin.txt", x)
    np.savetxt("best4liny.txt", y)

    return x, y


def best_dt_eq1(x1,x2,x3,x4,x5):
    """
    Param: X values
    Returns: Y value
    """
    sum = 0
    if x1>.5:
        sum+=1
    else:
        sum-=1

    if x2>.5:
        sum*=2
    else:
        sum/=2


    return sum


  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
def best_4_dt(seed=1489683273):  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    Returns data that performs significantly better with DTLearner than LinRegLearner.  		  	   		 	   		  		  		    	 		 		   		 		  
    The data set should include from 2 to 10 columns in X, and one column in Y.  		  	   		 	   		  		  		    	 		 		   		 		  
    The data should contain from 10 (minimum) to 1000 (maximum) rows.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    :param seed: The random seed for your data generation.  		  	   		 	   		  		  		    	 		 		   		 		  
    :type seed: int  		  	   		 	   		  		  		    	 		 		   		 		  
    :return: Returns data that performs significantly better with DTLearner than LinRegLearner.  		  	   		 	   		  		  		    	 		 		   		 		  
    :rtype: numpy.ndarray  		  	   		 	   		  		  		    	 		 		   		 		  
    """
    np.random.seed(seed)
    x = np.random.rand(100, 5)
    y = np.zeros((100,))
    for i in range(0, 100):

        y[i] = best_dt_eq1(x[i,0],x[i,1],x[i,2],x[i,3],x[i,4])

    np.savetxt("best4dt.txt", x)
    np.savetxt("best4dty.txt", y)
    return x, y
  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
def author():  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    :return: The GT username of the student  		  	   		 	   		  		  		    	 		 		   		 		  
    :rtype: str  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    return "bolney3"  # Change this to your user ID
  		  	   		 	   		  		  		    	 		 		   		 		  

