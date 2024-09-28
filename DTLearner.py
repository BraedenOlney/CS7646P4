""""""  		  	   		 	   		  		  		    	 		 		   		 		  
"""  		  	   		 	   		  		  		    	 		 		   		 		  
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		 	   		  		  		    	 		 		   		 		  
Note, this is NOT a correct DTLearner; Replace with your own implementation.  		  	   		 	   		  		  		    	 		 		   		 		  
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
  		  	   		 	   		  		  		    	 		 		   		 		  
import warnings

import math
  		  	   		 	   		  		  		    	 		 		   		 		  
import numpy as np  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
class DTLearner(object):  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    This is a decision tree learner object that is implemented incorrectly. You should replace this DTLearner with  		  	   		 	   		  		  		    	 		 		   		 		  
    your own correct DTLearner from Project 3.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    :param leaf_size: The maximum number of samples to be aggregated at a leaf, defaults to 1.  		  	   		 	   		  		  		    	 		 		   		 		  
    :type leaf_size: int  		  	   		 	   		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		 	   		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		 	   		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    def __init__(self, leaf_size=1, verbose=False):  		  	   		 	   		  		  		    	 		 		   		 		  
        """  		  	   		 	   		  		  		    	 		 		   		 		  
        Constructor method  		  	   		 	   		  		  		    	 		 		   		 		  
        """
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None

    def build_tree(self, data_x, data_y):
        # base cases

        # if all the Ys are the same make a leaf
        if np.all(data_y) == data_y[0]:
            # negative one for leaf and null/NA values
            return np.array([[-1, int(data_y[0]), -1, -1]]) # terminal error

        # if there are less than or equal to nodes as leaf_size remaining return as leaf
        if data_x.shape[0] <= self.leaf_size:
            y = data_y[0]
            if data_y.shape[0] > 1:
                y = data_y.mean()

            # negative one for leaf and null/NA values
            return np.array([[-1, y, -1,-1]])

        # recursive calls

        # combine x and y horizontally
        x_y = np.hstack((data_x, data_y))

        # find the correlations for all the X's

        # make y a one d numpy array
        y_flat = data_y.flatten()

        # make the array that will store the correlation coefficients
        corr = np.ones(data_x.shape[1])

        # for each column get the correlation to y as an absolute value
        for i in range(data_x.shape[1]):
            corr[i] = abs(np.corrcoef(data_x[:, i], y_flat)[0, 1])

        # index to use for correlation
        corr_index = np.argmax(corr)
        while math.isnan(corr[corr_index]):
            corr[corr_index] = 0
            corr_index = np.argmax(corr)
        # get the median of the best correlation index (Splitval
        median = np.median(data_x[:, corr_index])




        # split data on split value
        left = x_y[x_y[:, corr_index] <= median]
        right = x_y[x_y[:, corr_index] > median]

        # base case if either data set is empty
        if left.size==0 or right.size==0:
            y = data_y[0]
            if data_y.shape[0] > 1:
                y = data_y.mean()

            # negative one for leaf and null/NA values
            return np.array([[-1, y, -1, -1]])


        # slice the data to get the left x, y and right x,y

        # left data
        # left x is all but the last column
        left_x = left[:,:-1]
        # left y is only the last column
        left_y = left[:,[-1]]

        # right data
        # right x is all but the last column
        right_x = right[:, :-1]
        # right y is only the last column
        right_y = right[:, [-1]]



        # left call
        left_tree = np.array(self.build_tree(left_x, left_y))
        # right call
        right_tree = np.array(self.build_tree(right_x, right_y))

        # root
        root = np.array([[corr_index, median, 1, left_tree.shape[0] + 1]])
        # vertically stack the data
        return np.vstack((root,left_tree,right_tree))


    def author(self):
        """  		  	   		 	   		  		  		    	 		 		   		 		  
        :return: The GT username of the student  		  	   		 	   		  		  		    	 		 		   		 		  
        :rtype: str  		  	   		 	   		  		  		    	 		 		   		 		  
        """  		  	   		 	   		  		  		    	 		 		   		 		  
        return "bolney3"

    def study_group(self):
        return self.author()
  		  	   		 	   		  		  		    	 		 		   		 		  
    def add_evidence(self, data_x, data_y):  		  	   		 	   		  		  		    	 		 		   		 		  
        """  		  	   		 	   		  		  		    	 		 		   		 		  
        Add training data to learner  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
        :param data_x: A set of feature values used to train the learner  		  	   		 	   		  		  		    	 		 		   		 		  
        :type data_x: numpy.ndarray  		  	   		 	   		  		  		    	 		 		   		 		  
        :param data_y: The value we are attempting to predict given the X data  		  	   		 	   		  		  		    	 		 		   		 		  
        :type data_y: numpy.ndarray  		  	   		 	   		  		  		    	 		 		   		 		  
        """  		  	   		 	   		  		  		    	 		 		   		 		  

        if self.verbose:

            print("Starting Data\n")

            print("X: \n",data_x)

            print("\n\n")

            print("Y: \n",data_y)

        # make data_y a 2d array in the form of 1xn like data_x
        data_y = data_y.reshape(-1,1)

        # for building the tree call the build a tree method with the params and save to the return to the global
        self.tree = self.build_tree(data_x, data_y)
  		  	   		 	   		  		  		    	 		 		   		 		  
    def query(self, points):  		  	   		 	   		  		  		    	 		 		   		 		  
        """  		  	   		 	   		  		  		    	 		 		   		 		  
        Estimate a set of test points given the model we built.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
        :param points: A numpy array with each row corresponding to a specific query.  		  	   		 	   		  		  		    	 		 		   		 		  
        :type points: numpy.ndarray  		  	   		 	   		  		  		    	 		 		   		 		  
        :return: The predicted result of the input data according to the trained model  		  	   		 	   		  		  		    	 		 		   		 		  
        :rtype: numpy.ndarray  		  	   		 	   		  		  		    	 		 		   		 		  
        """
        # array to fill with results
        results = np.ones(points.shape[0])

        for i in range(points.shape[0]):
            curr_index = 0
            # while the tree is not at a leaf
            while self.tree[int(curr_index)][0] != -1:
                tree_val = int(self.tree[curr_index][0])
                # get the factor value for the current point and compare to the trees splitvalue
                if points[i][tree_val] > self.tree[curr_index][1]:
                    curr_index += int(self.tree[curr_index][3])
                else:
                    curr_index += int(self.tree[curr_index][2])
            # save the leafs Split Value (the prediction) into the results array

            results[i] = self.tree[curr_index][1]

        # with open("output.txt","w") as file:
        #     for i in self.tree:
        #         file.write(str(i))

        # np.savetxt("results.txt", results)

        return results
  		  	   		 	   		  		  		    	 		 		   		 		  

