#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

import utils


class PPR:
    """ Class used for constructing an image classifier based on the Personalized Page Rank (PPR) algorithm

    Attributes
    ----------
    k : int
        The number of outgoing edges in the generated image-to-image similarity graph
    image_names : list
        The names of the images in the dataset passed in by the user
    test_names : list
        The names of the images to be classified
    x_train : list
        The feature vectors of the images in the training set
    y_train : list
        The labels of the images in the training set


    Methods
    -------
    compute_and_store_ii_similarity(self, i_features, i_names, output_file)
        Generates a matrix of the cosine similarity between all images in the dataset
    get_k_similarity_graph(sim_matrix)
        Generates a graph of the images where each image node is connected to the k most similar images
    ppr(i_names, adj_matrix, image1, image2=None, image3=None)
        Performs Personalized Page Rank (PPR) on the images using 1 or 3 images as the starting point and returns the dominance score of each image
    fit(x_train, y_train)
        Sets up the PPR classifier by calculating the image-to-image similarity for the images in the training set
    predict(x_test)
        Labels each of the images in the test set as dorsal or palmar based on the PPR results using the training data points
    """

    def __init__(self, k, image_names):
        """
        Parameters
        ----------
        k : int
            The number of outgoing edges in the generated image-to-image similarity graph
        image_names : list
            The names of the images in the dataset passed in by the user
        """
        self.k = k
        self.image_names = image_names
        self.test_names = []
        self.x_train = []
        self.y_train = []

    def compute_and_store_ii_similarity(self, i_features, i_names, output_file):
        """ Generates a matrix of the image-to-image cosine similarity between all images in the dataset

        Parameters
        ----------
        i_features : DataFrame
            Feature vectors of images in dataset
        i_names : list
            Names of images in dataset
        output_file:
            Name of the CSV file to which the results will be written
        """

        # Dictionary of cosine similarities to all other images for each image
        sim_dict = {}

        length = len(i_names)

        # Calculate cosine similarity for all images
        for i in range(length):
            sim_dict[i_names[i]] = utils.cosine_similarity_mv(i_features, i_features[i])

        # Create a data frame with column and row names corresponding to the images
        sim_df = pd.DataFrame.from_dict(sim_dict, orient='index')
        sim_df.columns = i_names

        # Write to csv
        sim_df.to_csv(output_file)

    # Get top k similarity graph for each image
    # Using the ii_similarity matrix, sort by distance, get top k, and prune the rest
    # Also getting and returning the adjacency matrix here since the same values
    #  for the graph can be used to generate the matrix
    def get_k_similartiy_graph(self, sim_matrix):
        """ Generates a graph of the images where each image node is connected to the k most similar images using the similarities as edge weights

        Parameters
        ----------
        sim_matrix : DataFrame
            Image-to-image similarity matrix for all images in the dataset
        """

        # Initializing graph, node, and adjacencyMatrix
        graph = {}
        # Offseting k by 1 to get itself + k images
        node = [None] * (self.k + 1)
        adj_matrix = np.zeros([len(sim_matrix), len(sim_matrix)])

        # Iterating through similarity matrix to get top k most similar images for each image
        for i in (sim_matrix):
            # Images sorted on similarity in descending order
            image_index = np.argsort(sim_matrix[i].values)[::-1][0:self.k]
            edge = np.sort(sim_matrix[i].values)[::-1][0:self.k]

            # get image name using image index
            for j in range(self.k):
                node[j] = sim_matrix.columns[image_index[j]]
                if j > 0:
                    adj_matrix[image_index[0]][image_index[j]] = edge[j]

            # node[0] is itself (distance = 1, root)
            # node[1:k] are the top k images (leaves with corresponding edge weights)
            graph[node[0]] = (node[1:self.k], edge[1:self.k])

        return graph, adj_matrix

   # Perform PPR on the top k similarity graph using 1 or 3 image IDs
    def ppr(self, i_names, adj_matrix, image1, image2=None, image3=None):
        """ Performs Personalized Page Rank (PPR) with random walks and restarts on the images using 1 or 3 images as the starting point and returns the dominance score of each image

        Depending on the task being tested, one or three image IDs may be passed in
        Parameters
        ----------
        graph : dictionary
            Graph connecting each image in the dataset to the k most similar images with edge weights equal to the similarity values
        image_names : list
            Names of the images in the dataset
        adjacencyMatrix : ndarray
            Matrix denoting which images each node in the graph is connected to and the weights of these connections
        image1 : str
            ID (filename) of the first image to use as starting point for PPR
        image2 : str, optional
            ID (filename) of the second image to use as starting point for PPR (default is None)
        image3 : str, optional
            ID (filename) of the third image to use as starting point for PPR (default is None)
        """

        # Pseudocode step 1: Restart vector should be initialized to all 0s except for 1 in position of query image
        restart_vector = np.zeros(len(i_names))

        # Multiple image IDs have been passed in for task 3
        if image2 is not None:
            image_index1 = np.where(i_names == image1)
            image_index2 = np.where(i_names == image2)
            image_index3 = np.where(i_names == image3)

            restart_vector[image_index1] = (1/3)
            restart_vector[image_index2] = (1/3)
            restart_vector[image_index3] = (1/3)

        # One image ID has been passed in for task 4
        else:
            image_index = np.where(i_names == image1)
            restart_vector[image_index] = 1

        # Pseudocode step 2: Normalize columns of Adjacency matrix so that they sum to 1
        #  Columns that add up to 0 originally are kept as 0
        col_sums = adj_matrix.sum(axis=0, keepdims=1)
        col_sums[col_sums == 0] = 1
        normal_adj = adj_matrix/col_sums

        # Pseudocode step 3: Initialize state probability vector to initial value of restart vector
        state_prob_vector = restart_vector

        # Pseudocode step 4: While state prob vector has not converged or max number of iterations not reached, perform page rank calculation
        maxIter = 50
        iterCount = 0
        c = 0.85
        while iterCount < maxIter:
            state_prob_vector = (1-c)*normal_adj.dot(state_prob_vector) + (c*restart_vector)
            iterCount = iterCount + 1
        return state_prob_vector

    def fit(self, x_train, y_train):
        """ Sets up the PPR classifier by calculating the image-to-image similarity for the images in the training set

        Parameters
        ----------
        x_train : list
            Feature vectors of images in training set
        y_train : list
            Labels (dorsal or palmar) of images in training set
        """

        x_train = x_train.astype(int)
        self.x_train = x_train
        self.y_train = y_train
        inames = self.image_names
        self.compute_and_store_ii_similarity(x_train, inames, '../proj3t4.csv')

    def predict(self, x_test):
        """ Labels each of the images in the test set as dorsal or palmar based on the PPR results using the training data points

        Parameters
        ----------
        x_test : list
            Feature vectors of images in test set
        """
        x_test = x_test.astype(int)
        y_pred = []

        # Iterating through each image in the test set:
        #   Appending each image individually to the training data to generate an updated graph
        #   Performing PPR on the updated graph to retrieve the Page Rank score for each image relative to the test image
        #   Using the sums of these scores for the dorsal and palmar images to generate a classification label
        #   In the event of a tie, the k (typically 5) most dominant images are used to determine the label
        for i in range(len(x_test)):
            temp_array = np.vstack([self.x_train, x_test[i]])
            temp_names = np.append(self.image_names, self.test_names[i])
            self.compute_and_store_ii_similarity(temp_array, temp_names, '../proj3t4.csv')
            sim_df = pd.read_csv('../proj3t4.csv', index_col=0)

            graph, adj_matrix = self.get_k_similartiy_graph(sim_df)
            state_prob_vector = self.ppr(graph, temp_names, adj_matrix, temp_names[-1])

            # First attempt to classify the image: comparing the sums of the Page Rank scores for all of the dorsal and palmar images
            dorsal_sum = 0
            palmar_sum = 0
            for j in range(len(state_prob_vector) - 1):
                if self.y_train[j] == 'dorsal':
                    dorsal_sum = dorsal_sum + state_prob_vector[j]
                elif self.y_train[j] == 'palmar':
                    palmar_sum = palmar_sum + state_prob_vector[j]
            if dorsal_sum > palmar_sum:
                y_pred.append('dorsal')
            elif palmar_sum > dorsal_sum:
                y_pred.append('palmar')

            # Tiebreaker: Taking the k most dominant images and classifying based on the label with the majority
            else:
                dorsal_count = 0
                palmar_count = 0
                image_index = np.argsort(state_prob_vector)[::-1][1:(self.k + 1)]
                for ind in image_index:
                    if self.y_train[ind] == 'dorsal':
                        dorsal_count = dorsal_count + 1
                    elif self.y_train[ind] == 'palmar':
                        palmar_count = palmar_count + 1

                if dorsal_count > palmar_count:
                    y_pred.append('dorsal')
                else:
                    y_pred.append('palmar')
        return y_pred
