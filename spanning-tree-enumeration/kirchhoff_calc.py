#this program is used to mathematicall find the number of spanning trees of a graph G using Kirchhoffs's Matrix Tree Theorem

import numpy as np
import networkx as nx
import types

#Create the laplace matrix of any graph G
def laplacian(G): 

    # lap_matrix = [[0 for x in range(len(G.nodes()))] for y in range(len(G.nodes()))] 
    dim = len(G.nodes())

    lap_matrix = np.zeros((dim,dim))

    for edges in G.edges():
        lap_matrix[edges[0]-1][edges[1]-1] = -1

    # print lap_matrix

    for node in range(dim):
        lap_matrix[node][node] = G.in_degree(node+1)

    # print lap_matrix
    # lap_matrix = [[0 for x in range(len(G))] for y in range(len(G))] 

    # for x in range (len(G.edges())):
    #     edge = G.edges()[x]
    #     lap_matrix[edge[0]][edge[1]]=-1
    #     lap_matrix[edge[1]][edge[0]]=-1
        
    # for x in range(dim):
    #     lap_matrix[x][x]= G.degree(x)                 

    return lap_matrix

#Get the number of spanning trees of graph G
def kirchhoff(G): 
    
    matrix=laplacian(G)

    matrix=np.delete(matrix,(0),0)    
    matrix=np.delete(matrix,(0),1)  
    
    # print matrix

    num_span=1
    
    num_span = np.linalg.det(matrix)

    return num_span
