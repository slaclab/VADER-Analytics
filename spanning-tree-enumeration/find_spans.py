import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import kirchhoff_calc as kc
import find_all_spans as sp
import time

if __name__ == "__main__":

###################################

# Code to import a csv adjacency matrix

	# fname = "sce_network.csv"
	# fh    = open(fname, "r")

	# a = np.loadtxt(fh, delimiter = ",")
	# G = nx.from_numpy_matrix(a, create_using = nx.DiGraph())

###################################	

# Code to generate a directed graph G by adding edges
	
# Graph from paper

	
	G=nx.DiGraph()

	G.add_edge(1, 2)
	G.add_edge(1, 3)
	G.add_edge(2, 3)
	G.add_edge(3, 2)
	G.add_edge(2, 4)
	G.add_edge(4, 3)

# Code below can be used to create own graph

	# G=nx.DiGraph()

	# G.add_edge(1, 2)
	# G.add_edge(2, 1)
	# G.add_edge(1, 3)
	# G.add_edge(3, 1)
	# G.add_edge(1, 4)
	# G.add_edge(4, 1)
	# G.add_edge(1, 5)
	# G.add_edge(5, 1)
	# G.add_edge(1, 6)
	# G.add_edge(6, 1)
	# G.add_edge(1, 7)
	# G.add_edge(7, 1)
	# G.add_edge(1, 8)
	# G.add_edge(8, 1)
	# G.add_edge(1, 9)
	# G.add_edge(9, 1)
	# G.add_edge(1, 10)
	# G.add_edge(10, 1)	

	# G.add_edge(2, 3)
	# G.add_edge(3, 2)
	# G.add_edge(2, 4)
	# G.add_edge(4, 2)
	# G.add_edge(2, 5)
	# G.add_edge(5, 2)
	# G.add_edge(2, 6)
	# G.add_edge(6, 2)
	# G.add_edge(2, 7)
	# G.add_edge(7, 2)
	# G.add_edge(2, 8)
	# G.add_edge(8, 2)
	# G.add_edge(2, 9)
	# G.add_edge(9, 2)
	# G.add_edge(2, 10)
	# G.add_edge(10, 2)	

	# G.add_edge(3, 4)
	# G.add_edge(4, 3)
	# G.add_edge(3, 5)
	# G.add_edge(5, 3)
	# G.add_edge(3, 6)
	# G.add_edge(6, 3)
	# G.add_edge(3, 7)
	# G.add_edge(7, 3)
	# G.add_edge(3, 8)
	# G.add_edge(8, 3)
	# G.add_edge(3, 9)
	# G.add_edge(9, 3)
	# G.add_edge(3, 10)
	# G.add_edge(10, 3)	

	# G.add_edge(4, 5)
	# G.add_edge(5, 4)
	# G.add_edge(4, 6)
	# G.add_edge(6, 4)
	# G.add_edge(4, 7)
	# G.add_edge(7, 4)
	# G.add_edge(4, 8)
	# G.add_edge(8, 4)
	# G.add_edge(4, 9)
	# G.add_edge(9, 4)
	# G.add_edge(4, 10)
	# G.add_edge(10, 4)	

	# G.add_edge(5, 6)
	# G.add_edge(6, 5)
	# G.add_edge(5, 7)
	# G.add_edge(7, 5)	
	# G.add_edge(5, 8)
	# G.add_edge(8, 5)
 	# G.add_edge(5, 9)
	# G.add_edge(9, 5)
	# G.add_edge(5, 10)
	# G.add_edge(10, 5)	

	# G.add_edge(6, 7)
	# G.add_edge(7, 6)	
	# G.add_edge(6, 8)
	# G.add_edge(8, 6)
	# G.add_edge(6, 9)
	# G.add_edge(9, 6)
	# G.add_edge(6, 10)
	# G.add_edge(10, 6)	

	# G.add_edge(7, 8)
	# G.add_edge(8, 7)
	# G.add_edge(7, 9)
	# G.add_edge(9, 7)
	# G.add_edge(7, 10)
	# G.add_edge(10, 7)	

	# G.add_edge(8, 9)
	# G.add_edge(9, 8)
	# G.add_edge(8, 10)
	# G.add_edge(10, 8)	

	# G.add_edge(9, 10)
	# G.add_edge(10, 9)	

###################################

# Code that expands initial node to the neighboring nodes n times 

	# n = 4

	# G = sp.expand_graph(G, n)

###################################

# Code to display graph G

	# pos = nx.spring_layout(G) #create a consistent layout for the
	# # nx.draw_networkx_nodes(G, pos, node_size=8, cmap=plt.get_cmap('jet'))
	# nx.draw_networkx_nodes(G, pos, node_size=8, cmap=plt.get_cmap('jet'), with_labels=True)
	# nx.draw_networkx_edges(G, pos, edge_color='r')
	# labels=nx.draw_networkx_labels(G,pos=nx.spring_layout(G))
	# plt.show()


###################################

# Code to calculate number of spanning trees mathematically

	# calc_spans = int(round(kc.kirchhoff(G)))
	# print "Calculated spanning trees: ", calc_spans

###################################

# Code to run function that finds all spanning trees

	start = time.time()

	sp.find_all_spans(G) #does all actions

	finish = time.time()

	print "Calculated in", finish - start, "seconds"
