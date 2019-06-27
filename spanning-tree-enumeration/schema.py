import networkx as nx
import pickle

# Name:        make_schema(G)
# Description: creates a mapping schema where each edge is represented by a
#              single bit
# Parameter:   G       = NetworkX graph whose spanning trees are to be found
# Output:      exploredEdge = list that contains each edge of the graph
def make_schema(G):

	numEdges = len(G.edges()) # number of edges in graph
	lengthString = 0 # counts number of undirected edges
	exploredEdge = [] # will contain which bit maps to which edge

	#store all edges (A, B) where A < B in list exploredEdge
	for (A, B) in G.edges(): 

		if(A > B): 
			if((B, A) not in exploredEdge):
				lengthString = lengthString + 1
				exploredEdge.append((B, A))

		else:	
			if((A, B) not in exploredEdge):
				lengthString = lengthString + 1
				exploredEdge.append((A, B))

	return exploredEdge, lengthString

# Name:        dump(filename, data)
# Description: Does a pickle dump stores data in file named 'filename'
# Parameter:   filename = name of file to perform pickle dump on   
#			   data     = object to store in file
# Output:      None
def dump(filename, data):

	#open file as a writable binary file, perform dump, close it
    fileObject   = open(filename,'wb+')            
    pickle.dump(data,fileObject)
    fileObject.close()