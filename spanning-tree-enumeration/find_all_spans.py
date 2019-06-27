import networkx as nx
import os,sys
from copy import copy
import pickle
import schema as sc
import numpy as np
import decompress as dc

RECURSION = 1          # keeps track of level of recursion GROW is on
L =[] 			       # list of edges in the previous found spanning tree 
L_BLOCK = np.zeros(0)  # stores full blocks of spanning trees
L_COUNTER = 0          # counter that increments with each spanning tree     
BLOCKSIZE = 10000      # size of how many spanning trees each block should contain
ORGANIZED_EDGES = []   # contains list in mapping order to array
ORGANIZED_LENGTH = 0   # contains number of undirected total edges
PRINT_FLAG = False     # flag that turns on comment printing
SPAN_LENGTH = 0        # length of each spanning tree

# Name:        find_all_spans(G)
# Description: initializes needed variables and runs recursive GROW function
#              that stores all spanning trees of graph G in blocks
# Parameter:   G = NetworkX graph whose spanning trees are to be found
# Output:      None
def find_all_spans(G):

	global L_COUNTER
	global ORGANIZED_EDGES
	global ORGANIZED_LENGTH
	global L_BLOCK
	global SPAN_LENGTH	

	SPAN_LENGTH = len(G.nodes()) - 1

	#initializations
	T = [] #contains all visited edges for that single spanning tree
	F = G.edges(1) #contains neighboring edges to T that can be expanded to
	F.reverse() #reversed for stack purposes

	#create empty folder to put blocks
	create_folder()

	#get mapping 
	[ORGANIZED_EDGES, lengthString] = sc.make_schema(G)

	ORGANIZED_LENGTH = len(ORGANIZED_EDGES)

	L_BLOCK = np.zeros((BLOCKSIZE, lengthString), dtype = int)

	#start recursive GROW
	GROW(G, T, F)

	if(not np.all(L_BLOCK==0)):
			package = np.packbits(L_BLOCK, axis=-1)
			blockNum = int(L_COUNTER / BLOCKSIZE)
			dump("blockfiles/block-%d" % blockNum, package)   


	#get all data from blocks
	# dc.decompress(BLOCKSIZE, ORGANIZED_EDGES) 

	print "Found spanning trees: ", L_COUNTER

# Name:        GROW(G, T, F):
# Description: recursive function that expands from a root node using DFS
#              with other constraints until it reaches spanning trees and
#              and stores them in blocks
# Parameter:   G = NetworkX graph whose spanning trees are to be found
#                  it is changed in this funnction
# 		   	   T = List of edges that are a part of the current spanning tree
# 			   F = List of edges T can be expanded with (neighboring edges)
# Output:      None  
def GROW(G, T, F):

	global L	
	global L_COUNTER
	global RECURSION
	global L_BLOCK

	if(PRINT_FLAG):
		TNodes = set() #will contain visited nodes

		for (node_A, node_B) in T:
			TNodes.add(node_A)
			TNodes.add(node_B)

		print "------------GROW CALLED------------"
		print "dfs:", RECURSION, "current T =", T
		print "dfs:", RECURSION, "current F =", list(reversed(F))
		print "dfs:", RECURSION, "|T| =", len(TNodes)

	#line 1 - check if a full spanning tree has been found
	# if(len(T) == len(G.nodes())-1):
	if(len(T) == SPAN_LENGTH):

		#spanning tree has been found
		L = copy(T) #L contains new spanning tree

		currentBlock = L_COUNTER / BLOCKSIZE
		currentSpan = L_COUNTER % BLOCKSIZE
		
		if((currentSpan == 0) and (L_COUNTER > 0)):
			package = np.packbits(L_BLOCK, axis=-1)
			dump("blockfiles/block-%d" % (currentBlock-1), package) 

			if(((currentBlock) % 10) == 0):
				print "At block:", (currentBlock) 
				if(((currentBlock) % 100) == 0):
					print "Spanning trees found so far:", L_COUNTER	

			L_BLOCK.fill(False)

		#populate L_BLOCK using binary mapping to ORGANIZED_EDGES
		for i in range(ORGANIZED_LENGTH):
			edge = ORGANIZED_EDGES[i]
			for (A, B) in L:
				if(A < B):
					if(edge == (A, B)):
						L_BLOCK[currentSpan, i] = 1
				else:
					if(edge == (B, A)):
						L_BLOCK[currentSpan, i] = 1	

		#counter for total spanning trees
		L_COUNTER = L_COUNTER + 1

	#line 2	
	else:
		FF = []

		#line 3
		b = False #while bridge test is False
		while(not b): 

			if(PRINT_FLAG):
				print "-----------------------------------"	
				print "dfs:", RECURSION, "start while loop"
				print "|adj| =", len(G.edges())
				print "dfs:", RECURSION, "FF =", list(reversed(FF))
				print "dfs:", RECURSION, "T =", T
				print "dfs:", RECURSION, "F =", list(reversed(F))

			#line 4 action
			e = F.pop() #e is the edge to be added to T
			v = e[1] #v has node e is directed at

			if(PRINT_FLAG):
				print "dfs:", RECURSION, "line 4: pop e=", e
				print "dfs:", RECURSION, "line 4: process v=", v

			#line 5
			T.append(e) #T is expanded

			if(PRINT_FLAG):
				print "dfs:", RECURSION, "line 5: append e"
				print "dfs:", RECURSION, "line 5: new T =", T

			#line 6
			F = update_F(G, T, F, v) 

			if(PRINT_FLAG == 1):
				print "dfs:", RECURSION, "line 6: push e in F =", list(reversed(F))

			#line 7
			[F_del, del_idx, F_sav, sav_idx] = clean_F(T, F, v)

			if(PRINT_FLAG):
				print "after l7 Fsav = ", F_sav
				print "after l7 Fdel = ", F_del
				print "after l7 sav_idx = ", sav_idx
				print "after l7 del_idx = ", del_idx	

			#line 8 
			if(PRINT_FLAG):
				print "calling recursion - dfs_iter:", RECURSION

			F_sav_copy = copy(F_sav)		

			RECURSION = RECURSION + 1 #keep track of recursion steps
			GROW(G, T, F_sav_copy) 
			RECURSION = RECURSION - 1

			if(PRINT_FLAG):
				print "------------GROW RETURNS------------"
				print "dfs:", RECURSION, "after-rec T =", T
				print "dfs:", RECURSION, "after-rec F =", list(reversed(F))
				print "dfs:", RECURSION, "adj-size = ", len(G.edges())


				print "b4 l9 Fsav = ", F_sav
				print "b4 l9 Fdel = ", F_del
				print "b4 l9 sav_idx = ", sav_idx
				print "b4 l9 del_idx = ", del_idx	


			#line 9
			pop_edges = restore_F(T, F, v) #find which edges should not be added back

			if(PRINT_FLAG):
				print "dfs:", RECURSION, "line 9 after restoring F =", list(reversed(F))

				# print "popedges = ", pop_edges
				print "Fsav = ", F_sav
				print "Fdel = ", F_del
				print "sav_idx = ", sav_idx
				print "del_idx = ", del_idx	

			#line 10
			#restore line 7 variables
			F = populate_F(T, F, F_del, del_idx, F_sav, sav_idx, pop_edges, v) 

			if(PRINT_FLAG):
				print "dfs:", RECURSION, "after restoring F =", list(reversed(F))

			#line 11
			#e is added to FF and removed from G and T
			[T, G, FF] = delete_e(G, T, e, FF) 

			if(PRINT_FLAG):
				print "dfs:", RECURSION, "delete e =", e
				print "dfs:", RECURSION, "after restoring T =", T
				print "dfs:", RECURSION, "remove edge", e
				print "dfs:", RECURSION, "push", e, "to FF:", list(reversed(FF))

			#line 12
			#check if e was a bridge if so complete while loop
			b = bridge_test(G, L, v)

		#put edges in FF back into F and G
		[FF, F, G] = reconstruct(G, F, FF)	

# Name:        update_F(G, T, F, v)
# Description: updates F after T has been expanded so it contains the edges
#              that are new neighbors of T - line 6
# Parameter:   G = NetworkX graph whose spanning trees are to be found
#                  it is changed in this funnction
# 		       T = List of edges that are a part of the current spanning tree
# 			   F = List of edges T can be expanded with (neighboring edges)
#              v = The node that was just reached when T was expanded
# Output:      F	 
def update_F(G, T, F, v):

	TNodes = set() #will contain visited nodes

	for (node_A, node_B) in T:
		TNodes.add(node_A)
		TNodes.add(node_B)

	v_edges = G.edges(v) #edges neighboring v

	for (back, head) in v_edges:
		if((back == v) and (head not in TNodes)):
			F.append((back,head))

	return F

# Name:        clean_F(T, F, v)
# Description: cleans F so that it contains no edges between two already 
#              visited edges - line 7
# Parameter:   T = List of edges that are a part of the current spanning tree
# 			   F = List of edges T can be expanded with (neighboring edges)
#              v = The node that was just reached when T was expanded
# Output:      F_del = list of edges deleted from F
#              del_idx = indexes of deleted edges (for restoration)
#              F_sav = list of edges saved to F
#			   sav_idx = indexes of saved edges (for restoration)
def clean_F(T, F, v):

	TNodes = set() #will contain visited nodes

	#populate TNodes
	for (node_A, node_B) in T:
		TNodes.add(node_A)
		TNodes.add(node_B)

	#create variables
	F_del   = []
	del_idx = []
	F_sav   = []
	sav_idx = []

	#go though F and seperate it into delete and save files based on if it is
	#pointing to v
	for i in range(len(F)):

		(old, new) = F[i]

		if((new == v)):
			F_del.append((old, new))
			del_idx.append(i)

			if(PRINT_FLAG):
				print "Line 7 - removing this element."

		else:
			F_sav.append((old, new))
			sav_idx.append(i)

			if(PRINT_FLAG):
				print "Line 7 - saving this element."

	if(PRINT_FLAG):
		print "inside l7 Fsav = ", F_sav
		print "inside l7 sav_idx = ", sav_idx
		

	return [F_del, del_idx, F_sav, sav_idx]	

# Name:        restore_F(T, F, v)
# Description: populate set pop_edges with all edges pointing from last visited node 
#              to non-T nodes - line 9
# Parameter:   T = List of edges that are a part of the current spanning tree
# 			   F = List of edges T can be expanded with (neighboring edges)
#              v = The node that was just reached when T was expanded
# Output:      pop_edges = edges not to add to F when restoring
def restore_F(T, F, v):	

	TNodes = set() #will contain visited nodes

	#populate TNodes
	for (node_A, node_B) in T:
		TNodes.add(node_A)
		TNodes.add(node_B)	

	#initialize object to store edges to pop_edges
	pop_edges = set()

	#populate pop_edges
	for (old, new) in F:
	 	if((new not in TNodes) and (old == v)):
			pop_edges.add((old, new))

	return pop_edges	

# Name:        populate_F(T, F, F_del, del_idx, F_sav, sav_idx, pop_edges, v)
# Description: populate F with all edges removed in update_F() - line 10
# Parameter:   T = List of edges that are a part of the current spanning tree
# 			   F = List of edges T can be expanded with (neighboring edges)
#			   F_del = list of edges deleted from F in clean_F()
#              del_idx = indexes of deleted edges (for restoration)
#              F_sav = list of edges saved to F in clean_F()
#			   sav_idx = indexes of saved edges (for restoration)
#			   pop_edges = edges that should not be added to new F
#              v = The node that was just reached when T was expanded
# Output:      F_merge = updated F with saves and deletes indexed in
def populate_F(T, F, F_del, del_idx, F_sav, sav_idx, pop_edges, v):	

	F_sav2   = [] #will store F_sav with pop_edges removed
	sav2_idx = [] #will contain new indexes 

	#populate F_sav2 and sav2_idx
	for i in range(len(F_sav)):
		if(F_sav[i] not in pop_edges):

			F_sav2.append(F_sav[i])
			sav2_idx.append(sav_idx[i])
			
	if(PRINT_FLAG):		
		print "popedges = ", pop_edges
		print "Fsav = ", F_sav
		print "Fdel = ", F_del
		print "sav2_idx = ", sav2_idx
		print "del_idx = ", del_idx	

	#total number of elements to index together
	total_index = len(F_sav2) + len(F_del)

	F_merge = [] #initialize object to store merged F_sav2 and F_del

	for index in range(total_index):

		#update F_merge with whichever index comes first
		if(index in sav2_idx): 
			F_merge.append(F_sav2[0])
			del F_sav2[0]
			del sav2_idx[0]

		elif(index in del_idx):	
			F_merge.append(F_del[0])
			del F_del[0]
			del del_idx[0]

		else: 
			print "ERROR"	

	return F_merge		

# Name: 	   delete_e(G, T, e, FF)
# Description: delete edge e from graph G and T. This is done to find spanning 
#              trees that dont include 'e'. e is saved in FF for later restoration
# Parameter:   G  = NetworkX graph whose spanning trees are to be found
# 		       T  = List of edges that are a part of the current spanning tree
# 			   e  = edge to be removed from graph and T, added to FF
#              FF = list that contains all removed edges 'e' for that recursion level
# Output:      T  = updated T without edge 'e' 
#			   G  = updated graph G with edge 'e' removed
#			   FF = updated FF with newly removed edge 'e'
def delete_e(G, T, e, FF):

	T.remove(e)
	G.remove_edge(*e)
	FF.append(e)

	return [T, G, FF]

# Name: 	   bridge_test(G, L, v)
# Description: checks if edge 'e' that was removed in delete_e() was the last was
#              the last way to get to the node 'e' was pointing at(v). Basically checks 
#              the edge removed made the graph into two seperate graphs
# Parameter:   G  = NetworkX graph whose spanning trees are to be found
# 		       L  = List of edges that make the last found full spanning tree
# 			   v  = node that was last found
# Output:      returns True of False based on success of test. True means spanning trees 
#              left in G that contains v
def bridge_test(G, L, v):

	# turn edges in L into a graph
	L_graph = nx.DiGraph()
	L_graph.add_edges_from(L)

	# store descendents of v in L graph as D
	D = nx.descendants(L_graph, v)

	# will contain all edges of G pointing at v
	w_nb_v = set() # w neighbor v

	for (x,y) in G.edges():
		if(y == v):
			w_nb_v.add(x)

	#check if w_nb_v and D overlap
	set_diff = w_nb_v - D	

	if len(set_diff) == 0:
		return True
	
	return False	

# Name: 	   reconstruct(G, F, FF)
# Description: restores edges removed from graph (FF) back into G and F
# Parameter:   G  = NetworkX graph whose spanning trees are to be found
# 		       F  = F = List of edges T can be expanded with (neighboring edges)
# 			   FF = contains edges removed from G in the previous recursion
# Output:      G  = updated graph G with removed edges re-added
#			   F  = updated F with removed edges from graph re-added
#			   FF = emptied list FF
def reconstruct(G, F, FF):	

	C = copy(FF) #so object isn't manipulated inside own for loop

	#pop each edge from FF and add to G and F
	for edges in C:
		chosen = FF.pop() 
		F.append(chosen)
		G.add_edge(*chosen)

	return [FF, F, G]	

# Name: 	   create_folder()
# Description: Create a folder named "blockfiles" and empty it if already exists
# Parameter:   None
# Output:      None
def create_folder():

	#create blockfiles folder
    newpath = r'blockfiles'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    
    #empty folder
    dirPath  = "blockfiles"
    fileList = os.listdir(dirPath)
    for fileName in fileList:
        os.remove(dirPath+"/"+fileName)

# Name:        dump(filename, data)
# Description: Does a pickle dump of object 'data' into file 'filename'.
# Parameter:   filename = name of file to store pickle dump
# 			   data     = object to store in file
# Output:      None
def dump(filename, data):

	#open file as a writable binary file, perform dump, close it
    fileObject   = open(filename,'wb+')            
    pickle.dump(data,fileObject)
    fileObject.close()

# Name:        expand_graph(G, n)
# Description: Expands the root node to the neighboring nodes n times
# Parameter:   G = NetworkX graph
# 			   n = number of times to expand
# Output:      G = expanded version of same graph
def expand_graph(G, n):

	print "G has:", len(G.nodes()), "nodes"

	# number of times to expand
	for expand in range(n):
	
		nextEdges = G.edges(1)

		print "Root Neighbors:", nextEdges
		
		# print G.edges()

		print "expanding"

		# merges with all neighboring nodes
		for edge in nextEdges:
			G = nx.contracted_edge(G, edge, self_loops=False)

		print "G has:", len(G.nodes()), "nodes"
		# print G.edges()


	return G

