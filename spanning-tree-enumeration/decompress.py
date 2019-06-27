import os,sys
import glob
import pickle
import numpy as np

# Name:        decompress(BLOCKSIZE, ORGANIZED_EDGES, bit_list)
# Description: takes block data from files and puts them individually in L_check
# Parameter:   BLOCKSIZE = contains number of spanning trees per block
#              ORGANIZED_EDGES = contains mapping of boolean to edges
# Output:      None
def decompress(BLOCKSIZE, ORGANIZED_EDGES):

	fileName = "blockfiles/block-?*" 
	fileNum = len(glob.glob(fileName)) #count how many blocks there are

	spanningLength = len(ORGANIZED_EDGES)

	#run thru each block
	for block in range(fileNum):

		fileName = "blockfiles/block-%d" % block
		array_data = load(fileName)
		data = np.unpackbits(array_data, axis=1)

		L_check = []

		for row in range(BLOCKSIZE):

			if(len(L_check) > 0):
				print "Spanning Tree =", L_check #prints each spanning tree		

			L_check = []

			for column in range(spanningLength):
				if(data[row][column] == 1):
					L_check.append(ORGANIZED_EDGES[column])


# Name:        load(filename)
# Description: Does a pickle load from file 'filename' and returns it
# Parameter:   filename = name of file to perform pickle load on   
# Output:      data     = object to store file contents
def load(filename):

    fileObject   = open(filename,'rb+')           
    data = pickle.load(fileObject) 
    fileObject.close()

    return data
