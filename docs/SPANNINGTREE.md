# Spanning Tree Enumeration

This program helps you create a unweighted, directed graph G and finds all of its spanning trees

## Getting Started

Once the prerequisites are completed this program can be run with the command "python find_spans.py". This will run a version of this program that enumerates spanning trees based on an example graph.

### Prerequisites

To understand this algorithm, the paper "Finding All Spanning Trees of Directed and Undirected Graphs" by Harold N. Gabow and Eugene W. Myers, is essential. It can be found and purchase with this link: http://epubs.siam.org/doi/abs/10.1137/0207024. All variable names are written according to this paper. 

## Dependencies

The requirements.txt file is also included with this folder. I can be run using command "pip install -r requirements.txt" while in the folder directory.

## Running the tests

The code can be run by running the file "find_spans.py". In find_spans.py a DiGraph is created and its spanning trees are enumerated. In order to run your own graph create the NetworkX version of your graph. Name it G and run the code. To run decompression and print out all the outputs uncomment the line "# dc.decompress(BLOCKSIZE, ORGANIZED_EDGES)" from find_all_spans.py

## Authors

* Berk Serbetcioglu
* Raffi Sevlian

## Acknowledgments

This code was created by converting code written by Raffi Sevlian in MATLAB to Python

