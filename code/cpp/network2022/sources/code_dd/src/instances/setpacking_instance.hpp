/*
 * --------------------------------------------------------
 * Set Packing instance
 * --------------------------------------------------------
 */

#ifndef SETPACKING_INSTANCE_HPP_
#define SETPACKING_INSTANCE_HPP_

#include <cstring>
#include <fstream>
#include <iostream>

#include "indepset_instance.hpp"

using namespace std;



//
// SetPacking Instance
//
struct SetPackingInstance {
	// Number of variables
	int n_vars;   
	// Number of constraints
	int n_cons;
	// Number of objective functions
	int n_objs;
	// Objective functions
	vector< vector<int> > objs;
	// Variables indices in constraint
	vector< vector<int> > vars_cons;
	// Matrix bandwidth
	int bandwidth;

	// Constructors
	SetPackingInstance() { }
	SetPackingInstance(const char* filename);

	// Create independent set instance from set packing
	IndepSetInst* create_indepset_instance();

	// Heuristic to minimize bandwidth of the matrix
	void minimize_bandwidth();
};




//
// Inline constructor
//
inline SetPackingInstance::SetPackingInstance(const char* inputfile) {
	ifstream input(inputfile);
	if (!input.is_open()) {
		cout << "Error: could not open file " << inputfile << endl;
		exit(1);
	}

	input >> n_vars;
	input >> n_cons;

	int val;

	input >> n_objs;
	objs.resize(n_objs);
	for (int o = 0; o < n_objs; ++o) {
		objs[o].resize(n_vars);
		for (int i = 0; i < n_vars; ++i) {
			input >> val;
			objs[o][i] = val;
		}
	}

	vars_cons.resize(n_cons);
	int n_vars_c;
	for (int c = 0; c < n_cons; ++c) {
		input >> n_vars_c;
		vars_cons[c].resize(n_vars_c);
		for (int i = 0; i < n_vars_c; ++i) {
			input >> vars_cons[c][i];
			vars_cons[c][i]--;
		}
	}

	// cout << "\nMultiobjective Set Packing Instance" << endl;
	// cout << "\tnum of variables = " << n_vars << endl;
	// cout << "\tnum of constraints = " << n_cons << endl;
	// cout << "\tnum of objectives = " << n_objs << endl;
	// cout << endl;

	//minimize_bandwidth();
}



// Create independent set instance from set packing
inline IndepSetInst* SetPackingInstance::create_indepset_instance() {

	IndepSetInst* inst = new IndepSetInst;
	inst->graph = new Graph(n_vars);
	Graph* graph = inst->graph;

	for (int c = 0; c < n_cons; ++c) {
		for (size_t i = 0; i < vars_cons[c].size(); ++i) {
			for (size_t j = i+1; j < vars_cons[c].size(); ++j) {
				inst->graph->add_edge(vars_cons[c][i], vars_cons[c][j]);
			}
		}
	}

	// cout << "\tAuxiliary graph for set packing:" << endl;
	// cout << "\t\tnumber of vertices: " << graph->n_vertices << endl;
	// cout << "\t\tnumber of edges: " << graph->n_edges << endl;

	// create complement mask of adjacencies
	inst->adj_mask_compl.resize(graph->n_vertices);
	for (int v = 0; v < graph->n_vertices; ++v) {

		inst->adj_mask_compl[v].resize(graph->n_vertices, true);
		for (int w = 0; w < graph->n_vertices; w++ ) {
			if (graph->is_adj(v,w) ) {
				inst->adj_mask_compl[v].set(w, false);
			}
		}

		// we assume here a vertex is adjacent to itself
		inst->adj_mask_compl[v].set(v, false);

	}

	return inst;
}

#endif /* SETPACKING_INSTANCE_HPP_ */
