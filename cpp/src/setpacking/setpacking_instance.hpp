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

#include "../indepset/indepset_instance.hpp"

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
	vector< vector<int> > objs_canonical;

	// Variables indices in constraint
	vector< vector<int> > vars_cons;
	vector< vector<int> > vars_cons_canonical;

	// Matrix bandwidth
	int bandwidth;

	// Constructors
	SetPackingInstance() { }
	SetPackingInstance(const char* filename);

	// Create independent set instance from set packing
	IndepSetInst* create_indepset_instance();

	// Generate instance LP - Kirlik model
	void createLP_Kirlik(string lpfilename);

	// Generate instance LP - Melihozen model
	void createLP_Melihozen(string lpfilename);

	// Heuristic to minimize bandwidth of the matrix
	void minimize_bandwidth();

	void reset_order(vector<int> new_order);
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
	objs_canonical.resize(n_objs);

	for (int o = 0; o < n_objs; ++o) {
		objs[o].resize(n_vars);
		objs_canonical[o].resize(n_vars);
		for (int i = 0; i < n_vars; ++i) {
			input >> val;
			objs[o][i] = val;
			objs_canonical[o][i] = val;
		}
	}

	vars_cons.resize(n_cons);
	vars_cons_canonical.resize(n_cons);
	int n_vars_c;
	for (int c = 0; c < n_cons; ++c) {
		input >> n_vars_c;
		vars_cons[c].resize(n_vars_c);
		vars_cons_canonical[c].resize(n_vars_c);
		for (int i = 0; i < n_vars_c; ++i) {
			input >> vars_cons[c][i];
			vars_cons[c][i]--;
			
			vars_cons_canonical[c][i] = vars_cons[c][i];
		}
	}

	cout << "\nMultiobjective Set Packing Instance" << endl;
	cout << "\tnum of variables = " << n_vars << endl;
	cout << "\tnum of constraints = " << n_cons << endl;
	cout << "\tnum of objectives = " << n_objs << endl;
	cout << endl;

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

	cout << "\tAuxiliary graph for set packing:" << endl;
	cout << "\t\tnumber of vertices: " << graph->n_vertices << endl;
	cout << "\t\tnumber of edges: " << graph->n_edges << endl;

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



// Generate instance LP - Kirlik model
inline void SetPackingInstance::createLP_Kirlik(string lpfilename) {
	ofstream out(lpfilename.c_str());   
	out << "\\" << n_objs << endl;
	out << "Minimize\n obj: z";
	out << endl;
	out << "Subject To" << endl;

	for (int o = 0; o < n_objs; ++o) {
		out << "\\New objective is defined" << endl;
		for (int i = 0; i < n_vars; ++i) {
			out << " - " << objs[o][i] << " x" << i << " ";
		}
		out << " <= " << 1 << endl << endl;
	}

	out << "\\Set Packing Constraint" << endl;
	for (int c = 0; c < n_cons; ++c) {
		out << "x" << vars_cons[c][0];
		for (size_t i = 1; i < vars_cons[c].size(); ++i) {
			out << " + " << "x" << vars_cons[c][i];
		}
		out << " <= " << 1 << endl;
		out << endl;        
	}

	out << "Bounds" << endl;
	out << " z = 1" << endl;
	for (int i = 0; i < n_vars; ++i) {
		out << "0 <= x" << i << " <= 1" << endl;
	}


	out << "\\Integer constraints" << endl;
	out << "binaries" << endl;
	out << " ";
	for (int i = 0; i < n_vars; ++i) {
		out << "x" << i << " ";
	}
	out << endl;

	out << "end" << endl;

	out.close();
}


//
// Generate Melihozen instance LP
//
inline void SetPackingInstance::createLP_Melihozen(string lpfilename) {
	ofstream out(lpfilename.c_str());   
	out << "maximize 0";
	out << endl;
	out << "subject to" << endl;

	out << "\\Set Packing Constraint" << endl;
	for (int c = 0; c < n_cons; ++c) {
		out << "x" << vars_cons[c][0];
		for (size_t i = 1; i < vars_cons[c].size(); ++i) {
			out << " + " << "x" << vars_cons[c][i];
		}
		out << " <= " << 1 << endl;
		out << endl;        
	}

	for (int o = 0; o < n_objs; ++o) {
		out << "\\New objective is defined" << endl;
		out << objs[o][0] << " x" << 0;
		for (int i = 1; i < n_vars; ++i) {
			out << " + " << objs[o][i] << " x" << i << " ";
		}
		out << " > " << (o+1) << endl << endl;
	}

	out << "\\Integer constraints" << endl;
	out << "binaries" << endl;
	out << " ";
	for (int i = 0; i < n_vars; ++i) {
		out << "x" << i << " ";
	}
	out << endl;

	out << "end" << endl;

	out.close();
}

inline void SetPackingInstance::reset_order(vector<int> new_order){
	for (int c = 0; c < n_cons; ++c) {
		for (size_t i = 0; i < vars_cons[c].size(); ++i) {
			vars_cons[c].push_back( new_order[vars_cons_canonical[c][i]] );			
		}
	}
	
	for (int p = 0; p < n_objs; ++p) {
		for (int j = 0; j < n_vars; ++j) {
			objs[p][new_order[j]] = objs_canonical[p][j];			
		}
	}
}

#endif /* SETPACKING_INSTANCE_HPP_ */
