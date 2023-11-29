/*
 * --------------------------------------------------------
 * Set Covering instance
 * --------------------------------------------------------
 */

#ifndef SETCOVERING_INSTANCE_HPP_
#define SETCOVERING_INSTANCE_HPP_

#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;



//
// SetCovering Instance
//
struct SetCoveringInstance {
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
	// Constraints that a variable participates
	vector< vector<int> > cons_var;
	// Matrix bandwidth
	int bandwidth;

	// Constructors
	SetCoveringInstance() { }
	SetCoveringInstance(const char* filename);

	// Heuristic to minimize bandwidth of the matrix
	void minimize_bandwidth();
};




//
// Inline constructor
//
inline SetCoveringInstance::SetCoveringInstance(const char* inputfile) {
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
			objs[o][i] = (-1)*val;
		}
	}

	vars_cons.resize(n_cons);
	cons_var.resize(n_vars);
	int n_vars_c;
	for (int c = 0; c < n_cons; ++c) {
		input >> n_vars_c;
		vars_cons[c].resize(n_vars_c);
		for (int i = 0; i < n_vars_c; ++i) {
			input >> vars_cons[c][i];
			vars_cons[c][i]--;
			cons_var[vars_cons[c][i]].push_back(c);
		}
	}

	// cout << "\nMultiobjective Set Covering Instance" << endl;
	// cout << "\tnum of variables = " << n_vars << endl;
	// cout << "\tnum of constraints = " << n_cons << endl;
	// cout << "\tnum of objectives = " << n_objs << endl;
	// cout << endl;

	//minimize_bandwidth();
}

#endif /* SETCOVERING_INSTANCE_HPP_ */
