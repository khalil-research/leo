/*
 * --------------------------------------------------------
 * Set Packing instance
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
	vector< vector<int> > objs_canonical;

	// Variables indices in constraint
	vector< vector<int> > vars_cons;
	vector< vector<int> > vars_cons_canonical;

	// Constraints that a variable participates
	vector< vector<int> > cons_var;
	vector< vector<int> > cons_var_canonical;

	// Matrix bandwidth
	int bandwidth;

	// Constructors
	SetCoveringInstance() { }
	SetCoveringInstance(const char* filename);

	// Generate instance LP - Kirlik model
	void createLP_Kirlik(string lpfilename);

	// Generate instance LP - Melihozen model
	void createLP_Melihozen(string lpfilename);

	// Heuristic to minimize bandwidth of the matrix
	void minimize_bandwidth();

	// Reset instance order based on the input		
	void reset_order(vector<int> new_order);
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
	cout << "Here";

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

	objs_canonical = objs;
	vars_cons_canonical = vars_cons;
	cons_var_canonical = cons_var;

	cout << "\nMultiobjective Set Covering Instance" << endl;
	cout << "\tnum of variables = " << n_vars << endl;
	cout << "\tnum of constraints = " << n_cons << endl;
	cout << "\tnum of objectives = " << n_objs << endl;
	cout << endl;

	//minimize_bandwidth();
}




// Generate instance LP - Kirlik model
inline void SetCoveringInstance::createLP_Kirlik(string lpfilename) {
	ofstream out(lpfilename.c_str());   
	out << "\\" << n_objs << endl;
	out << "Maximize\n obj: z";
	out << endl;
	out << "Subject To" << endl;

	for (int o = 0; o < n_objs; ++o) {
		out << "\\New objective is defined" << endl;
		for (int i = 0; i < n_vars; ++i) {
			out << "+" << (-1)*objs[o][i] << " x" << i << " ";
		}
		out << " <= " << 1 << endl << endl;
	}

	out << "\\Set Covering Constraint" << endl;
	for (int c = 0; c < n_cons; ++c) {
		out << "x" << vars_cons[c][0];
		for (size_t i = 1; i < vars_cons[c].size(); ++i) {
			out << " + " << "x" << vars_cons[c][i];
		}
		out << " >= " << 1 << endl;
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
inline void SetCoveringInstance::createLP_Melihozen(string lpfilename) {
	ofstream out(lpfilename.c_str());   
	out << "minimize 0";
	out << endl;
	out << "subject to" << endl;

	out << "\\Set Covering Constraint" << endl;
	for (int c = 0; c < n_cons; ++c) {
		out << "x" << vars_cons[c][0];
		for (size_t i = 1; i < vars_cons[c].size(); ++i) {
			out << " + " << "x" << vars_cons[c][i];
		}
		out << " >= " << 1 << endl;
		out << endl;        
	}

	for (int o = 0; o < n_objs; ++o) {
		out << "\\New objective is defined" << endl;
		out << (-1)*objs[o][0] << " x" << 0;
		for (int i = 1; i < n_vars; ++i) {
			out << " + " << (-1)*objs[o][i] << " x" << i << " ";
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



inline void SetCoveringInstance::reset_order(vector<int> new_order){
	cout << "\tReordering instance based on input order...";
	for (int c = 0; c < n_cons; ++c) {
		for (size_t i = 0; i < vars_cons[c].size(); ++i) {
			vars_cons[c].push_back( new_order[vars_cons_canonical[c][i]] );
			cons_var[ new_order[vars_cons_canonical[c][i]] ].push_back(c);
		}
	}

	for (int p = 0; p < n_objs; ++p) {
		for (int j = 0; j < n_vars; ++j) {
			objs[p][new_order[j]] = objs_canonical[p][j];			
		}
	}

}

#endif /* SETCOVERING_INSTANCE_HPP_ */
