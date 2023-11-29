// ----------------------------------------------------------
// Assignment Instance
// ----------------------------------------------------------

#ifndef ASSIGNMENT_INSTANCE_HPP_
#define ASSIGNMENT_INSTANCE_HPP_

#include <vector>

using namespace std;


//
// Assignment Instance
//
struct AssignmentInstance {
	// Number of variables
	int n_vars;   
	// Number of objective functions
	int n_objs;
	// Objective functions  (indexed by variable/variable/objective)
	vector< vector< vector<int> > > objs;

	// Empty Constructor
	AssignmentInstance() { }

	// Read instance based on kirlik model 
	void read_kirlik(char* filename);

	// Read instance based on Ozlen & Azizoglu
	void read_ozlen_azizoglu(char* filename);

	// Read instance based on Ozlen, Burton, & MacRae
	void read_ozlen_et_al(char* filename);

	// Read instance based on our model
	void read(char* filename);
};



#endif