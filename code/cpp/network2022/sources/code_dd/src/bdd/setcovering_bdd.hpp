// ----------------------------------------------------------
// Set Covering BDD Constructor
// ----------------------------------------------------------

#ifndef SETCOVERING_BDD_HPP_
#define SETCOVERING_BDD_HPP_

#define BOOST_DYNAMIC_BITSET_DONT_USE_FRIENDS

#include <boost/dynamic_bitset.hpp>
#include <boost/unordered_map.hpp>
#include <boost/foreach.hpp>

#include <cassert>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <fstream>

#include "../bdd/bdd.hpp"
#include "../util/util.hpp"

#include "../instances/setcovering_instance.hpp"

using namespace std;


//
// Indepset BDD constructor
//
class SetCoveringBDDConstructor {
public:
	// State definitions
	typedef boost::dynamic_bitset<> State;
	typedef boost::unordered_map<State*, Node*, bitset_hash, bitset_equal_to> StateNodeMap;

	// Constructor
	SetCoveringBDDConstructor(SetCoveringInstance* _inst, vector< vector<int> >& _objs);

	// Generate exact BDD
	BDD* generate_exact();

	// Destructor
	~SetCoveringBDDConstructor() { }

	// Variable in a layer
	vector<int> var_layer;

private:
	// Indepset instance
	SetCoveringInstance* inst;
	// Objectives
	vector< vector<int> > objs;
	// Number of objectives
	const int num_objs;
	// Marker of the end of a state (for iteration purposes)
	const int state_end;
	// State maps
	StateNodeMap states[2];
	// Auxiliary
	vector< pair<State, Node*> > aux_nodes;
	// Clauses that are eliminated after variable branching
	vector< vector<State> > clauses_var_cons;
	// If constraint needs to be checked for absorption
	bool** cons_needs_checking;
	// Masks to set state after variable was assigned to 0 or 1
	vector<State> mask_set_zero;
	vector<State> mask_set_one;
	// Constraints for which a variable is last
	vector<State> last_cons;

	// Preprocess data for BDD construction
	void preprocess();
};





#endif /* SETCOVERING_BDD_HPP_ */

// ----------------------------------------------------------



//
// Constructor
//
inline SetCoveringBDDConstructor::SetCoveringBDDConstructor(
	SetCoveringInstance* _inst, 
	vector< vector<int> >& _objs)
	: inst(_inst), objs(_objs), num_objs(_objs.size()),
	 state_end(static_cast<int>(boost::dynamic_bitset<>::npos))
{ }
