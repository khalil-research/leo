// ----------------------------------------------------------
// Indepset BDD Constructor
// ----------------------------------------------------------

#ifndef INDEPSET_BDD_HPP_
#define INDEPSET_BDD_HPP_

#define BOOST_DYNAMIC_BITSET_DONT_USE_FRIENDS

#include <boost/unordered_map.hpp>
#include <boost/foreach.hpp>

#include <cassert>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <fstream>
#include <queue>

#include "../bdd/bdd.hpp"
#include "../util/util.hpp"

#include "../instances/indepset_instance.hpp"

using namespace std;



//
// Indepset BDD constructor
//
class IndepSetBDDConstructor {
public:

	// State definition
	typedef boost::dynamic_bitset<> State;
	typedef boost::unordered_map<State*, Node*, bitset_hash, bitset_equal_to> StateNodeMap;

	// Constructor
	IndepSetBDDConstructor(IndepSetInst* _inst, vector< vector<int> >& _objs);

	// Generate exact BDD
	BDD* generate_exact();

	// Destructor
	~IndepSetBDDConstructor() {
		delete[] in_state_counter;
	}

	// Variable in a layer
	vector<int> var_layer;

private:
	// Indepset instance
	IndepSetInst* inst;
	// Objectives
	vector< vector<int> > objs;
	// Number of objectives
	const int num_objs;
	// Marker of the end of a state (for iteration purposes)
	const int state_end;
	// State maps
	StateNodeMap states[2];
	// Active vertices (for variable ordering)
	vector<int> active_vertices;
	// Auxiliary
	vector< pair<State, Node*> > aux_nodes;
	// Used for min-in-state variable ordering
	int* in_state_counter;  		

	// Choose next vertex in min-in-state strategy
	int choose_next_vertex_min_size_next_layer(StateNodeMap& states);

	void compute_states(BDD* bdd);
};


// ------------------------------------------------------------------------------------------------
// Inline implementations
// ------------------------------------------------------------------------------------------------


//
// IndepsetBDD Constructor
//
inline IndepSetBDDConstructor::IndepSetBDDConstructor(IndepSetInst* _inst, 
													  vector< vector<int> >& _objs)
	: inst(_inst), 
	  objs(_objs), 
	  num_objs(_objs.size()),
	  state_end(static_cast<int>(boost::dynamic_bitset<>::npos))
{
	in_state_counter = new int[inst->graph->n_vertices];
}


//
// Choose next variable for the BDD 
//
inline int IndepSetBDDConstructor::choose_next_vertex_min_size_next_layer(StateNodeMap& states) { 

	// update counter
	for (size_t i = 0; i < active_vertices.size(); ++i) {
		in_state_counter[active_vertices[i]] = 0;
	}
	BOOST_FOREACH(StateNodeMap::value_type i, states) {
		const State& state = *(i.first);
		for (int v = state.find_first(); v != state_end; v = state.find_next(v)) {
			in_state_counter[v]++;
		}
	}

	int sel = 0;
	for (size_t i = 1; i < active_vertices.size(); ++i) {
		if (in_state_counter[active_vertices[i]] < in_state_counter[active_vertices[sel]]) {
			sel = i;
		}
	}
	// remove vertex from active list and return it
	int v = active_vertices[sel];
	active_vertices[sel] = active_vertices.back();
	active_vertices.pop_back();

	return v;
}


#endif /* INDEPSET_BDD_HPP_ */

// ----------------------------------------------------------
