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

#include "../bdd.hpp"
#include "../pareto_util.hpp"
#include "../util.hpp"

#include "indepset_instance.hpp"

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
	IndepSetBDDConstructor(IndepSetInst* _inst, vector< vector<int> >& _objs, int _maxwidth);

	// Generate exact BDD
	BDD* generate_exact(bool new_order_provided);

	// Generate exact BDD for tree graphs
	BDD* generate_exact_tree();

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
	// Maximum width
	const int maxwidth;
	// Number of objectives
	const int num_objs;
	// Marker of the end of a state (for iteration purposes)
	const int state_end;
	// State maps
	StateNodeMap states[2];
	// Auxiliary
	vector< pair<State, Node*> > aux_nodes;
	// Used for min-in-state variable ordering
	int* in_state_counter;  		
	// Active vertices (i.e., needs to be branched on)
	vector<int> active_vertices;

	// Choose next vertex in min-in-state strategy
	int choose_next_vertex_min_size_next_layer(StateNodeMap& states);
};


//
// IndepsetBDD Constructor
//
inline IndepSetBDDConstructor::IndepSetBDDConstructor(IndepSetInst* _inst, 
													  vector< vector<int> >& _objs, 
													  int _maxwidth)
	: inst(_inst), 
	  objs(_objs), 
	  maxwidth(_maxwidth), 
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


//
// Class for Ordering based on cut vertices for trees
//
class CutVertexDecomposition {
public:
	CutVertexDecomposition(IndepSetInst *_inst) : inst(_inst) { 
		v_in_layer.resize(inst->graph->n_vertices);
		component.resize(inst->graph->n_vertices);
		component_map.resize(inst->graph->n_vertices);
		construct_ordering();
	}

	int vertex_in_layer(BDD* bdd, int layer) {
		return v_in_layer[layer];
	}

private:
	// Independent set instance
	IndepSetInst *inst;      
	// Vertex at each layer
	vector<int> v_in_layer;   
	// Component of a vertex
	vector<int> component;
	// Component <--> index map
	vector<int> component_map;

	// Construct ordering
	void construct_ordering();

	// Create components 
	void identify_components(vector< vector<int> > &comps, vector<bool> &is_in_graph);

	// Find orderings relative to a particular subgraph 
	vector<int> find_ordering(vector<bool> is_in_graph);

	// Graph component find
	int find(int i) {
		if (component[i] == i) return i;
		return find(component[i]);
	}

	// Graph component union
	void union_f(int i, int j) {
		component[find(i)] = find(j);
	}
};




#endif /* INDEPSET_BDD_HPP_ */

// ----------------------------------------------------------
