// ----------------------------------------------------------
// BDD General Algorithms
// ----------------------------------------------------------

#ifndef BDD_ALG_HPP_
#define BDD_ALG_HPP_

#include <boost/unordered_map.hpp>
#include <boost/foreach.hpp>

#include "bdd.hpp"


//
// BDD Algorithms
//
struct BDDAlg {
	// Reduce BDD
	static void reduce(BDD* bdd);
};



//
// Reduce BDD
//
inline void BDDAlg::reduce(BDD* bdd) {
	// State for equivalence test
	typedef pair<int, int> state;

	// State map
	typedef boost::unordered_map<state, Node*> node_map;
	node_map states;
	state new_state;

	// Merge equivalent nodes
	for (int l = bdd->num_layers-2; l >= 0; --l) {
		states.clear();

		while (!bdd->layers[l].empty()) {
			// remove node from layer and add to map
			Node* node = bdd->layers[l].back();
			bdd->layers[l].pop_back();

            if (node->arcs[0] != NULL || node->arcs[1] != NULL) {
				// create new state
				new_state.first = (node->arcs[0] != NULL ? node->arcs[0]->index : -1);
				new_state.second = (node->arcs[1] != NULL ? node->arcs[1]->index : -1);

				// check if states exists
				node_map::iterator it = states.find(new_state);
				if (it == states.end()) {
					states[new_state] = node;
				} else {
					// state exists: merge node with existing one
					Node* original = it->second;
					assert(node->arcs[0] == original->arcs[0]);
					assert(node->arcs[1] == original->arcs[1]);

                    for (int arc_type = 0; arc_type < 2; ++arc_type) {
                        for (vector<Node*>::iterator it = node->prev[arc_type].begin();
                                it != node->prev[arc_type].end(); ++it) {
                            (*it)->arcs[arc_type] = original;
                        }                        
                    }
					delete node;
				}
			} else {
				// node can be removed
				bdd->remove_node(node);
			}
		}

		// add nodes back to layer, fixing indices
		BOOST_FOREACH(node_map::value_type it, states) {
			bdd->layers[l].push_back(it.second);
			bdd->layers[l].back()->index = bdd->layers[l].size()-1;
		}
	}

	// fix indices and incoming arcs
	for (int l = bdd->num_layers-1; l >= 0; --l) {
		for (int i = 0; i < bdd->layers[l].size(); ++i) {
			Node* node = bdd->layers[l][i];
			node->index = i;

			// clear incoming
            for (int arc_type = 0; arc_type < 2; ++arc_type) {
                node->prev[arc_type].clear();
                if (node->arcs[arc_type] != NULL) {
                    node->arcs[arc_type]->prev[arc_type].push_back(node);
                }                                
            }
		}
	}
}


#endif 


