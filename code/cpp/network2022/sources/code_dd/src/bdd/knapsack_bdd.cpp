// ----------------------------------------------------------
// Knapsack BDD Constructor - Implementations
// ----------------------------------------------------------

#include "knapsack_bdd.hpp"

//
// Generate exact BDD
//
BDD* KnapsackBDDConstructor::generate_exact() {
 	//cout << "Generating exact Knapsack BDD...\n";

	// Knapsack BDD
	BDD* bdd = new BDD(inst->n_vars+1);

	// State maps
	int iter = 0;
	int next = 1;

    // State information
    State state(inst->n_cons, 0);

	// create root node
	Node* root_node = bdd->add_node(0);
	states[iter].clear();
	states[iter][state] = root_node;

	// create terminal node
	Node* terminal_node = bdd->add_node(inst->n_vars);

    // auxiliaries
    bool feasible = false;

	// Zero-arc weights
	ObjType* zero_weights = new ObjType[NOBJS];
	memset(zero_weights, 0, sizeof(ObjType)*NOBJS);

	for (int l = 0; l < inst->n_vars; ++l) {   
		//cout << "\tLayer " << l << " - number of nodes: " << states[next].size() << endl;
		
		// Initialize one-arc weights
		ObjType* one_weights = new ObjType[NOBJS];
		for (int p = 0; p < NOBJS; ++p) {
			one_weights[p] = inst->obj_coeffs[l][p];
		}

	    states[next].clear();
	 	BOOST_FOREACH(StateNodeMap::value_type i, states[iter]) {
            // Obtain node and states	 		
	 		Node* node = i.second;
            state = i.first;

	 		if (l < inst->n_vars-1) {
				// zero arc
				StateNodeMap::iterator it = states[next].find(state);
				if (it == states[next].end()) {
					Node* new_node = bdd->add_node(l+1);
					states[next][state] = new_node;
					node->add_out_arc_fast(new_node, 0);
					node->set_arc_weights(0, zero_weights);
				} else {
					node->add_out_arc_fast(it->second, 0);
					node->set_arc_weights(0, zero_weights);
				}

				// one arc

                // update state
                feasible = true;
                for (int c = 0; c < inst->n_cons && feasible; ++c) {
                    state[c] += inst->coeffs[c][l];
                    feasible = (state[c] <= inst->rhs[c]);
                }				
				if (feasible) {
					StateNodeMap::iterator it = states[next].find(state);
					if (it == states[next].end()) {
						Node* new_node = bdd->add_node(l+1);
						states[next][state] = new_node;
						node->add_out_arc_fast(new_node, 1);
						node->set_arc_weights(1, one_weights);
					} else {
						node->add_out_arc_fast(it->second, 1);
						node->set_arc_weights(1, one_weights);
					}
				}

	 		} else {
	 			// if last layer, just add arcs to the terminal node

	 			// zero arc
	 			node->add_out_arc_fast(terminal_node, 0);
				node->set_arc_weights(0, zero_weights);

	 			// one arc
                feasible = true;
                for (int c = 0; c < inst->n_cons && feasible; ++c) {
                    feasible = (i.first[c] + inst->coeffs[c][l] <= inst->rhs[c]);
                }				
	 			if (feasible) {
	 				node->add_out_arc_fast(terminal_node, 1);
					node->set_arc_weights(1, one_weights);
	 			}
	 		}
	 	}

	 	// invert iter and next
	 	next = !next;
	 	iter = !iter;
	}
	//cout << "\n\tupdating incoming arcs..." << endl;
	bdd->update_incoming_arcsets();
	
	// Fix indices
	bdd->fix_indices();
		
    //cout << "\tdone" << endl;
    return bdd;
}


//
// Update node weights
//
void KnapsackBDDConstructor::update_node_weights(BDD* bdd) {
    // Run top-down to compute the minimum node weights (i.e., the used capacity)
	//cout << endl << "Updating node weights..." << endl;
    
    // root node
    bdd->get_root()->min_weight = 0;
    
    for (int l = 1; l < bdd->num_layers; ++l) {
        // iterate on layers
        for (vector<Node*>::iterator it = bdd->layers[l].begin(); it != bdd->layers[l].end(); ++it) {
            (*it)->min_weight = INT_MAX;
            
            // iterate over the incoming zero arcs
            for(vector<Node*>::iterator it_prev = (*it)->prev[0].begin(); it_prev != (*it)->prev[0].end(); ++it_prev) {
                (*it)->min_weight = min( (*it)->min_weight , (*it_prev)->min_weight );
            }
            
            // iterate over the incoming one arcs
            for(vector<Node*>::iterator it_prev = (*it)->prev[1].begin(); it_prev != (*it)->prev[1].end(); ++it_prev) {
                (*it)->min_weight = min( (*it)->min_weight , (*it_prev)->min_weight + inst->coeffs[0][l-1] );
            }
        }
    }
    
}	
