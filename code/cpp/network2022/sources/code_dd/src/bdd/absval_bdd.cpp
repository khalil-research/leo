// ----------------------------------------------------------
// AbsVal BDD Constructor - Implementations
// ----------------------------------------------------------

#include "absval_bdd.hpp"


//
// Generate exact BDD
// TODO: optimize vector allocation
//
BDD* AbsValBDDConstructor::generate_exact() {
    //cout << "Generating exact AbsVal BDD...\n";
    
    // AbsVal BDD
    BDD* bdd = new BDD(inst->n_xvars+1);
    
    // State maps
    int iter = 0;
    int next = 1;
    
    // State information at root node
    State state(NOBJS+1, 0);  // absolute value, plus cardinality
    for (int i = 0; i < NOBJS; ++i) {
        state[i] = (-1) * inst->b[i];
    }

    // create root node
    Node* root_node = bdd->add_node(0);
    states[iter].clear();
    states[iter][state] = root_node;
    
    // create terminal node
    Node* terminal_node = bdd->add_node(inst->n_xvars);
    
    // auxiliaries
    bool feasible = false;

    // Zero-arc weights
    ObjType* zero_weights = new ObjType[NOBJS];
    memset(zero_weights, 0, sizeof(ObjType)*NOBJS);

    int card = inst->n_xvars * inst->cardinality_ratio;

    for (int l = 0; l < inst->n_xvars; ++l) {
        // if (l == 1) {
        //     exit(1);
        // }
      //cout << "\tLayer " << l << " - size: " << states[iter].size() << endl; 

        // Reset next states
        states[next].clear();

        // Create states associated with next layer
        BOOST_FOREACH(StateNodeMap::value_type st, states[iter]) {
            // Obtain node and states
            Node* node = st.second;
            state = st.first;

            // cout << "\n\nState: " << endl;
            // for (int i = 0; i < NOBJS; ++i) {
            //     cout << "\t" << state[i] << endl;
            // }
            // cout << endl;

            if (l == 0) {
                // ---------------------------------------
                // Layer 0
                // ---------------------------------------            

                // zero arc: add constant cost of not taking any items
                ObjType* zero_weights = new ObjType[NOBJS];
                for (int i = 0; i < NOBJS; ++i) {
                    zero_weights[i] = (-1) * std::abs(state[i]);
                }

                // cout << "Zero arc: " << endl;
                // for (int i = 0; i < NOBJS; ++i) {
                //     cout << "\t" << zero_weights[i];
                // }
                // cout << endl;
                            
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

                // check feasibility with respect to cardinality
                ++state[NOBJS];
                feasible = (state[NOBJS] <= card);

                if (feasible) {
                    // Construct one-arc weights, also considering root node constant cost
                    ObjType* one_weights = new ObjType[NOBJS];
                    for (int i = 0; i < NOBJS; ++i) {
                        one_weights[i] = (-1) * ((std::abs(state[i] + inst->A[i][l]) - std::abs(state[i])) + std::abs(state[i]));
                    }

                    // cout << "One arc: " << endl;
                    // for (int i = 0; i < NOBJS; ++i) {
                    //     cout << "\t" << one_weights[i];
                    // }
                    // cout << endl;
    
                    // Update states
                    //cout << "Node: " << endl;
                    for (int i = 0; i < NOBJS; ++i) {
                        state[i] += inst->A[i][l];
                        //cout << "\t" << i << ": " << state[i] << endl;
                    }

                    // Add to node map
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
              
            } else if (l >= 1 && l < inst->n_xvars-1) {

                // ---------------------------------------
                // Layers 1, ..., n-1
                // ---------------------------------------            

                // zero arc: just replicate state
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

                // check feasibility with respect to cardinality
                ++state[NOBJS];
                feasible = (state[NOBJS] <= card);

                if (feasible) {
                    // Construct one-arc weights
                    ObjType* one_weights = new ObjType[NOBJS];
                    for (int i = 0; i < NOBJS; ++i) {
                        one_weights[i] = (-1) * (std::abs(state[i] + inst->A[i][l]) - std::abs(state[i]));
                    }
                    
                    // Update states
                    //cout << "Node: " << endl;
                    for (int i = 0; i < NOBJS; ++i) {
                        state[i] += inst->A[i][l];
                        //cout << "\t" << i << ": " << state[i] << endl;
                    }

                    // Add to node map
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
                // ---------------------------------------
                // Layer n
                // ---------------------------------------            

                // if last layer, just add arcs to the terminal node
                
                // zero arc
                node->add_out_arc_fast(terminal_node, 0);
                node->set_arc_weights(0, zero_weights);
                
                // cout << "Zero arc: " << endl;
                // for (int i = 0; i < NOBJS; ++i) {
                //     cout << "\t" << zero_weights[i];
                // }
                // cout << endl;

                // one arc
                feasible = (st.first[NOBJS] + 1 <= card);
                
                if (feasible) {
                    // Construct one-arc weights
                    //cout << "A: " << endl;
                    ObjType* one_weights = new ObjType[NOBJS];
                    for (int i = 0; i < NOBJS; ++i) {
                        one_weights[i] = (-1) * (std::abs(state[i] + inst->A[i][l]) - std::abs(state[i]));
                        //cout << "\t" << inst->A[i][l] << endl;
                    }
                    //cout << endl;

                    // cout << "One arc: " << endl;
                    // for (int i = 0; i < NOBJS; ++i) {
                    //     cout << "\t" << one_weights[i];
                    // }
                    // cout << endl;
    
                    node->add_out_arc_fast(terminal_node, 1);    
                    node->set_arc_weights(1, one_weights);
                }
            }
        }
 
        // invert iter and next
        next = !next;
        iter = !iter; 
    }
            
    // Update BDD
    bdd->update_incoming_arcsets();
    bdd->fix_indices();

    return bdd;
}
