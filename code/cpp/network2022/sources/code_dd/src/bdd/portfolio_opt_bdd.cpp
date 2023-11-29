// ----------------------------------------------------------
// Portfolio BDD Constructor - Implementations
// ----------------------------------------------------------

#include "portfolio_opt_bdd.hpp"

//
// Generate exact BDD
//
BDD* PortfolioOptBDDConstructor::generate_exact() {
    cout << "Generating exact Portfolio Optimization BDD...\n";
    
    // Knapsack BDD
    BDD* bdd = new BDD(inst->n_vars+1);
    
    // State maps
    int iter = 0;
    int next = 1;
    
    // State information
    State state(NOBJS, 0);  // sigma_sq, gamma_cube, beta_fourth and knapsack capacity
    
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
        cout << "\tLayer " << l << endl; // << " - number of nodes: " << states[next].size() << endl;
        
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
                }
                else {
                    node->add_out_arc_fast(it->second, 0);
                    node->set_arc_weights(0, zero_weights);
                }
                
                // one arc
                feasible = (state[NOBJS-1] + inst->a[l] <= inst->b);
                
                if (feasible) {
                    
                    // Construct one-arc weights
                    ObjType* one_weights = new ObjType[NOBJS];
                    one_weights[0] = inst->mu[l];
                    if(NOBJS >= 2)
                        one_weights[1] = - (sqrt(state[0]+inst->sigma_sq[l]) - sqrt(state[0]));
                    if(NOBJS >= 3)
                        one_weights[2] = cbrt(state[1]+inst->gamma_cube[l]) - cbrt(state[1]);
                    if(NOBJS >= 4)
                        one_weights[3] = - (pow(state[2]+inst->beta_fourth[l],0.25) - pow(state[2],0.25));
                    
                    // update state
                    state[NOBJS-1] += inst->a[l];
                    if(NOBJS >= 2)
                        state[0] += inst->sigma_sq[l];
                    if(NOBJS >= 3)
                        state[1] += inst->gamma_cube[l];
                    if(NOBJS >= 4)
                        state[2] += inst->beta_fourth[l];
                    
                    StateNodeMap::iterator it = states[next].find(state);
                    if (it == states[next].end()) {
                        Node* new_node = bdd->add_node(l+1);
                        states[next][state] = new_node;
                        node->add_out_arc_fast(new_node, 1);
                        node->set_arc_weights(1, one_weights);
                    }
                    else {
                        node->add_out_arc_fast(it->second, 1);
                        node->set_arc_weights(1, one_weights);
                    }
                }
            }
            else {
                // if last layer, just add arcs to the terminal node
                
                // zero arc
                node->add_out_arc_fast(terminal_node, 0);
                node->set_arc_weights(0, zero_weights);
                
                // one arc
                feasible = (i.first[NOBJS-1] + inst->a[l] <= inst->b);
                
                if (feasible) {
                    node->add_out_arc_fast(terminal_node, 1);
                    
                    // Construct one-arc weights
                    ObjType* one_weights = new ObjType[NOBJS];
                    one_weights[0] = inst->mu[l];
                    if(NOBJS >= 2)
                        one_weights[1] = - (sqrt(state[0]+inst->sigma_sq[l]) - sqrt(state[0]));
                    if(NOBJS >= 3)
                        one_weights[2] = cbrt(state[1]+inst->gamma_cube[l]) - cbrt(state[1]);
                    if(NOBJS >= 4)
                        one_weights[3] = - (pow(state[2]+inst->beta_fourth[l],0.25) - pow(state[2],0.25));
                    
                    node->set_arc_weights(1, one_weights);
                }
            }
        }
        
        // invert iter and next
        next = !next;
        iter = !iter;
    }
    
    cout << "\n\tupdating incoming arcs..." << endl;
    bdd->update_incoming_arcsets();
    
    // Fix indices
    bdd->fix_indices();
    
    cout << "\tdone" << endl;
//    bdd->print();
    
    return bdd;
}
