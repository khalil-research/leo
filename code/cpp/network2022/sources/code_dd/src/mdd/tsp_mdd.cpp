// ----------------------------------------------------------
// MDD Constructor for TSP - Implementation
// ----------------------------------------------------------

#include "tsp_mdd.hpp"

#define BOOST_DYNAMIC_BITSET_DONT_USE_FRIENDS

#include <boost/dynamic_bitset.hpp>
#include <boost/unordered_map.hpp>
#include <boost/foreach.hpp>


//
// Constructor
//
MDDTSPConstructor::MDDTSPConstructor(TSPInstance* _inst) 
    : inst(_inst) 
{ }



//
// Generate exact MDD
// 
MDD* MDDTSPConstructor::generate_exact() {

    // Equality functions for Nodes
    struct bitset_equal_to : std::binary_function<MDDNode*, MDDNode*, bool> {
        bool operator()(const MDDNode* const x, 
                        const MDDNode* const y) const 
        {
            return (x->S == y->S) && (x->last_city == y->last_city);
        }
    };

    // Hash functions of dynamic_bitset pointer
    struct bitset_hash : std::unary_function<MDDNode*, std::size_t> {
        std::size_t operator()(const MDDNode* const node) const {            
            std::size_t hash_val = 0;
            boost::hash_combine(hash_val, node->S.m_bits);
            boost::hash_combine(hash_val, node->last_city);

            return hash_val;
        }
    };

    // State map
    typedef boost::unordered_map<MDDNode*, MDDNode*, bitset_hash, bitset_equal_to> StateNodeMap;

    // Initialize MDD
    MDD* mdd = new MDD(inst->n_cities + 2);

    // array of cities to visit
    vector<int> exact_vals;
    for (int i = 0; i < inst->n_cities; ++i) {
        exact_vals.push_back( i );
    }

    // Create root node
    MDDNode* root_node = mdd->add_node(0, inst->n_cities);
    root_node->S.resize( exact_vals.size() );
    root_node->S.reset();
    root_node->last_city = -1;

    // Values that have to be fixed.
    // This is used for branching, but in this case only the first city (0) is fixed.
    vector<int> fixed_vals;
    fixed_vals.push_back(0);

    bool* is_fixed = new bool[inst->n_cities]; 
    memset(is_fixed, false, sizeof(bool)*inst->n_cities);
    for (int c = 0; c < fixed_vals.size(); ++c) {
        is_fixed[ fixed_vals[c] ] = true; 
    }

    // Values that have to be exact
    bool* is_city_exact = new bool[inst->n_cities]; 
    memset(is_city_exact, false, sizeof(bool)*inst->n_cities);
    for (int c = 0; c < exact_vals.size(); ++c) {
        is_city_exact[ exact_vals[c] ] = true; 
    }
     
    // Map between exact values and their positions in the bitset
    vector<int> map(inst->n_cities);
    std::fill(map.begin(), map.end(), -1);
    int pos = 0;
    for (int c = 0; c < exact_vals.size(); ++c) {
        map[ exact_vals[c] ] = pos++;         
    }        

    // Initialize node map
    StateNodeMap states[2];
    StateNodeMap::iterator it;
    int iter = 0;
    int next = 1;

    // Add root node to state set
    states[iter][root_node] = root_node;

    // Initialize node buffer for general operations
    MDDNode* node_buffer = new MDDNode;

    // Create layers associated with fixed variables
    // All fixed arcs have a zero length
    for (int l = 0; l < fixed_vals.size(); ++l) {
        assert( mdd->layers[l].size() == 1 );

        // Reset next state
        states[next].clear();
                
        // Obtain node
        MDDNode* node = mdd->layers[l].back();

        // Create next layer node associated with fixed variable
        MDDNode* new_node = mdd->add_node(l+1, inst->n_cities);
        new_node->S = node->S;
        new_node->S[fixed_vals[l]] = true;
        new_node->last_city = fixed_vals[l];

        states[next][new_node] = new_node;
        MDDArc* arc = node->add_out_arc( new_node, fixed_vals[l], 0 );
        
        arc->weights = new ObjType[inst->n_objs];
        memset(arc->weights, 0, sizeof(ObjType) * inst->n_objs);

        // invert iter and next
        next = !next;
        iter = !iter;
    }

    // Create layers associated with non-fixed variables
    for (int l = fixed_vals.size(); l < inst->n_cities; ++l) {
        // cout << "Layer " << l << endl;

        // Reset next state
        states[next].clear();

        // Extend each state
        BOOST_FOREACH(StateNodeMap::value_type i, states[iter]) {
            MDDNode* node_state = i.first;
            MDDNode* node = i.second;

            for (int v = 0; v < inst->n_cities; ++v) {
                // Check if it is possible to extend state
                if (!node_state->S[v]) {
                    // Visited city state
                    node_buffer->S = node_state->S;
                    node_buffer->S[v] = true;
                    // Last city
                    node_buffer->last_city = v;

                    // Check if state exists in the next layer
                    it = states[next].find(node_buffer);
                    if (it == states[next].end()) {
                        MDDNode* new_node = mdd->add_node(l+1, inst->n_cities);
                        new_node->S = node_buffer->S;
                        new_node->last_city = node_buffer->last_city;

                        states[next][new_node] = new_node;
    
                        MDDArc* arc = node->add_out_arc( new_node, v, 0 );
                        arc->weights = new ObjType[inst->n_objs];
                        for (int o = 0; o < inst->n_objs; ++o) {
                            arc->weights[o] = (-1) * inst->objs[o][node_state->last_city][v];
                        }

                    } else {
                        MDDNode* head = it->second;
                        MDDArc* arc = node->add_out_arc(head, v, 0);
                        arc->weights = new ObjType[inst->n_objs];
                        for (int o = 0; o < inst->n_objs; ++o) {
                            arc->weights[o] = (-1) * inst->objs[o][node_state->last_city][v];
                        }
                    }
                }
            }
            assert( node->out_arcs_list.size() > 0 );            
        }        

        // invert iter and next
        next = !next;
        iter = !iter;        
    }

    // Create terminal node in last layer
    MDDNode* terminal = mdd->add_node(inst->n_cities+1, inst->n_cities);
    terminal->last_city = -1;
    for (MDDNode* node : mdd->layers[inst->n_cities]) {
        MDDArc* arc = node->add_out_arc(terminal, 0, 0);
        arc->weights = new ObjType[inst->n_objs];
        for (int o = 0; o < inst->n_objs; ++o) {
            arc->weights[o] = (-1) * inst->objs[o][node->last_city][0];
        }
    }

    // Update width
    mdd->update();
    assert( mdd->check_consistency() );

    // mdd->print();
    return mdd;
}