// ----------------------------------------------------------
// Decision Diagram Data Structure
// ----------------------------------------------------------

#ifndef MDD_HPP_
#define MDD_HPP_

#define BOOST_DYNAMIC_BITSET_DONT_USE_FRIENDS

#include "../bdd/pareto_frontier.hpp"
#include <algorithm>
#include <boost/dynamic_bitset.hpp>
#include <cassert>
#include <vector>
#include <iostream>


using namespace std; 



//
// Forward declarations
//
struct MDDArc;
struct MDDNode;
struct MDD;


//
// DD MDDArc
//
struct MDDArc {
    // MDDArc value
    const int val;
    // MDDNode tail
    MDDNode* tail;
    // MDDNode head
    MDDNode* head;
    // MDDArc length
    int length;
    // MDDArc index
    int index;
    // Objective weights
    ObjType* weights;

    // Constructor
    MDDArc(const int _val, MDDNode* _tail, MDDNode* _head);

    // Constructor with lengths
    MDDArc(const int _val, MDDNode* _tail, MDDNode* _head, int _length);

    // Destructor
    ~MDDArc() {
        delete[] weights;
    }

    // Set arc length
    void set_length(const int _length);
};


//
// DD MDDNode
//
struct MDDNode {
    // MDDNode layer
    const int layer;
    // MDDNode index
    int index;
    // Max value of outgoing arcs
    const int max_value;
    // Incoming arcs
    vector<MDDArc*> in_arcs_list;
    // Outgoing arc list
    vector<MDDArc*> out_arcs_list;
    // Outgoing arc per value
    vector<MDDArc*> out_arc_val;
    // Set of cities visited
    boost::dynamic_bitset<> S;
    // Last city visited
    int last_city;
    // Length from root to node
    int length;
    // Pareto frontier
    ParetoFrontier* pareto_frontier;
    // Pareto frontier from the bottom
    ParetoFrontier* pareto_frontier_bu;

    // MDDNode constructor, where outgoing values range in [0, _maxvalue)
	MDDNode(const int _layer, const int _index, int _max_value);

    // Empty node constructor
	MDDNode() : layer(-1), max_value(-1) { }

    // MDDNode destructor
    ~MDDNode();

    // Add outgoing arc
    MDDArc* add_out_arc(MDDNode* head, int val, int length);

    // Remove incoming arc (do not check arc tail)
    void remove_incoming(MDDArc* a);

    // Remove outgoing arc (do not check arc head)
    void remove_outgoing(MDDArc* a);
};


//
// Decision diagram
//
struct MDD {
    // Number of layers
	const int num_layers;
    // List of nodes per layer
    vector< vector<MDDNode*> > layers;
    // MDD width
    int width;
    // Number of nodes
    int num_nodes;
    // Max arc value in any layer
    int max_value;
    // Longest path 
    double longest_path;

    // Constructor
	MDD(const int _num_layers);

    // Destructor
    ~MDD();

    // Add a node in layer, considering max value of outgoing arcs
    MDDNode* add_node(const int layer, const int max_value);

    // Add an arc in layer
    MDDArc* add_arc(MDDNode* tail, MDDNode* head, int val);

    // Add an arc in layer considering length
    MDDArc* add_arc(MDDNode* tail, MDDNode* head, int val, int length);

    // Remove arc
    void remove(MDDArc* a);

    // Get root node
    MDDNode* get_root() const;

    // Get terminal node
    MDDNode* get_terminal() const;

    // Redirect incoming arcs from nodeB to node A
    void redirect_incoming(MDDNode* nodeA, MDDNode* nodeB);

    // Redirect existing arc as incoming to a node
    void redirect_incoming(MDDNode* node, MDDArc* in_arc);

    // Repair node indices 
    void repair_node_indices();

    // Update MDD info
    void update();

    // Get MDD width (only valid when updated)
    int get_width() const;

    // Get max arc value in any layer (only valid when updated)
    int get_max_value() const;

    // Print MDD
    void print() const;

    // Check MDD consistency
    bool check_consistency();

    // Update number of nodes
    void update_num_nodes();

    // Remove nodes with zero outgoing arcs
    void remove_dangling_outgoing();

    // Remove nodes with zero incoming arcs
    void remove_dangling_incoming();

    // Remove dangling nodes
    void remove_dangling();

    // Remove node
    void remove(MDDNode* node);
};



// ----------------------------------------------------------
// Inline implementations
// ----------------------------------------------------------


//
// MDDArc constructor
//
inline MDDArc::MDDArc(const int _val, MDDNode* _tail, MDDNode* _head) 
    : val(_val), tail(_tail), head(_head), length(0)
{ 
    assert( tail != NULL );
    assert( head != NULL );
}


//
// MDDArc constructor with lengths
//
inline MDDArc::MDDArc(const int _val, MDDNode* _tail, MDDNode* _head, int _length) 
    : val(_val), tail(_tail), head(_head), length(_length)
{ 
    assert( tail != NULL );
    assert( head != NULL );
}


//
// Set arc length
//
inline void MDDArc::set_length(const int _length) {
    length = _length;
}


//
// MDDNode constructor, where outgoing values range in [0,...,maxvalue)
//
inline MDDNode::MDDNode(const int _layer, const int _index, const int _max_value) 
    : layer(_layer), index(_index), max_value(_max_value)
{ 
    // allocate list of arcs per value
    out_arc_val.resize( max_value, NULL );
}


//
// Add outgoing arc
//
inline MDDArc* MDDNode::add_out_arc(MDDNode* head, int val, int length) {
    assert( head != NULL );
    assert( layer == head->layer -1 );
    assert( val >= 0 && val < max_value );
    assert( out_arc_val[val] == NULL );

    MDDArc* a = new MDDArc(val, this, head, length);
    out_arc_val[val] = a;
    out_arcs_list.push_back( a );
    head->in_arcs_list.push_back( a );

    return a;
}


//
// MDDNode destructor
//
inline MDDNode::~MDDNode() {
    for (int j = 0; j < out_arcs_list.size(); ++j) {
        assert( out_arcs_list[j] != NULL );
        delete out_arcs_list[j];
    }
}


//
// MDD constructor
//
inline MDD::MDD(const int _num_layers) : num_layers(_num_layers) {
    layers.resize(num_layers);
}


//
// MDD destructor
//
inline MDD::~MDD() {
    for (int l = 0; l < num_layers; ++l) {
        for (int i = 0; i < layers[l].size(); ++i) {
            delete layers[l][i];
        }
    }
}


//
// Add a node in layer, considering max value of outgoing arcs
//
inline MDDNode* MDD::add_node(const int layer, const int max_value) {
	assert(layer >= 0 && (int)layer < layers.size());
    layers[layer].push_back( new MDDNode(layer, layers[layer].size(), max_value) );
    ++num_nodes;
	return layers[layer].back();
}


//
// Remove incoming arc (do not check arc tail)
//
inline void MDDNode::remove_incoming(MDDArc* a) {
    assert( a != NULL );
    assert( a->head == this );

    for (int j = 0; j < in_arcs_list.size(); ++j) {
        if (in_arcs_list[j] == a) {
            in_arcs_list[j] = in_arcs_list.back();
            in_arcs_list.pop_back();
            return;
        }
    }
}


//
// Remove outgoing arc (do not check arc head)
//
inline void MDDNode::remove_outgoing(MDDArc* a) {
    assert( a != NULL );
    assert( a->tail == this );
    
    out_arc_val[a->val] = NULL;
    for (int j = 0; j < out_arcs_list.size(); ++j) {
        if (out_arcs_list[j] == a) {
            out_arcs_list[j] = out_arcs_list.back();
            out_arcs_list.pop_back();
            return;
        }
    }
}


//
// Add an arc in layer
//
inline MDDArc* MDD::add_arc(MDDNode* tail, MDDNode* head, int val) {
    assert( tail != NULL );
    assert( head != NULL );
    assert( tail->layer == head->layer -1 );
    assert( val >= 0 && val < tail->max_value );
    assert( tail->out_arc_val[val] == NULL );

    MDDArc* a = new MDDArc(val, tail, head);
    tail->out_arc_val[val] = a;
    tail->out_arcs_list.push_back( a );
    head->in_arcs_list.push_back( a );

    return a;
}


//
// Add an arc in layer considering length
//
inline MDDArc* MDD::add_arc(MDDNode* tail, MDDNode* head, int val, int length) {
    assert( tail != NULL );
    assert( head != NULL );
    assert( tail->layer == head->layer -1 );
    assert( val >= 0 && val < tail->max_value );
    assert( tail->out_arc_val[val] == NULL );

    MDDArc* a = new MDDArc(val, tail, head, length);
    tail->out_arc_val[val] = a;
    tail->out_arcs_list.push_back( a );
    head->in_arcs_list.push_back( a );

    return a;
}


//
// Remove arc
//
inline void MDD::remove(MDDArc* a) {
    assert( a != NULL );
    a->tail->remove_outgoing(a);
    a->head->remove_incoming(a);
    delete a;
}


//
// Remove node
//
inline void MDD::remove(MDDNode* node) {
    // Remove outgoing arcs
    for (int a = 0; a < node->out_arcs_list.size(); ++a) {
        MDDArc* arc = node->out_arcs_list[a];
        arc->head->remove_incoming(arc);
        delete arc;
    }
    node->out_arcs_list.clear();    
    // Remove incoming arcs
    for (int a = 0; a < node->in_arcs_list.size(); ++a) {
        MDDArc* arc = node->in_arcs_list[a];
        arc->tail->remove_outgoing(arc);
        delete arc;
    }
    node->in_arcs_list.clear();    
    // Deallocate node
    delete node;
}


// 
// Update MDD information
//
inline void MDD::update() {
    // Update width
    width = 0;
	for (int l = 0; l < layers.size(); ++l) {
		width = std::max(width, (int)layers[l].size());
	}
    // Update maximum arc value in any layer
    max_value = -1;
	for (int l = 0; l < layers.size(); ++l) {
        for (int i = 0; i < layers[l].size(); ++i) {
		    max_value = std::max(max_value, layers[l][i]->max_value);
        }
    }
    // Update number of nodes
    update_num_nodes();
}


//
// Redirect incoming arcs from nodeB to node A
//
inline void MDD::redirect_incoming(MDDNode* nodeA, MDDNode* nodeB) {
    for (int j = 0; j < nodeB->in_arcs_list.size(); ++j) {
        MDDArc* a = nodeB->in_arcs_list[j];
        a->head = nodeA;
        nodeA->in_arcs_list.push_back( a );
    }
    nodeB->in_arcs_list.clear();
}


//
// Redirect existing arc as incoming to a node
//
inline void MDD::redirect_incoming(MDDNode* node, MDDArc* in_arc) {
    node->in_arcs_list.push_back(in_arc);
    in_arc->head = node;
}


//
// Get MDD width
//
inline int MDD::get_width() const {
    return width;
}


//
// Get max arc value in any layer 
//
inline int MDD::get_max_value() const {
    return max_value;
}


//
// Print MDD
//
inline void MDD::print() const {
    cout << endl;
	cout << "** MDD **" << endl;
	for (int l = 0; l < num_layers; ++l) {
		cout << "\tLayer " << l << endl;
		for (vector<MDDNode*>::const_iterator it = layers[l].begin(); it != layers[l].end(); ++it) {
			MDDNode* node = *it;			
            cout << "\t\tNode " << node->layer << "," << node->index;
            cout << " - S=" << node->S;
            cout << " - last_city=" << node->last_city;
            cout << endl;
            for (int j = 0; j < node->out_arcs_list.size(); ++j) {
                MDDArc* a = node->out_arcs_list[j];
                cout << "\t\t\tArc val=" << a->val;
                cout << " - head=" << a->head->index;
                cout << " - tail=" << a->tail->index;
                cout << " - weights={ ";
                for (int o = 0; o < NOBJS; ++o) {
                    cout << a->weights[o] << " ";
                }
                cout << "}";
                cout << endl;
            }
		}
	}
	cout << "** Done **" << endl << endl;
}


//
// Get root node
//
inline MDDNode* MDD::get_root() const {
    assert( num_layers > 0 );
    assert( layers[0].size() > 0 );

    return layers[0][0];
}


//
// Get terminal node
//
inline MDDNode* MDD::get_terminal() const {
    assert( num_layers > 0 );
    assert( layers[num_layers-1].size() > 0 );

    return layers[num_layers-1][0];
}


//
// Remove nodes with zero outgoing arcs
//
inline void MDD::remove_dangling_outgoing() {
    for (int l = num_layers-2; l >= 0; --l) {
        int i = 0;
        while (i < layers[l].size()) {
            MDDNode* node = layers[l][i];
            if (node->out_arcs_list.empty()) {
                // Remove node from layer
                layers[l][i] = layers[l].back();
                layers[l].pop_back();
                // Remove node
                remove(node);
            } else {
                ++i;
            }
        }
    }
}


//
// Remove nodes with zero incoming arcs
//
inline void MDD::remove_dangling_incoming() {
    for (int l = 1; l < num_layers; ++l) {
        int i = 0;
        while (i < layers[l].size()) {
            MDDNode* node = layers[l][i];
            if (node->in_arcs_list.empty()) {
                // Remove node from layer
                layers[l][i] = layers[l].back();
                layers[l].pop_back();
                // Remove node
                remove(node);
            } else {
                ++i;
            }
        }
    }
}


//
// Remove dangling nodes
//
inline void MDD::remove_dangling() {
    remove_dangling_incoming();
    remove_dangling_outgoing();
}


//
// Check MDD consistency
//
inline bool MDD::check_consistency() {
    //cout << endl << endl;
    for (int l = 0; l < num_layers; ++l) {
        //cout << "Layer " << l << endl;
        if (layers[l].size() == 0) {
            //cout << "Error: layer " << l << " is empty." << endl;
            return false;
        }
        for (int i = 0; i < layers[l].size(); ++i) {
            MDDNode* node = layers[l][i];
            //cout << "\tMDDNode " << node->index << endl;

            if (l < num_layers-1) {
                if (node->out_arcs_list.size() == 0) {
                    cout << "Error: node " << node->layer << "," << node->index;
                    cout << " has zero outgoing arcs." << endl;
                    return false;
                }
                for (int j = 0; j < node->out_arcs_list.size(); ++j) {
                    MDDArc* a = node->out_arcs_list[j];
                    if (a != node->out_arc_val[a->val]) {
                        cout << "Error: ";
                        cout << "node " << node->layer << "," << node->index;
                        cout << " - arc: " << a->head->index << "," << a->val;
                        cout << " does not correspond to correct out_arcs_val" << endl;
                        return false;
                    }
                    const MDDNode* head = a->head;
                    bool found = false;
                    for (int k = 0; k < head->in_arcs_list.size() && !found; ++k) {
                        found = (head->in_arcs_list[k] == a);
                    }
                    if (!found) {
                        cout << "Error: ";
                        cout << "node " << node->layer << "," << node->index;
                        cout << " - arc: " << a->head->index << "," << a->val;
                        cout << " was not found in head incoming list" << endl;
                        return false;
                    }                    
                }
                for (int v = 0; v < node->max_value; ++v) {
                    if (node->out_arc_val[v] == NULL) {
                        continue;
                    }
                    MDDArc* a = node->out_arc_val[v];
                    bool found = false;
                    for (int j = 0; j < node->out_arcs_list.size() && !found; ++j) {
                        found = (a == node->out_arcs_list[j]);
                    }
                    if (!found) {
                        cout << "Error: ";
                        cout << "node " << node->layer << "," << node->index;
                        cout << " - arc: " << a->head->index << "," << a->val;
                        cout << " was not found in outgoing list" << endl;
                        return false;
                    }                    
                }
            }
            if (l > 1) {
                if (node->in_arcs_list.size() == 0) {
                    cout << "Error: node " << node->layer << "," << node->index;
                    cout << " has zero incoming arcs." << endl;
                    return false;
                }
                for (int j = 0; j < node->in_arcs_list.size(); ++j) {
                    MDDArc* a = node->in_arcs_list[j];
                    bool found = false;
                    for (int k = 0; k < a->tail->out_arcs_list.size() && !found; ++k) {
                        found = (a->tail->out_arcs_list[k] == a);
                    }
                    if (!found) {
                        cout << "Error: ";
                        cout << "node " << node->layer << "," << node->index;
                        cout << " - arc tail: " << a->tail->index << "," << a->val;
                        cout << " was not found in outgoing list of tail" << endl;
                        return false;
                    }                                        
                }                
            }
        }
    }
    return true;
}


//
// Repair MDD node indices 
//
inline void MDD::repair_node_indices() {
    for (int l = 0; l < num_layers; ++l) {
		for (int i = 0; i < layers[l].size(); ++i) {
			layers[l][i]->index = i;
		}
	}
}


// Update number of nodes
inline void MDD::update_num_nodes() {
    num_nodes = 0;
    for (int l = 0; l < num_layers; ++l) {
        num_nodes += layers[l].size();
    }
}
    

#endif
