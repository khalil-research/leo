// ----------------------------------------------------------
// BDD Data Structure
// ----------------------------------------------------------

#ifndef BDD_HPP_
#define BDD_HPP_

#include <cassert>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#include "pareto_frontier.hpp"
#include "../util/util.hpp"

using namespace std;

//
// BDD Node
//
struct Node {
	// Node layer
	const int layer;
	// Node index
	int index;
	// Arcs
	Node* arcs[2];
    // Arc weights
    ObjType* weights[2];
	// Incoming arcs
	vector<Node*> prev[2];
    // Node status (for general operations)
    char status;
    // Pareto frontier from top at a node
    ParetoFrontier* pareto_frontier;
    // Pareto frontier from bottom at a node
    ParetoFrontier* pareto_frontier_bu;
	// State info for knapsack: minimum node weight
	int min_weight;
	// State info for set packing: the set of variables we can still choose
	boost::dynamic_bitset<> setpack_state;
    // State info for set covering: the set of constraints we have to cover
    boost::dynamic_bitset<> setcover_state;

    // S-approximation 
    ParetoFrontier* pareto_frontier_S;
    // T-approximation 
    ParetoFrontier* pareto_frontier_T;

	// Constructor
	Node(const int _layer, int _index);

	// Add outgoing arc
	void add_out_arc(Node* target, const int arc_type);

	// Add outgoing arc without updating incoming arcs in target node
	void add_out_arc_fast(Node* target, const int arc_type);

    // Set arc weights
    void set_arc_weights(const int type, ObjType* arc_weights) {
        weights[type] = arc_weights;
    }
};



//
// BDD
//
struct BDD {
	// Number of layers
	const int num_layers;
	// Set of layers
	vector< vector<Node*> > layers;

	// Constructor
	BDD(int _num_layers) : num_layers(_num_layers) {
		layers.resize(num_layers);
	}

	// Destructor
	~BDD();

	// Add node in layer
	Node* add_node(int layer);

	// Remove node (not from layer) 
	void remove_node(Node* node);

	// Merge nodeB into nodeA, adjusting incoming/outgoing arcs approprietly
	// (Obs.: Does not modify outgoing arcs from nodeB)
	void merge_nodes(Node* nodeA, Node* nodeB); 

	// Ensure node indices are consistent
	void fix_indices();

	// Get BDD width
	int get_width();

	// Get number of nodes
	int get_num_nodes();

	// Print BDD
	void print();

    // Remove dangling nodes (incoming arcs are not updated)
    void remove_dangling_nodes();

    // Update incoming arc sets of each node
    void update_incoming_arcsets();

	// Return terminal
	Node* get_terminal() const {
		return layers[num_layers-1][0];	
	}

	// Return root
	Node* get_root() const {
		return layers[0][0];	
	}
};



//
// Node constructor
//
inline Node::Node(const int _layer, int _index) 
	: layer(_layer), index(_index)
{ 
    for (int arc_type = 0; arc_type < 2; ++arc_type) {
        arcs[arc_type] = NULL;
        weights[arc_type] = NULL;
    }
}



//
// Add outgoing arc
//
inline void Node::add_out_arc(Node* tgt, const int arc_type) {
	assert(tgt != NULL);
	assert(tgt->layer == this->layer + 1);
	assert(arc_type == 0 || arc_type == 1);
	arcs[arc_type] = tgt;
	tgt->prev[arc_type].push_back(this);
}



//
// Add outgoing arc without updating incoming arcs in target node
//
inline void Node::add_out_arc_fast(Node* tgt, const int arc_type) {
	assert(tgt != NULL);
	assert(tgt->layer == this->layer + 1);
	assert(arc_type == 0 || arc_type == 1);
	arcs[arc_type] = tgt;
}



//
// Remove node (not from layer) 
//
inline void BDD::remove_node(Node* node) {
	for (int arc_type = 0; arc_type < 2; ++arc_type) {
		// Remove node from incoming arc references
		for (int i = 0; i < node->prev[arc_type].size(); ++i) {
			node->prev[arc_type][i]->arcs[arc_type] = NULL;
		}
		// Remove node from outgoing arc references
		if (node->arcs[arc_type] != NULL) {
			for (int i = 0; i < node->arcs[arc_type]->prev[arc_type].size(); ++i) {
				if (node->arcs[arc_type]->prev[arc_type][i] == node) {
					node->arcs[arc_type]->prev[arc_type][i] = node->arcs[arc_type]->prev[arc_type].back();
					node->arcs[arc_type]->prev[arc_type].pop_back();
					break;
				}
			}			
		}
	}
	delete node;
}



//
// BDD Destructor
//
inline BDD::~BDD() {
	for (int l = 0; l < num_layers; ++l) {
		for (int i = 0; i < layers[l].size(); ++i) {
			delete layers[l][i];
		}
	}
}



//
// Add node in layer
//
inline Node* BDD::add_node(int layer) {
	assert(layer >= 0 && (int)layer < layers.size());
	layers[layer].push_back( new Node(layer, layers[layer].size()) );
	return layers[layer].back();
}



//
// Get BDD width
//
inline int BDD::get_width() {
	size_t w = 0;
	for (int i = 0; i < layers.size(); ++i) {
		w = std::max(w, layers[i].size());
	}
	return w;
}



//
// Print BDD
//
inline void BDD::print() {
	cout << endl;
	cout << "** BDD **" << endl;
	for (int l = 0; l < num_layers; ++l) {
		cout << "\tLayer " << l << endl;
        
        // cout << "\t  Node weights: ";
        // for (vector<Node*>::iterator it = layers[l].begin(); it != layers[l].end(); ++it)
        //     cout << (*it)->min_weight << ",";
        // cout << endl;
        
        cout << "\t  Arcs:" <<endl;
        for (vector<Node*>::iterator it = layers[l].begin(); it != layers[l].end(); ++it) {
			Node* node = *it;
			cout << "\t\t";
			cout << node->index;
			if (node->arcs[0] != NULL) {
				cout << " --> 0-arc (";
				cout << node->arcs[0]->index;
                cout << ")";
				cout << " - arc weight: ";
               	for	(int i = 0; i < NOBJS; ++i) {
                	cout << node->weights[0][i] << " ";
				}
			}
			if (node->arcs[1] != NULL) {
				cout << " --> 1-arc (";
				cout << node->arcs[1]->index;
                cout << ")";
				for	(int i = 0; i < NOBJS; ++i) {
                	cout << node->weights[1][i] << " ";
				}
			}
			cout << endl;
		}
	}
	cout << "** Done **" << endl << endl;
}



//
// Ensure node indices are consistent
//
inline void BDD::fix_indices() {
	for (int l = 0; l < num_layers; ++l) {
		for (int i = 0; i < layers[l].size(); ++i) {
			layers[l][i]->index = i;
		}
	}
}



//
// Get number of nodes of the BDD
//
inline int BDD::get_num_nodes() {
	int num_nodes = 0;
	for (int l = 0; l < num_layers; ++l) {
		num_nodes += layers[l].size();
	}
	return num_nodes;
}



//
// Update incoming arc sets of each node
//
inline void BDD::update_incoming_arcsets() {
    for (int l = 0; l < num_layers-1; ++l) {
        // Reset incoming arc sets in next-layer nodes
        for (vector<Node*>::iterator it = layers[l+1].begin(); it != layers[l+1].end(); ++it) {
            for (int arc_type = 0; arc_type < 2; ++arc_type) {
                (*it)->prev[arc_type].clear();
            }
        }
        // Update incoming arcs
        for (vector<Node*>::iterator it = layers[l].begin(); it != layers[l].end(); ++it) {
            if ((*it)->arcs[0] != NULL) {
                (*it)->arcs[0]->prev[0].push_back( (*it) );
            }
            if ((*it)->arcs[1] != NULL) {
            	(*it)->arcs[1]->prev[1].push_back( (*it) );
            }            
        }        
    }
}



//
// Remove dangling nodes (incoming arcs are not updated)
//
inline void BDD::remove_dangling_nodes() {
	// Mark nodes to remove
	get_terminal()->status = 'T';
	for (int l = num_layers-2; l >= 0; --l) {
        for (vector<Node*>::iterator it = layers[l].begin(); it != layers[l].end(); ++it) {
            Node* node = (*it);
			if (node->arcs[0] != NULL && node->arcs[1] != NULL) {
				node->status = ( (node->arcs[0]->status == 'F') && (node->arcs[1]->status == 'F') ? 'F' : 'T' );
			} else if (node->arcs[0] != NULL) {
				node->status = ( (node->arcs[0]->status == 'F') ? 'F' : 'T' );
			} else if (node->arcs[1] != NULL) {
				node->status = ( (node->arcs[1]->status == 'F') ? 'F' : 'T' );
			} else {
				node->status = 'F';
			}            
        }
    }
	// Remove nodes
	for (int l = 1; l < num_layers-2; ++l) {
        int k = 0;
        int removed = 0;
        while (k < layers[l].size()) {
            if (layers[l][k]->status == 'F') {
                layers[l][k] = layers[l].back();
                layers[l].pop_back();
                removed++;
            } else {
                ++k;
            }
        }
    }
	// Fix indices
	fix_indices();
}



#endif
