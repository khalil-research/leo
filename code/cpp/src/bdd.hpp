// ----------------------------------------------------------
// BDD Data Structure
// ----------------------------------------------------------

#ifndef BDD_HPP_
#define BDD_HPP_

#include <cassert>
#include <boost/dynamic_bitset.hpp>
#include <vector>
#include <iostream>
#include <limits>
#include <bits/stdc++.h>
#include "pareto_util.hpp"

#define INF_D std::numeric_limits<int>::max()

using namespace std;

//
// BDD Node
//
struct Node
{

	// Add incoming arc
	void add_in_arc(Node *source, bool one_arc, int length);

	// Add outgoing arc
	void add_out_arc(Node *target, bool one_arc, int length);

	// Constructor
	Node(int _layer, int _index);

	// Destructor
	~Node();

	// Layer of node
	int layer;
	// Node index in layer
	int index;
	// One arc
	Node *one_arc;
	// One arc cost
	int one_arc_length;
	// Zero arc
	Node *zero_arc;
	// Zero arc length
	int zero_arc_length;
	// One incoming ars
	vector<Node *> one_prev;
	// Zero incoming arc
	vector<Node *> zero_prev;
	// Int state of a node
	int intState;
	// Set state of a node
	boost::dynamic_bitset<> setState;
	// Length to this arc (please compute before using)
	int length;
	// If node is inactive (i.e., no arcs but should not be removed)
	bool active;
	// If node is relaxed (i.e., was forcefully merged during creation)
	bool exact;
	// Pareto set of node
	ParetoSet *pareto_set;
	// Solution up to this node
	vector<int> sol;

	// Multi-length: used for multi-objective
	vector<int> multi_length;

	// Longest path w.r.t. each objective function
	vector<int> ub_longest_path;
};

//
// BDD
//
struct BDD
{
	// Add node in layer
	Node *add_node(int layer);

	// Add arc
	void add_arc(Node *source, Node *target, bool one_arc, int length);

	// Merge nodeB into nodeA, adjusting incoming/outgoing arcs approprietly
	// (Obs.: Does not modify outgoing arcs from nodeB)
	void merge_nodes(Node *nodeA, Node *nodeB);

	// Remove node (not from layer)
	void remove_node(Node *node);

	// Get BDD width
	int get_width();

	// Get number of nodes
	int get_num_nodes();

	// Get number of nodes per layer
	vector<int> get_num_nodes_per_layer();

	// Get number of arcs
	int get_num_arcs();

	// Get average in-degree
	double get_average_in_degree();

	// Get number of nodes per layer
	// int *get_num_nodes_per_layer();

	// // get number of arcs per layer
	// int *get_num_arcs_per_layer();

	void prune_non_pareto_states(vector<vector<int>>);

	// Print BDD
	void print();

	// Ensure node indices are consistent
	void repair_node_indices();

	// return terminal
	Node *get_terminal()
	{
		return layers[num_layers - 1][0];
	}

	// Constructor
	BDD(int _num_layers) : num_layers(_num_layers)
	{
		layers.resize(num_layers);
		//	layers.reserve(num_layers);
	}

	// Destructor
	~BDD()
	{
		for (int l = 0; l < num_layers; ++l)
		{
			for (size_t i = 0; i < layers[l].size(); ++i)
			{
				// delete layers[l][i]->pareto_set;
				delete layers[l][i];
			}
		}
	}

	// Number of layers
	const int num_layers;
	// Set of layers
	vector<vector<Node *>> layers;
	//  vector< vector<Node*> > layers(num_layers);
};

// ---------------------------------------------------------------------------------------------
// Inline implementations
// ---------------------------------------------------------------------------------------------

//
// Node constructor
//
inline Node::Node(int _layer, int _index)
	: layer(_layer), index(_index),
	  one_arc(NULL), one_arc_length(0),
	  zero_arc(NULL), zero_arc_length(0), length(0),
	  active(true), exact(true), intState(-1), pareto_set(NULL)
{
	multi_length.resize(10);
	ub_longest_path.resize(10);
	for (int i = 0; i < 10; i++)
		ub_longest_path[i] = 0;
}

//
// Node destructor
//
inline Node::~Node()
{
	if (pareto_set != NULL)
	{
		delete pareto_set;
	}
}

//
// Add incoming arc
//
inline void Node::add_in_arc(Node *src, bool one_arc, int length)
{
	assert(src != NULL);
	assert(src->layer == this->layer - 1);
	if (one_arc)
	{
		src->one_arc = this;
		this->one_prev.push_back(src);
		src->one_arc_length = length;
	}
	else
	{
		src->zero_arc = this;
		this->zero_prev.push_back(src);
		src->zero_arc_length = length;
	}
}

//
// Add outgoing arc
//
inline void Node::add_out_arc(Node *tgt, bool one_arc, int length)
{
	assert(tgt != NULL);
	assert(tgt->layer == this->layer + 1);
	if (one_arc)
	{
		this->one_arc = tgt;
		tgt->one_prev.push_back(this);
		this->one_arc_length = length;
	}
	else
	{
		this->zero_arc = tgt;
		tgt->zero_prev.push_back(this);
		this->zero_arc_length = length;
	}
}

//
// Add node in layer
inline Node *BDD::add_node(int layer)
{
	assert(layer >= 0 && (int)layer < layers.size());
	layers[layer].push_back(new Node(layer, layers[layer].size()));
	return layers[layer].back();
}

//
// Add BDD arc
//
inline void BDD::add_arc(Node *src, Node *tgt, bool one_arc, int length)
{
	assert(src != NULL);
	assert(tgt != NULL);
	assert(src->layer == tgt->layer - 1);
	if (one_arc)
	{
		src->one_arc = tgt;
		tgt->one_prev.push_back(src);
		src->one_arc_length = length;
	}
	else
	{
		src->zero_arc = tgt;
		tgt->zero_prev.push_back(src);
		src->zero_arc_length = length;
	}
}

//
// Get BDD width
//
inline int BDD::get_width()
{
	size_t w = 0;
	for (size_t i = 0; i < layers.size(); ++i)
	{
		w = std::max(w, layers[i].size());
	}
	return w;
}

//
// Print BDD
//
inline void BDD::print()
{
	cout << endl;
	cout << "** BDD **" << endl;
	for (int l = 0; l < num_layers; ++l)
	{
		cout << "\tLayer " << l << endl;
		for (vector<Node *>::iterator it = layers[l].begin(); it != layers[l].end(); ++it)
		{
			Node *node = *it;
			cout << "\t\t";
			cout << node->index;
			if (node->zero_arc != NULL)
			{
				cout << " --> 0-arc (";
				cout << node->zero_arc->index;
				cout << ",";
				cout << node->zero_arc_length;
				cout << ")";
			}
			if (node->one_arc != NULL)
			{
				cout << " --> 1-arc (";
				cout << node->one_arc->index;
				cout << ",";
				cout << node->one_arc_length;
				cout << ")";
			}
			cout << endl;
		}
	}
	cout << "** Done **" << endl
		 << endl;
}

//
// Remove node
// TODO: also remove from the layer
//
void inline BDD::remove_node(Node *node)
{
	for (size_t i = 0; i < node->zero_prev.size(); ++i)
	{
		node->zero_prev[i]->zero_arc = NULL;
	}
	for (size_t i = 0; i < node->one_prev.size(); ++i)
	{
		node->one_prev[i]->one_arc = NULL;
	}
	if (node->zero_arc != NULL)
	{
		for (size_t i = 0; i < node->zero_arc->zero_prev.size(); ++i)
		{
			if (node->zero_arc->zero_prev[i] == node)
			{
				node->zero_arc->zero_prev[i] = node->zero_arc->zero_prev.back();
				node->zero_arc->zero_prev.pop_back();
				break;
			}
		}
	}
	if (node->one_arc != NULL)
	{
		for (size_t i = 0; i < node->one_arc->one_prev.size(); ++i)
		{
			if (node->one_arc->one_prev[i] == node)
			{
				node->one_arc->one_prev[i] = node->one_arc->one_prev.back();
				node->one_arc->one_prev.pop_back();
				break;
			}
		}
	}
	delete node;
}

//
// Ensure node indices are consistent
//
inline void BDD::repair_node_indices()
{
	for (int l = 0; l < num_layers; ++l)
	{
		for (size_t i = 0; i < layers[l].size(); ++i)
		{
			layers[l][i]->index = i;
		}
	}
}

//
// Get number of nodes of the BDD
//
inline int BDD::get_num_nodes()
{
	size_t num_nodes = 0;
	for (int l = 0; l < num_layers; ++l)
	{
		num_nodes += layers[l].size();
	}
	return num_nodes;
}

//
// Get number of nodes per layer the BDD
//
inline vector<int> BDD::get_num_nodes_per_layer()
{
	vector<int> num_nodes_per_layer;
	for (int l = 0; l < num_layers; ++l)
	{
		num_nodes_per_layer.push_back(layers[l].size());
	}
	return num_nodes_per_layer;
}

//
// Get number of arcs of the BDD
//
inline int BDD::get_num_arcs()
{
	size_t num_arcs = 0;
	for (int l = 0; l < num_layers; ++l)
	{
		for (int m = 0; m < layers[l].size(); ++m)
		{
			if (layers[l][m]->one_arc != NULL)
			{
				num_arcs += 1;
			}
			if (layers[l][m]->zero_arc != NULL)
			{
				num_arcs += 1;
			}
		}
	}
	return num_arcs;
}

inline double BDD::get_average_in_degree()
{
	double avg = 0;
	size_t count = 0;
	for (int l = 0; l < num_layers; ++l)
	{
		for (int m = 0; m < layers[l].size(); ++m)
		{
			avg += layers[l][m]->zero_prev.size() + layers[l][m]->one_prev.size();
			count += 1;
		}
	}
	avg = avg / count;

	return avg;
}

//
// Merge nodeB into nodeA, adjusting incoming/outgoing arcs approprietly
// (Obs.: Does not modify outgoing arcs from nodeB)
//
inline void BDD::merge_nodes(Node *nodeA, Node *nodeB)
{
	// readjust incoming/outgoing arcs
	for (vector<Node *>::iterator it = nodeB->zero_prev.begin();
		 it != nodeB->zero_prev.end(); ++it)
	{
		if (find(nodeA->zero_prev.begin(), nodeA->zero_prev.end(), *it) == nodeA->zero_prev.end())
		{
			nodeA->zero_prev.push_back(*it);
		}
		(*it)->zero_arc = nodeA;
	}
	for (vector<Node *>::iterator it = nodeB->one_prev.begin();
		 it != nodeB->one_prev.end(); ++it)
	{
		if (find(nodeA->one_prev.begin(), nodeA->one_prev.end(), *it) == nodeA->one_prev.end())
		{
			nodeA->one_prev.push_back(*it);
		}
		(*it)->one_arc = nodeA;
	}
}

inline void BDD::prune_non_pareto_states(vector<vector<int>> paretoStates)
{
	assert(paretoStates.size() == layers.size() - 2);
	int n;
	Node *node;
	bool isPareto;
	// For each layer [1, size(layers)-1)
	for (int l = 1; l < layers.size() - 1; ++l)
	{
		n = 0;
		// For each node in layer l
		while (n < layers[l].size())
		{
			node = layers[l][n];
			isPareto = false;
			// For each state to prune in layer l
			for (int s = 0; s < paretoStates[l - 1].size(); ++s)
			{
				// If the current Node is a Pareto state, don't do anything and process the next node
				if (node->intState == paretoStates[l - 1][s])
				{
					isPareto = true;
					++n;
					break;
				}
			}

			// If the current Node is not a Pareto State then remove it
			if (!isPareto)
			{
				// Remove incoming one-arcs
				for (vector<Node *>::iterator it = node->one_prev.begin();
					 it != node->one_prev.end();
					 ++it)
				{
					(*it)->one_arc = NULL;
				}
				node->one_prev.clear();

				// Remove incoming zero-arcs
				for (vector<Node *>::iterator it = node->zero_prev.begin();
					 it != node->zero_prev.end();
					 ++it)
				{
					(*it)->zero_arc = NULL;
				}
				node->zero_prev.clear();

				// Remove reference to the current node from the one_prev of node reached by one_arc
				if (node->one_arc != NULL)
				{
					for (size_t i = 0; i < node->one_arc->one_prev.size(); ++i)
					{
						if (node->one_arc->one_prev[i] == node)
						{
							node->one_arc->one_prev[i] = node->one_arc->one_prev.back();
							node->one_arc->one_prev.pop_back();
							break;
						}
					}
				}

				// Remove reference to the current node from the zero_prev of node reached by zero_arc
				if (node->zero_arc != NULL)
				{
					for (size_t i = 0; i < node->zero_arc->zero_prev.size(); ++i)
					{
						if (node->zero_arc->zero_prev[i] == node)
						{
							node->zero_arc->zero_prev[i] = node->zero_arc->zero_prev.back();
							node->zero_arc->zero_prev.pop_back();
							break;
						}
					}
				}

				layers[l][n] = layers[l].back();
				layers[l].pop_back();
				delete node;
			}
		}
	}
}

#endif /* BDD_HPP_ */

// ----------------------------------------------------------
