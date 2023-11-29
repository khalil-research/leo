// ----------------------------------------------------------
// Knapsack BDD Constructor
// ----------------------------------------------------------

#ifndef KNAPSACK_BDD_HPP_
#define KNAPSACK_BDD_HPP_

#include <boost/unordered_map.hpp>
#include <boost/foreach.hpp>

#include "bdd.hpp"
#include "../instances/knapsack_instance.hpp"

// 
// Knapsack BDD constructor
//
class KnapsackBDDConstructor {
public:
	// State definitions
	typedef vector<int> State;
	typedef boost::unordered_map<State, Node*> StateNodeMap;

	// Constructor
	KnapsackBDDConstructor(KnapsackInstance* _inst)
		: inst(_inst)
	{ }

	// Generate exact BDD
	BDD* generate_exact();

	// Update node weights
	void update_node_weights(BDD* bdd);

private:
	// Knapsack Instance
	KnapsackInstance* inst;
	// State maps
	StateNodeMap states[2];
};


#endif