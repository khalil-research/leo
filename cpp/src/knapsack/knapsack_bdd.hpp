// ----------------------------------------------------------
// Knapsack BDD Constructor
// ----------------------------------------------------------

#ifndef KNAPSACK_BDD_HPP_
#define KNAPSACK_BDD_HPP_

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

#include "knapsack_instance.hpp"

using namespace std;

//
// Restricted BDD Knapsack constructor
//
class KnapsackBDDConstructor
{
public:
	// State definition
	typedef int State;
	typedef boost::unordered_map<State, Node *> StateNodeMap;

	// Constructor
	KnapsackBDDConstructor(KnapsackInstance *_inst, int _maxwidth)
		: inst(_inst), maxwidth(_maxwidth), MAXW(10000)
	{
	}

	// Generate exact BDD
	BDD *generate_exact();

	// Destructor
	~KnapsackBDDConstructor();

private:
	// Knapsack instance
	KnapsackInstance *inst;
	// Maximum width
	const int maxwidth;
	// Maximum width for memory
	const int MAXW;

	// State maps
	StateNodeMap states[2];
};

#endif /* KNAPSACK_BDD_HPP_ */

// ----------------------------------------------------------
