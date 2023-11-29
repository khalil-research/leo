// ----------------------------------------------------------
// Absolute Value BDD Constructor
// ----------------------------------------------------------

#ifndef ABSVAL_BDD_HPP_
#define ABSVAL_BDD_HPP_

#include <boost/unordered_map.hpp>
#include <boost/foreach.hpp>

#include "bdd.hpp"
#include "../instances/absval_instance.hpp"

// 
// Absolute Value BDD constructor
//
class AbsValBDDConstructor {
public:
	// State definitions
	typedef vector<int> State;
	typedef boost::unordered_map<State, Node*> StateNodeMap;

	// Constructor
	AbsValBDDConstructor(AbsValInstance* _inst)
		: inst(_inst)
	{ }

	// Generate exact BDD
	BDD* generate_exact();

private:
	// Knapsack Instance
	AbsValInstance* inst;
	// State maps
	StateNodeMap states[2];
};


#endif