// ----------------------------------------------------------
// Portfolio Optimization BDD Constructor
// ----------------------------------------------------------

#ifndef PORTFOLIO_OPT_BDD_HPP_
#define PORTFOLIO_OPT_BDD_HPP_

#include <boost/unordered_map.hpp>
#include <boost/foreach.hpp>

#include "bdd.hpp"
#include "../instances/portfolio_opt_instance.hpp"

// 
// Knapsack BDD constructor
//
class PortfolioOptBDDConstructor {
public:
	// State definitions
	typedef vector<int> State;
	typedef boost::unordered_map<State, Node*> StateNodeMap;

	// Constructor
	PortfolioOptBDDConstructor(PortfolioInstance* _inst)
		: inst(_inst)
	{ }

	// Generate exact BDD
	BDD* generate_exact();

private:
	// Knapsack Instance
	PortfolioInstance* inst;
	// State maps
	StateNodeMap states[2];
};


#endif