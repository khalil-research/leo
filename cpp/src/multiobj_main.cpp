// --------------------------------------------------
// Multiobjective
// --------------------------------------------------

#include <ctime>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <list>
#include <random>
#include <sstream> // for std::stringstream
#include <vector>

// #include <ilcplex/ilocplex.h>

#include "bdd.hpp"
#include "bdd_util.hpp"
#include "pareto_util.hpp"
#include "stats.hpp"

#include "knapsack/knapsack_solver.hpp"

using namespace std;

//
// Main function
//
int main(int argc, char *argv[])
{
	// catch input filename
	char *input_file = argv[1];
	int num_items = atoi(argv[2]);
	vector<int> new_order;
	// Catch order
	for (int i = 0; i < num_items; i++)
	{
		new_order.push_back(atoi(argv[3 + i]));
	}

	KnapsackBDDSolver solver(input_file, new_order);
	solver.solve();

	return 0;
}
