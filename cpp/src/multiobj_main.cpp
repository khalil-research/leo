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
	/***
	 * Usage
	cout << endl;
	cout << "Usage: multiobj "
			"[input file] "
			"[feature weight 1] ... [feature 7]\n";

	Knapsack features
	1. weight
	2. avg_value
	3. max_value
	4. min_value
	5. avg_value_by_weight
	6. max_value_by_weight
	7. min_value_by_weight
	***/

	// catch input filename
	char *input_file = argv[1];

	int num_features = 7;
	vector<float> feature_weights;
	feature_weights.resize(num_features);
	// Catch weights
	for (int i = 0; i < num_features; i++)
	{
		feature_weights[i] = atof(argv[2 + i]);
	}

	KnapsackBDDSolver solver(input_file, feature_weights);
	solver.solve();

	return 0;
}
