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
	cout << '\n';
	cout << "Usage: multiobj "
			"[input file] [output file] [problem type] "
			"[order type weight 1] ... [order type weight n]\n";

	cout << "\n\twhere:\n";
	cout << "\t\tproblem_type = 1: knapsack\n";
	cout << "\t\tproblem_type = 2: independent set\n";
	cout << "\t\tproblem_type = 3: set packing\n";
	cout << "\t\tproblem_type = 4: set covering\n";
	
	Knapsack ordertypes
	1. max_weight
	2. min_weight
	3. max_avg_value
	4. min_avg_value
	5. max_max_value
	6. min_max_value
	7. max_min_value
	8. min_min_value
	9. max_avg_value_by_weight
	10. max_max_value_by_weight
	***/

	// read command line inputs
	char *input_file = argv[1];
	char *output_file = argv[2];
	int problem_type = atoi(argv[3]);
	int num_orders;

	vector<float> order_weights;
	if (problem_type == 1)
	{
		num_orders = 10;
		order_weights.resize(num_orders);
		float total_weight = 0;
		// Catch weights
		for (int i = 0; i < num_orders; i++)
		{
			order_weights[i] = atof(argv[4 + i]);
			// cout << argv[4 + i] << endl;
			// total_weight += order_weights[i];
		}

		// Normalize
		// for (int i = 0; i < num_orders; i++)
		// {
		// 	order_weights[i] /= total_weight;
		// }
	}

	// Seeds for generating random ordering
	KnapsackBDDSolver solver(input_file, output_file, order_weights);
	solver.solve();

	return 0;
}
