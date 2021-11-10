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
	if (argc != 5)
	{
		cout << '\n';
		cout << "Usage: multiobj [input file] [problem type] [order type] [outpath]\n";

		cout << "\n\twhere:\n";
		cout << "\t\tproblem_type = 1: knapsack\n";
		cout << "\t\tproblem_type = 2: independent set\n";
		cout << "\t\tproblem_type = 3: set packing\n";
		cout << "\t\tproblem_type = 4: set covering\n";

		cout << "\n\twhere:\n";
		cout << "\t\t order_id = 1: max_weight\n";
		cout << "\t\t order_id = 2: min_weight\n";
		cout << "\t\t order_id = 3: max_avg_value\n";
		cout << "\t\t order_id = 4: min_avg_value\n";
		cout << "\t\t order_id = 5: max_max_value\n";
		cout << "\t\t order_id = 6: min_max_value\n";
		cout << "\t\t order_id = 7: max_min_value\n";
		cout << "\t\t order_id = 8: min_min_value\n";
		cout << "\t\t order_id = 9: max_avg_value_by_weight\n";
		cout << "\t\t order_id = 10: max_max_value_by_weight\n";

		cout << "\n";
		exit(1);
	}

	// read command line inputs
	char *input_file = argv[1];
	int problem_type = atoi(argv[2]);
	int order_id = atoi(argv[3]);
	char *output_file = argv[4];

	// Seeds for generating random ordering
	KnapsackBDDSolver solver(input_file, output_file, order_id);
	solver.solve();

	return 0;
}
