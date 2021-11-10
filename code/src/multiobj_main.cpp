// --------------------------------------------------
// Multiobjective
// --------------------------------------------------

#include <ctime>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <list>
#include <random>
#include <vector>

// #include <ilcplex/ilocplex.h>

#include "bdd.hpp"
#include "bdd_util.hpp"
#include "pareto_util.hpp"
#include "stats.hpp"

#include "knapsack/knapsack_solver.hpp"

// #include "indepset/indepset_instance.hpp"
// #include "indepset/indepset_bdd.hpp"

// #include "setpacking/setpacking_instance.hpp"

// #include "setcovering/setcovering_instance.hpp"
// #include "setcovering/setcovering_bdd.hpp"

using namespace std;

// void solve(int problem_type, char *instance_file, char *out_file, int (&seeds)[5])
// {
// 	// void *inst = nullptr;

// 	if (problem_type == 1)
// 	{
// 		// read instance
// 		MultiObjKnapsackInstanceOrdered *inst = new MultiObjKnapsackInstanceOrdered(instance_file);
// 	}
// 	else
// 	{
// 		cout << "\nError: problem type not known.\n";
// 		exit(1);
// 	}
// 	inst.reset_order(order);

// 	// shuffle(foo.begin(), foo.end(), default_random_engine(seed));

// 	// Initialize timers (for stats)
// 	Stats timers;
// 	int bdd_compilation_time = timers.register_name("BDD compilation time");
// 	int bdd_reduction_time = timers.register_name("BDD reduction time");
// 	int bdd_pareto_time = timers.register_name("Pareto time");

// 	// BDD and objective function coefficients
// 	BDD *bdd = NULL;
// 	vector<vector<int>> obj_coefficients;

// 	// Read problem and construct BDDs
// 	timers.start_timer(bdd_compilation_time);

// 	// MultiObjKnapsackInstance inst(argv[1]);

// 	// create initial BDD
// 	KnapsackBDDConstructor bddConstructor(
// 		new KnapsackInstance(inst.n_vars, inst.obj_coeffs[0], inst.coeffs, inst.rhs),
// 		0);

// 	bdd = bddConstructor.generate_exact();

// 	// get objective function coefficients
// 	obj_coefficients = inst.obj_coeffs;

// 	timers.end_timer(bdd_compilation_time);

// 	// Reduce BDD
// 	cout << "\nReducing BDD...\n";
// 	timers.start_timer(bdd_reduction_time);

// 	int initial_width = bdd->get_width();
// 	int initial_node_size = bdd->get_num_nodes();

// 	BDDAlg::reduce(bdd);

// 	int reduced_width = bdd->get_width();
// 	int reduced_node_size = bdd->get_num_nodes();

// 	cout << "\tinitial width = " << initial_width << '\n';
// 	cout << "\treduced width = " << reduced_width << '\n';

// 	timers.end_timer(bdd_reduction_time);

// 	// Generate pareto set
// 	timers.start_timer(bdd_pareto_time);

// 	ParetoSet *paretoSet = BDDAlg::pareto_set(bdd, obj_coefficients);

// 	timers.end_timer(bdd_pareto_time);

// 	// print statistics
// 	cout << "Stats: " << endl;
// 	cout << "\tInitial BDD size: width = " << initial_width << " - nodes = " << initial_node_size << '\n';
// 	cout << "\tReduced BDD size: width = " << reduced_width << " - nodes = " << reduced_node_size << '\n';
// 	cout << "\tSize of pareto set: " << paretoSet->sols.size() << '\n';
// 	cout << "\tBDD compilation time: " << timers.get_time(bdd_compilation_time) << "s\n";
// 	cout << "\tBDD reduction time: " << timers.get_time(bdd_reduction_time) << "s\n";
// 	cout << "\tPareto time: " << timers.get_time(bdd_pareto_time) << "s\n";
// 	cout << '\n';

// 	ofstream instance_result;
// 	string out = argv[6];
// 	instance_result.open(out, ios::app);
// 	instance_result << argv[1] << ", ";
// 	instance_result << initial_width << ", ";
// 	instance_result << reduced_width << ", ";
// 	instance_result << paretoSet->sols.size() << ", ";
// 	instance_result << timers.get_time(bdd_compilation_time) << ", ";
// 	instance_result << timers.get_time(bdd_reduction_time) << ", ";
// 	instance_result << timers.get_time(bdd_pareto_time) << endl;
// 	instance_result.close();
// }

//
// Main function
//
int main(int argc, char *argv[])
{
	if (argc != 5)
	{
		cout << '\n';
		cout << "Usage: multiobj [input file] [problem type] [preprocess] [outpath]\n";

		cout << "\n\twhere:\n";
		cout << "\t\tproblem_type = 1: knapsack\n";
		cout << "\t\tproblem_type = 2: independent set\n";
		cout << "\t\tproblem_type = 3: set packing\n";
		cout << "\t\tproblem_type = 4: set covering\n";

		cout << "\n";
		cout << "\t\tpreprocess = 0: no preprocessing\n";
		cout << "\t\tpreprocess = 1: minimize bandwidth (only setpack/setcover)\n";

		cout << "\n";
		exit(1);
	}

	// read input
	int problem_type = atoi(argv[2]);
	bool preprocess = (atoi(argv[3]) == 1);
	// Seeds for generating random ordering
	int seeds[] = {13, 444, 1212, 1003, 7517};
	KnapsackBDDSolver solver(argv[1], argv[4], seeds);
	solver.solve();

	return 0;
}
