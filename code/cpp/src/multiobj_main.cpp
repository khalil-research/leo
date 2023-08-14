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

#include "bdd.hpp"
#include "bdd_util.hpp"
#include "pareto_util.hpp"
#include "stats.hpp"

#include "knapsack/knapsack_instance.hpp"
#include "knapsack/knapsack_instance_ordered.hpp"
#include "knapsack/knapsack_bdd.hpp"

#include "indepset/indepset_instance.hpp"
#include "indepset/indepset_bdd.hpp"
#include "setpacking/setpacking_instance.hpp"
#include "setcovering/setcovering_instance.hpp"
#include "setcovering/setcovering_bdd.hpp"

using namespace std;

//
// Main function
//
int main(int argc, char *argv[])
{

	if (argc <= 4)
	{
		// Usage instructions
		cout << '\n';
		cout << "Usage: multiobj [input file] [problem type] [preprocess] [bdd_type] [max_width] [num_items] [item_1] ... [item_<num_items>]\n";

		cout << "\n\twhere:\n";
		cout << "\t\tproblem_type = 1: knapsack\n";
		cout << "\t\tproblem_type = 2: set packing\n";
		cout << "\t\tproblem_type = 3: set covering\n";

		cout << "\n";
		cout << "\t\tpreprocess = 0: no preprocessing\n";
		cout << "\t\tpreprocess = 1: minimize bandwidth (only setpack/setcover)\n";

		cout << "\n";
		cout << "\t\tbdd_type = 0: Generate exact BDD\n";
		cout << "\t\tbdd_type = 1: Generate restricted BDD\n";

		cout << "\n";
		cout << "\t\tmax_width : Width of restricted BDD.\n";

		cout << "\n";
	}

	// -------------------------------------------------
	// Read commnad line input(for stats)

	// catch input filename
	char *input_file = argv[1];

	// catch problem type
	int problem_type = atoi(argv[2]);

	// catch preprocess
	bool preprocess = (atoi(argv[3]) == 1);

	// 0: Exact, 1: Restricted, 2: Relaxed
	int bdd_type = atoi(argv[4]);

	int maxwidth = atoi(argv[5]);

	// Catch number of variables
	int num_items = atoi(argv[6]);
	bool new_order_provided = (num_items > 0);

	// Catch order
	vector<int> new_order;
	for (int i = 0; i < num_items; i++)
	{
		new_order.push_back(atoi(argv[7 + i]));
	}

	// -------------------------------------------------
	// Initialize timers (for stats)
	Stats timers;
	int bdd_compilation_time = timers.register_name("BDD compilation time");
	int bdd_reduction_time = timers.register_name("BDD reduction time");
	int bdd_pareto_time = timers.register_name("Pareto time");

	size_t initial_width, initial_node_count, initial_arcs_count;
	size_t reduced_width, reduced_node_count, reduced_arcs_count;
	double initial_avg_in_degree, reduced_avg_in_degree;

	// BDD and objective function coefficients
	BDD *bdd = NULL;
	vector<vector<int>> obj_coefficients;

	// Result object
	MultiobjResult *mo_result = NULL;

	ofstream output;

	// -------------------------------------------------
	// Read problem and construct BDDs

	timers.reset_timer(bdd_compilation_time);
	timers.start_timer(bdd_compilation_time);

	// ---- Knapsack ----
	if (problem_type == 1)
	{
		MultiObjKnapsackInstanceOrdered *inst = new MultiObjKnapsackInstanceOrdered(input_file);
		if (new_order_provided)
		{
			cout << "\tReordering instance based on input order...";
			inst->reset_order(new_order);
		}

		// Construct BDD
		KnapsackInstance *inst_so = new KnapsackInstance(inst->n_vars, inst->obj_coeffs[0], inst->coeffs, inst->rhs);

		// Get BDD and objective function coefficients
		if (bdd_type == 0)
		{
			KnapsackBDDConstructor bddConstructor(inst_so, 0);
			bdd = bddConstructor.generate_exact();
		}
		else
		{
			KnapsackBDDConstructor bddConstructor(inst_so, maxwidth);
			bdd = bddConstructor.generate_restricted();
		}
		obj_coefficients = inst->obj_coeffs;
	}

	// ---- Set packing ----
	else if (problem_type == 2)
	{
		// read instance
		SetPackingInstance setpack(argv[1]);

		// Reset instances based on new order
		if (new_order_provided)
		{
			cout << "\tReordering instance based on input order...";
			setpack.reset_order(new_order);
		}

		// preprocess only if a new order is not provided
		if (!new_order_provided && preprocess)
		{
			setpack.minimize_bandwidth();
		}

		// reduce set packing instance to an independent set problem
		IndepSetInst *inst = setpack.create_indepset_instance();

		// create artificial objective vector with one unit
		vector<vector<int>> obj_coeffs = setpack.objs;

		// create BDD with unitary objective function
		IndepSetBDDConstructor bddConstructor(inst, obj_coeffs, 0);
		bdd = bddConstructor.generate_exact(new_order_provided);

		// remap objective function vector to consider variable ordering
		vector<int> &var_layer = bddConstructor.var_layer;
		obj_coefficients.resize(setpack.objs.size(), vector<int>(setpack.n_vars, -1));
		for (size_t p = 0; p < setpack.objs.size(); ++p)
		{
			for (int i = 0; i < setpack.n_vars; ++i)
			{
				obj_coefficients[p][i] = setpack.objs[p][var_layer[i]];
			}
		}

		delete inst;
	}

	// ---- Set covering ----
	else if (problem_type == 3)
	{
		cout << "Set cover\n";
		// read instance
		SetCoveringInstance setcover(argv[1]);

		// Reset instances based on new order
		if (new_order_provided)
		{
			setcover.reset_order(new_order);
		}

		// preprocess
		if (!new_order_provided && preprocess)
		{
			setcover.minimize_bandwidth();
		}

		// create initial BDD
		SetCoveringBDDConstructor bddConstructor(&setcover, setcover.objs, 0);
		bdd = bddConstructor.generate_exact();

		// remap objective function vector to consider variable ordering
		vector<int> &var_layer = bddConstructor.var_layer;
		obj_coefficients.resize(setcover.objs.size(), vector<int>(setcover.n_vars, -1));
		for (size_t p = 0; p < setcover.objs.size(); ++p)
		{
			for (int i = 0; i < setcover.n_vars; ++i)
			{
				// cout << var_layer[i] << " ";
				obj_coefficients[p][i] = setcover.objs[p][var_layer[i]];
			}
			// cout << "\n";
		}
	}

	// ---- Invalid problem selection ----
	else
	{
		cout << "\nError: problem type not known.\n";
		exit(1);
	}
	timers.end_timer(bdd_compilation_time);

	initial_width = bdd->get_width();
	initial_node_count = bdd->get_num_nodes();
	initial_arcs_count = bdd->get_num_arcs();
	initial_avg_in_degree = bdd->get_average_in_degree();

	// -------------------------------------------------
	// Reduce BDD

	// cout << "\nReducing BDD...\n";

	timers.reset_timer(bdd_reduction_time);
	timers.start_timer(bdd_reduction_time);

	BDDAlg::reduce(bdd);

	timers.end_timer(bdd_reduction_time);

	reduced_width = bdd->get_width();
	reduced_node_count = bdd->get_num_nodes();
	reduced_arcs_count = bdd->get_num_arcs();
	reduced_avg_in_degree = bdd->get_average_in_degree();

	// bdd->print();

	// -------------------------------------------------
	// Generate pareto set
	timers.reset_timer(bdd_pareto_time);
	timers.start_timer(bdd_pareto_time);
	mo_result = BDDAlg::pareto_set(bdd, obj_coefficients);
	timers.end_timer(bdd_pareto_time);

	// -------------------------------------------------
	// Output
	cout << "Solved:";
	cout << mo_result->pareto_set->sols.size() << ", ";
	cout << initial_width << ", ";
	cout << reduced_width << ", ";
	cout << initial_node_count << ", ";
	cout << reduced_node_count << ", ";
	cout << initial_arcs_count << ", ";
	cout << reduced_arcs_count << ", ";
	cout << initial_avg_in_degree << ", ";
	cout << reduced_avg_in_degree << ", ";
	cout << mo_result->num_comparisons << ", ";

	cout << timers.get_time(bdd_compilation_time) << ", ";
	cout << timers.get_time(bdd_reduction_time) << ", ";
	cout << timers.get_time(bdd_pareto_time);
	mo_result->print_num_pareto_sol();

	// Clean memory
	delete bdd;
	delete mo_result;

	return 0;
}
