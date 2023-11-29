// --------------------------------------------------
// Multiobjective
// --------------------------------------------------

// General includes
#include <iostream>
#include <cstdlib>

#include "bdd/bdd.hpp"
#include "bdd/bdd_alg.hpp"
#include "bdd/bdd_multiobj.hpp"
#include "util/stats.hpp"
#include "util/util.hpp"
#include "bdd/pareto_frontier.hpp"

// Knapsack includes
#include "instances/knapsack_instance.hpp"
#include "bdd/knapsack_bdd.hpp"

// Set packing / Independent set includes
#include "instances/indepset_instance.hpp"
#include "instances/setpacking_instance.hpp"
#include "bdd/indepset_bdd.hpp"

// Set covering includes
#include "instances/setcovering_instance.hpp"
#include "bdd/setcovering_bdd.hpp"

// Portfolio optimization includes
#include "instances/portfolio_opt_instance.hpp"
#include "bdd/portfolio_opt_bdd.hpp"

// Absolute value instance
#include "instances/absval_instance.hpp"
#include "bdd/absval_bdd.hpp"

// TSP instance
#include "instances/tsp_instance.hpp"
#include "mdd/tsp_mdd.hpp"

using namespace std;

//
// Main function
//
int main(int argc, char *argv[])
{
    if (argc < 8)
    {
        cout << '\n';
        cout << "Usage: multiobj [input file] [problem type] [preprocess?] [method] [appr-S and T] [dominance] [num_items] [item1] ... [itemN]\n";

        cout << "\n\twhere:";

        cout << "\n";
        cout << "\t\tproblem_type = 1: knapsack\n";
        cout << "\t\tproblem_type = 2: set packing\n";
        cout << "\t\tproblem_type = 3: set covering\n";
        cout << "\t\tproblem_type = 4: portfolio optimization\n";
        cout << "\t\tproblem_type = 5: absolute value\n";
        cout << "\t\tproblem_type = 6: TSP\n";

        cout << "\n";
        cout << "\t\tpreprocess = 0: do not preprocess instance\n";
        cout << "\t\tpreprocess = 1: preprocess input to minimize BDD size\n";

        cout << "\n";
        cout << "\t\tmethod = 1: top-down BFS\n";
        cout << "\t\tmethod = 2: bottom-up BFS\n";
        cout << "\t\tmethod = 3: dynamic layer cutset\n";

        cout << "\n";
        cout << "\t\tapprox = n m: approximate n-sized S set and m-sized T set (n=0 if disabled)\n";

        cout << "\n";
        cout << "\t\tdominance = 0:  disable state dominance\n";
        cout << "\t\tdominance = 1:  state dominance strategy 1\n";

        cout << "\n";
        cout << "\t\tn_items = 0: Use default variable order\n";
        cout << "\t\tn_items = N: Number of items in the variable order\n";
        cout << "\t\titem1 (optional) = First item to use to construct the BDD \n";
        cout << "\t\t...\n";
        cout << "\t\titemN (optional) = Nth item to use to construct the BDD \n";
        exit(1);
    }

    // Read input
    int problem_type = atoi(argv[2]);
    bool preprocess = (argv[3][0] == '1');
    int method = atoi(argv[4]);
    bool maximization = true;
    int approx_S = atoi(argv[5]);
    int approx_T = atoi(argv[6]);
    int dominance = atoi(argv[7]);

    int num_items = atoi(argv[8]);
    bool new_order_provided = (num_items > 0);

    vector<int> new_order;
    for (int i = 0; i < num_items; i++)
    {
        new_order.push_back(atoi(argv[9 + i]));
    }

    // For statistical analysis
    Stats timers;
    int bdd_compilation_time = timers.register_name("BDD compilation time");
    int pareto_time = timers.register_name("BDD pareto time");
    int approx_time = timers.register_name("BDD approximation time");
    long int original_width;
    long int reduced_width;
    long int original_num_nodes;
    long int reduced_num_nodes;

    // Read problem instance and construct BDD
    BDD *bdd = NULL;
    vector<vector<int>> obj_coeffs;
    timers.start_timer(bdd_compilation_time);

    // --- Knapsack ---
    if (problem_type == 1)
    {

        // Read instance
        KnapsackInstance inst;
        inst.read(argv[1]);

        // if (preprocess) {
        //     // Reorder variables
        //     inst.reorder_coefficients();
        // }
        if (new_order_provided)
        {
            cout << "New order provided" << endl;
            inst.reset_order(new_order);
        }

        // Construct BDD
        KnapsackBDDConstructor bddCons(&inst);
        bdd = bddCons.generate_exact();
        // obj_coeffs = inst.obj_coeffs;

        original_width = bdd->get_width();
        original_num_nodes = bdd->get_num_nodes();

        // cout << "Original width: " << original_width << " - number of nodes: " << original_num_nodes << endl;

        // Reduce BDD
        BDDAlg::reduce(bdd);

        reduced_width = bdd->get_width();
        reduced_num_nodes = bdd->get_num_nodes();

        // cout << "Reduced width: " << reduced_width << " - number of nodes: " << reduced_num_nodes << endl;

        // Update node weights
        bddCons.update_node_weights(bdd);

        // Compute approximation
        if (approx_S != 0)
        {
            timers.start_timer(approx_time);
            // BDDMultiObj::approximate_pareto_frontier_bottomup(bdd, approx_S, approx_T);
            // BDDMultiObj::approximate_pareto_frontier_topdown(bdd, approx_S, approx_T);
            // BDDMultiObj::approximate_pareto_frontier_topdown_dominance(bdd, approx_S, approx_T);
            timers.end_timer(approx_time);
        }

        // Reduce BDD
        BDDAlg::reduce(bdd);

        reduced_width = bdd->get_width();
        reduced_num_nodes = bdd->get_num_nodes();

        // cout << "Reduced-2 width: " << reduced_width << " - number of nodes: " << reduced_num_nodes << endl;

        // Update node weights
        bddCons.update_node_weights(bdd);

        //        bdd->print();
    }

    // --- Set Packing ---
    else if (problem_type == 2)
    {

        // read instance
        SetPackingInstance setpack(argv[1]);

        // create associated independent set instance
        IndepSetInst *inst = setpack.create_indepset_instance();

        // generate independent set BDD
        IndepSetBDDConstructor bddConstructor(inst, setpack.objs);
        bdd = bddConstructor.generate_exact();

        original_width = bdd->get_width();
        original_num_nodes = bdd->get_num_nodes();

        reduced_width = bdd->get_width();
        reduced_num_nodes = bdd->get_num_nodes();
    }

    // --- Set Covering ---
    else if (problem_type == 3)
    {
        // set objective sense
        maximization = false;

        // read instance
        SetCoveringInstance setcover(argv[1]);

        // preprocess
        if (preprocess)
        {
            setcover.minimize_bandwidth();
        }

        // create BDD
        SetCoveringBDDConstructor bddConstructor(&setcover, setcover.objs);
        bdd = bddConstructor.generate_exact();

        original_width = bdd->get_width();
        original_num_nodes = bdd->get_num_nodes();

        // Reduce BDD
        // BDDAlg::reduce(bdd);

        reduced_width = bdd->get_width();
        reduced_num_nodes = bdd->get_num_nodes();
    }

    // --- Portfolio Optimization ---
    else if (problem_type == 4)
    {

        // Read instance
        PortfolioInstance inst;
        inst.read_BDD(argv[1]);

        // Construct BDD
        PortfolioOptBDDConstructor bddCons(&inst);
        bdd = bddCons.generate_exact();

        assert(bdd != NULL);
        original_width = bdd->get_width();
        original_num_nodes = bdd->get_num_nodes();

        // Reduce BDD
        // BDDAlg::reduce(bdd);

        reduced_width = bdd->get_width();
        reduced_num_nodes = bdd->get_num_nodes();
    }

    // --- AbsVal ---
    else if (problem_type == 5)
    {
        // Set objective sense
        maximization = false;

        AbsValInstance inst;
        inst.read_BDD(argv[1]);

        // Reset variable order
        if (new_order_provided)
        {
            cout << "New order provided" << endl;
            inst.reset_order(new_order);
        }

        AbsValBDDConstructor bddCons(&inst);
        bdd = bddCons.generate_exact();

        assert(bdd != NULL);
        original_width = bdd->get_width();
        original_num_nodes = bdd->get_num_nodes();

        // Reduce BDD
        // BDDAlg::reduce(bdd);

        reduced_width = bdd->get_width();
        reduced_num_nodes = bdd->get_num_nodes();
    }

    // --- TSP ---
    else if (problem_type == 6)
    {

        clock_t init_tsp = clock();

        // Read instance
        TSPInstance inst;
        inst.read(argv[1]);

        // Reset variable order
        if (new_order_provided)
        {
            cout << "New order provided" << endl;
            inst.reset_order(new_order);
        }

        // Construct MDD
        clock_t compilation_tsp = clock();

        MDDTSPConstructor mddCons(&inst);
        MDD *mdd = mddCons.generate_exact();
        assert(mdd != NULL);

        compilation_tsp = clock() - compilation_tsp;

        // Generate frontier
        clock_t frontier_tsp = clock();

        // cout << "\nGenerating frontier..." << endl;
        MultiObjectiveStats *statsMultiObj = new MultiObjectiveStats;
        ParetoFrontier *pareto_frontier = BDDMultiObj::pareto_frontier_dynamic_layer_cutset(mdd, statsMultiObj);
        assert(pareto_frontier != NULL);

        frontier_tsp = clock() - frontier_tsp;

        cout << pareto_frontier->get_num_sols() << endl;
        cout << (double)(compilation_tsp + frontier_tsp) / CLOCKS_PER_SEC << endl;
        cout << (double)compilation_tsp / CLOCKS_PER_SEC;
        cout << "\t" << frontier_tsp / CLOCKS_PER_SEC;
        cout << endl;

        return 0;
    }
    else
    {
        cout << "Error - problem type not recognized" << endl;
        exit(1);
    }

    timers.end_timer(bdd_compilation_time);

    // cout << "\nBDD Info:\n";
    // cout << "\tOriginal width: " << original_width << endl;
    // cout << "\tOriginal number of nodes: " << original_num_nodes << endl;
    // cout << "\n\tReduced width: " << reduced_width << endl;
    // cout << "\tReduced number of nodes: " << reduced_num_nodes << endl;
    // cout << "\n\tBDD compilation total time: " << timers.get_time(bdd_compilation_time) << endl;

    // Initialize multiobjective stats
    MultiObjectiveStats *statsMultiObj = new MultiObjectiveStats;

    // Compute pareto frontier based on methodology
    // cout << "\n\nComputing pareto frontier..." << endl;
    ParetoFrontier *pareto_frontier = NULL;
    timers.start_timer(pareto_time);
    cout << method << endl;
    if (method == 1)
    {
        // -- Optimal BFS algorithm: top-down --
        pareto_frontier = BDDMultiObj::pareto_frontier_topdown(bdd, maximization, problem_type, dominance, statsMultiObj);
    }
    else if (method == 2)
    {
        // -- Optimal BFS algorithm: bottom-up --
        pareto_frontier = BDDMultiObj::pareto_frontier_bottomup(bdd, maximization, problem_type, dominance, statsMultiObj);
    }
    else if (method == 3)
    {
        // -- Dynamic layer cutset --
        pareto_frontier = BDDMultiObj::pareto_frontier_dynamic_layer_cutset(bdd, maximization, problem_type, dominance, statsMultiObj);
    }

    if (pareto_frontier == NULL)
    {
        cout << "\nError - pareto frontier not computed" << endl;
        exit(1);
    }

    timers.end_timer(pareto_time);

    double total_time = (timers.get_time(bdd_compilation_time) + timers.get_time(approx_time) + timers.get_time(pareto_time));

    // cout << "\nPareto frontier: " << endl;
    // cout << "\tNumber of solutions: " << pareto_frontier->get_num_sols() << endl;
    // cout << "\n\tBDD time: " << timers.get_time(bdd_compilation_time) << endl;
    // cout << "\tApproximation filtering time: " << timers.get_time(approx_time) << endl;
    // cout << "\tPareto time: " << timers.get_time(pareto_time) << endl;
    // cout << "\tTotal time: " << total_time << endl;
    // cout << endl;

    // cout << "\n\nPareto frontier: " << endl;
    // pareto_frontier->print();
    // cout << endl;

    // // Statistics file
    // ofstream stats("stats.txt", ios::app);
    // stats << argv[1];
    // stats << "\t" << problem_type;
    // stats << "\t" << NOBJS;
    // stats << "\t" << preprocess;
    // stats << "\t" << method;
    // stats << "\t" << approx_S;
    // stats << "\t" << approx_T;
    // stats << "\t" << pareto_frontier->get_num_sols();
    // stats << "\t" << original_width;
    // stats << "\t" << original_num_nodes;
    // stats << "\t" << reduced_width;
    // stats << "\t" << reduced_num_nodes;
    // stats << "\t" << timers.get_time(bdd_compilation_time);
    // stats << "\t" << timers.get_time(pareto_time);
    // stats << "\t" << (timers.get_time(bdd_compilation_time) + timers.get_time(pareto_time));
    // stats << endl;
    // stats.close();

    // Original
    // cout << pareto_frontier->get_num_sols() << endl;
    // cout << (timers.get_time(bdd_compilation_time) + timers.get_time(pareto_time)) << endl;

    // cout << method;
    // cout << "\t" << dominance;
    // cout << "\t" << original_width;
    // cout << "\t" << reduced_width;
    // cout << "\t" << original_num_nodes;
    // cout << "\t" << reduced_num_nodes;
    // cout << "\t" << timers.get_time(bdd_compilation_time);
    // cout << "\t" << timers.get_time(pareto_time);
    // cout << "\t" << statsMultiObj->layer_coupling;
    // cout << "\t" << statsMultiObj->pareto_dominance_filtered;
    // cout << "\t" << ((double)statsMultiObj->pareto_dominance_time) / CLOCKS_PER_SEC;
    // cout << endl;

    cout << "Solved:";
    cout << pareto_frontier->get_num_sols() << ", ";
    cout << original_width << ", ";
    cout << reduced_width << ", ";
    cout << original_num_nodes << ", ";
    cout << reduced_num_nodes << ", ";
    cout << "-1, ";
    cout << "-1, ";
    // cout << "\t" << ((double)statsMultiObj->pareto_dominance_time) / CLOCKS_PER_SEC;
    cout << timers.get_time(bdd_compilation_time) << ", ";
    cout << "0, ";
    cout << timers.get_time(pareto_time);
    cout << " # ";
    cout << endl;

    return 0;
}
