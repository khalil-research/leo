#include "bddenv.hpp"

BDDEnv::BDDEnv()
{
    initialize();
}

BDDEnv::~BDDEnv()
{
    // Clean memory
    delete bdd;
    delete mo_result;
}

void BDDEnv::initialize()
{
    // ----------------------------------------------------------------
    // Run status
    status = RESET;

    // ----------------------------------------------------------------
    // Timers (for stats)
    timers.reset_timer(bdd_compilation_time);
    timers.reset_timer(bdd_reduction_time);
    timers.reset_timer(bdd_pareto_time);
    time_result.clear();

    // ----------------------------------------------------------------
    // BDD and its stats
    bdd = NULL;
    initial_width = 0, initial_node_count = 0, initial_arcs_count = 0;
    reduced_width = 0, reduced_node_count = 0, reduced_arcs_count = 0;
    initial_avg_in_degree = 0, reduced_avg_in_degree = 0;
    initial_num_nodes_per_layer.clear();
    reduced_num_nodes_per_layer.clear();

    // ----------------------------------------------------------------
    // MOO Result
    mo_result = NULL;
    nnds = 0;
    num_comparisons = 0;
    num_pareto_sol_per_layer.clear();
    x_sol.clear();
    z_sol.clear();
}

void BDDEnv::reset(int _problem_type,
                   string _filepath,
                   bool _preprocess,
                   int _bdd_type,
                   int _maxwidth,
                   vector<int> _order,
                   bool _anytime)
{

    problem_type = _problem_type;
    filepath = _filepath;
    preprocess = _preprocess;
    bdd_type = _bdd_type;
    maxwidth = _maxwidth;
    order = _order;
    anytime = _anytime;

    initialize();
}

void BDDEnv::compute_pareto_frontier()
{
    vector<vector<int>> obj_coefficients;
    ofstream output;

    // -------------------------------------------------
    // Read problem and construct BDDs
    timers.reset_timer(bdd_compilation_time);
    timers.start_timer(bdd_compilation_time);

    const int length = filepath.length();
    // declaring character array (+1 for null terminator)
    char *input_file = new char[length + 1];
    // copying the contents of the
    // string to char array
    strcpy(input_file, filepath.c_str());

    // ---- Knapsack ----
    if (problem_type == 1)
    {

        MultiObjKnapsackInstanceOrdered *inst = new MultiObjKnapsackInstanceOrdered(input_file);
        if (order.size())
        {
            cout << "\tReordering instance based on input order...";
            inst->reset_order(order);
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
    initial_num_nodes_per_layer = bdd->get_num_nodes_per_layer();

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
    reduced_num_nodes_per_layer = bdd->get_num_nodes_per_layer();

    // bdd->print();

    // -------------------------------------------------
    // Generate pareto set
    timers.reset_timer(bdd_pareto_time);
    timers.start_timer(bdd_pareto_time);
    mo_result = BDDAlg::pareto_set(bdd, obj_coefficients);
    timers.end_timer(bdd_pareto_time);

    // Output
    status = SUCCESS;
    get_time_result();
    nnds = mo_result->pareto_set->sols.size();
    num_comparisons = mo_result->num_comparisons;
    get_num_pareto_sols_per_layer();
    get_pareto_sols();
}

void BDDEnv::compute_pareto_frontier_with_pruning(vector<vector<int>> paretoStates)
{
    vector<vector<int>> obj_coefficients;
    ofstream output;

    // -------------------------------------------------
    // Read problem and construct BDDs
    timers.reset_timer(bdd_compilation_time);
    timers.start_timer(bdd_compilation_time);

    const int length = filepath.length();
    // declaring character array (+1 for null terminator)
    char *input_file = new char[length + 1];
    // copying the contents of the
    // string to char array
    strcpy(input_file, filepath.c_str());

    // ---- Knapsack ----
    if (problem_type == 1)
    {
        MultiObjKnapsackInstanceOrdered *inst = new MultiObjKnapsackInstanceOrdered(input_file);
        if (order.size())
        {
            cout << "\tReordering instance based on input order...";
            inst->reset_order(order);
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
    initial_num_nodes_per_layer = bdd->get_num_nodes_per_layer();

    // -------------------------------------------------
    // Prune BDD
    bdd->prune_non_pareto_states(paretoStates);

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
    reduced_num_nodes_per_layer = bdd->get_num_nodes_per_layer();

    // bdd->print();

    // -------------------------------------------------
    // Generate pareto set
    timers.reset_timer(bdd_pareto_time);
    timers.start_timer(bdd_pareto_time);
    mo_result = BDDAlg::pareto_set(bdd, obj_coefficients);
    timers.end_timer(bdd_pareto_time);

    // mo_result->pareto_set->print_objectives(objfile);
    // Output
    status = SUCCESS;
    get_time_result();
    nnds = mo_result->pareto_set->sols.size();
    num_comparisons = mo_result->num_comparisons;
    get_num_pareto_sols_per_layer();
    get_pareto_sols();
}

void BDDEnv::get_time_result()
{
    time_result = {
        {"compilation", timers.get_time(bdd_compilation_time)},
        {"reduction", timers.get_time(bdd_reduction_time)},
        {"pareto", timers.get_time(bdd_pareto_time)}};
}

void BDDEnv::get_num_pareto_sols_per_layer()
{

    for (int i = 0; i < bdd->num_layers; i++)
    {
        num_pareto_sol_per_layer.push_back(mo_result->num_pareto_sol[i]);
    }
}

void BDDEnv::get_pareto_sols()
{
    x_sol = mo_result->pareto_set->get_x_sols();
    z_sol = mo_result->pareto_set->get_z_sols();
}
