#include "knapsack_solver.hpp"

KnapsackBDDSolver::KnapsackBDDSolver(char *instance_file,
                                     char *output_file,
                                     int (&seeds)[5]) : instance_file(instance_file),
                                                        output_file(output_file),
                                                        seeds(seeds)
{
    inst = new MultiObjKnapsackInstanceOrdered(instance_file);
};

KnapsackBDDSolver::~KnapsackBDDSolver()
{
    delete inst;
}

string KnapsackBDDSolver::vint_to_str(vector<int> order)
{
    ostringstream stream;
    for (int i : order)
    {
        stream << i << "|";
    }
    string order_str(stream.str());

    return order_str;
}

void KnapsackBDDSolver::generate_orders()
{
    order_map.clear();
    int s_count = 0;
    string order_name;

    // Generate random orders
    for (const auto &s : seeds)
    {
        order_name = "rnd" + to_string(s_count);
        order_map.insert({order_name, {}});
        for (int i = 0; i < inst->n_vars; i++)
        {
            order_map[order_name].push_back(i);
        }
        shuffle(order_map[order_name].begin(),
                order_map[order_name].end(),
                std::default_random_engine((unsigned)s));
        s_count++;
    }

    // Generate heuristic orders
    for (const auto &ot : kp::all_OrderType)
    {
        order_name = kp::get_order_name(ot);
        order_map.insert({order_name,
                          kp::get_order(ot,
                                        inst->coeffs_canonical,
                                        inst->obj_coeffs_canonical)});
    }
}

void KnapsackBDDSolver::solve()
{
    BDD *bdd = NULL;
    vector<vector<int>> obj_coefficients;

    long int initial_width, initial_node_count, initial_arcs_count;
    long int reduced_width, reduced_node_count, reduced_arcs_count;
    ParetoSet *paretoSet = NULL;

    ofstream output;
    Stats timers;
    int bdd_compilation_time = timers.register_name("BDD compilation time");
    int bdd_reduction_time = timers.register_name("BDD reduction time");
    int bdd_pareto_time = timers.register_name("Pareto time");

    generate_orders();
    for (const auto &om : order_map)
    {
        // Reset timers
        timers.reset_timer(bdd_compilation_time);
        timers.reset_timer(bdd_reduction_time);
        timers.reset_timer(bdd_pareto_time);

        // Modify order
        inst->reset_order(om.second);

        ////////////////////////////////////////////////////
        // Read problem and construct BDDs
        timers.start_timer(bdd_compilation_time);
        // Construct BDD
        KnapsackBDDConstructor bddConstructor(
            new KnapsackInstance(inst->n_vars, inst->obj_coeffs[0], inst->coeffs, inst->rhs),
            0);
        bdd = bddConstructor.generate_exact();

        // get objective function coefficients
        obj_coefficients = inst->obj_coeffs;
        timers.end_timer(bdd_compilation_time);

        ////////////////////////////////////////////////////
        // Reduce BDD
        // cout << "\nReducing BDD...\n";
        timers.start_timer(bdd_reduction_time);

        initial_width = bdd->get_width();
        initial_node_count = bdd->get_num_nodes();
        initial_arcs_count = bdd->get_num_arcs();

        BDDAlg::reduce(bdd);

        reduced_width = bdd->get_width();
        reduced_node_count = bdd->get_num_nodes();
        reduced_arcs_count = bdd->get_num_arcs();

        // cout << "\tinitial width = " << initial_width << '\n';
        // cout << "\treduced width = " << reduced_width << '\n';

        timers.end_timer(bdd_reduction_time);

        ////////////////////////////////////////////////////

        // Generate pareto set
        timers.start_timer(bdd_pareto_time);
        paretoSet = BDDAlg::pareto_set(bdd, obj_coefficients);
        timers.end_timer(bdd_pareto_time);

        ////////////////////////////////////////////////////

        // log
        output.open(output_file, ios::app);
        output << instance_file << ", ";
        output << om.first << ", ";
        output << initial_width << ", ";
        output << reduced_width << ", ";
        output << initial_node_count << ", ";
        output << reduced_node_count << ", ";
        output << initial_arcs_count << ", ";
        output << reduced_arcs_count << ", ";
        output << timers.get_time(bdd_compilation_time) << ", ";
        output << timers.get_time(bdd_reduction_time) << ", ";
        output << timers.get_time(bdd_pareto_time) << ", ";
        output << paretoSet->sols.size() << ", ";
        output << vint_to_str(om.second) << endl;
        output.close();

        delete bdd;
    }
}
