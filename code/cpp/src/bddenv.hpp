#pragma once

#include <vector>
#include <string>
#include <map>

#include "stats.hpp"
#include "bdd.hpp"
#include "bdd_util.hpp"

#include "knapsack/knapsack_instance.hpp"
#include "knapsack/knapsack_instance_ordered.hpp"
#include "knapsack/knapsack_bdd.hpp"

#define RESET 0
#define SUCCESS 1
#define ERROR 2

using namespace std;

class BDDEnv
{
private:
    void initialize();
    void get_time_result();
    void get_num_pareto_sols_per_layer();
    void get_pareto_sols();

public:
    // ----------------------------------------------------------------
    // Inputs
    int problem_type;
    string filepath;
    bool preprocess;
    int bdd_type;
    int maxwidth;
    vector<int> order;
    bool anytime;

    // ----------------------------------------------------------------
    // Run status
    int status = RESET;

    // ----------------------------------------------------------------
    // Timers (for stats)
    Stats timers;
    int bdd_compilation_time = timers.register_name("BDD compilation time");
    int bdd_reduction_time = timers.register_name("BDD reduction time");
    int bdd_pareto_time = timers.register_name("Pareto time");
    map<string, float> time_result;

    // ----------------------------------------------------------------
    // BDD and its stats
    BDD *bdd;
    size_t initial_width, initial_node_count, initial_arcs_count;
    size_t reduced_width, reduced_node_count, reduced_arcs_count;
    double initial_avg_in_degree, reduced_avg_in_degree;
    vector<int> initial_num_nodes_per_layer, reduced_num_nodes_per_layer;

    // ----------------------------------------------------------------
    // MOO Result
    MultiobjResult *mo_result;
    unsigned long int nnds = 0;
    unsigned long int num_comparisons = 0;
    vector<unsigned long int> num_pareto_sol_per_layer;
    vector<vector<int>> x_sol;
    vector<vector<int>> z_sol;

    BDDEnv();

    ~BDDEnv();

    void reset(int problem_type,
               string filepath,
               bool preprocess,
               int bdd_type,
               int maxwidth,
               vector<int> order,
               bool anytime);

    void compute_pareto_frontier();

    void compute_pareto_frontier_with_pruning(vector<vector<int>> statesToPrune);
};