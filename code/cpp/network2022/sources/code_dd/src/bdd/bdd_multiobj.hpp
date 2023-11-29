// ----------------------------------------------------------
// BDD Multiobjective Algorithms
// ----------------------------------------------------------

#ifndef BDD_MULTIOBJ_HPP_
#define BDD_MULTIOBJ_HPP_

#include "../mdd/mdd.hpp"
#include "../util/util.hpp"
#include "bdd.hpp"
#include "pareto_frontier.hpp"


//
// Multiobjective stats
//
struct MultiObjectiveStats {
    // Time spent in pareto dominance filtering
    clock_t pareto_dominance_time;
    // Solutions filtered by pareto dominance
    int pareto_dominance_filtered;
    // Layer where coupling happened
    int layer_coupling;

    // Constructor
    MultiObjectiveStats() 
    : pareto_dominance_time(0), pareto_dominance_filtered(0), layer_coupling(0)
    { }
};


//
// BDD Multiobjective Algorithms
//
struct BDDMultiObj {
    // Find pareto frontier from top-down approach
	static ParetoFrontier* pareto_frontier_topdown(BDD* bdd, bool maximization=true, const int problem_type=-1, const int dominance_strategy=0, MultiObjectiveStats* stats = NULL);

    // Find pareto frontier from bottom-up approach
	static ParetoFrontier* pareto_frontier_bottomup(BDD* bdd, bool maximization=true, const int problem_type=-1, const int dominance_strategy=0, MultiObjectiveStats* stats = NULL);

    // Find pareto frontier using dynamic layer cutset
    static ParetoFrontier* pareto_frontier_dynamic_layer_cutset(BDD* bdd, bool maximization=true, const int problem_type=-1, const int dominance_strategy=0, MultiObjectiveStats* stats = NULL);

    // Approximate pareto frontier / top-down
    static void approximate_pareto_frontier_topdown(BDD* bdd, const int s_max, const int t_max);

    // Approximate pareto frontier with dominance filtering / top-down
    static void approximate_pareto_frontier_topdown_dominance(BDD* bdd, const int s_max, const int t_max);
    
    // Approximate pareto frontier / bottom-up
    static void approximate_pareto_frontier_bottomup(BDD* bdd, const int s_max, const int t_max);

    // Filter layer based on dominance
    static void filter_dominance(BDD* bdd, const int layer, const int problem_type, const int dominance_strategy, MultiObjectiveStats* stats);
    
    // Filter layer based on dominance / knapsack
    static void filter_dominance_knapsack(BDD* bdd, const int layer, MultiObjectiveStats* stats);
    
    // Filter layer based on dominance / set packing
    static void filter_dominance_setpacking(BDD* bdd, const int layer, MultiObjectiveStats* stats);
    
    // Filter layer based on dominance / set covering
    static void filter_dominance_setcovering(BDD* bdd, const int layer, MultiObjectiveStats* stats);

    // Filter layer based on dominance during approximation / knapsack
    static void filter_dominance_knapsack_approx(BDD* bdd, const int layer);
    
    // Filter layer based on node completion
    static void filter_completion(BDD* bdd, const int layer);    

    // Find pareto frontier from top-down approach - MDD version
	static ParetoFrontier* pareto_frontier_topdown(MDD* bdd, MultiObjectiveStats* stats);

    // Find pareto frontier using dynamic layer cutset
    static ParetoFrontier* pareto_frontier_dynamic_layer_cutset(MDD* mdd, MultiObjectiveStats* stats);
};



#endif 
