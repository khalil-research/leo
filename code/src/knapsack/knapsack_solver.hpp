
#include <ctime>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <vector>
#include <sstream>

#include "../bdd.hpp"
#include "../bdd_util.hpp"
#include "../pareto_util.hpp"
#include "../stats.hpp"

#include "knapsack_instance.hpp"
#include "knapsack_instance_ordered.hpp"
#include "knapsack_bdd.hpp"
#include "knapsack_order.hpp"

using namespace std;

class KnapsackBDDSolver
{
public:
    char *instance_file;
    char *output_file;
    int (&seeds)[5];

    MultiObjKnapsackInstanceOrdered *inst;

    map<string, vector<int>> order_map;

    // Constructor
    KnapsackBDDSolver(char *instance_file, char *output_file, int (&seeds)[5]);
    // Destructor
    ~KnapsackBDDSolver();

    // Converts a vector of int to string
    string vint_to_str(vector<int> order);
    // Generate random and heuristic orderings and stores it in order_map
    void generate_orders();
    // Main loop to create bdds corresponding to different orderings
    // and computing the pareto frontier
    void solve();
};