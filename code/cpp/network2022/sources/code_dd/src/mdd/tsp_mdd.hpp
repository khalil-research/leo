// ----------------------------------------------------------
// MDD Constructor for TSP
// ----------------------------------------------------------

#ifndef TSP_MDD_HPP_
#define TSP_MDD_HPP_

#include "../bdd/pareto_frontier.hpp"
#include "../instances/tsp_instance.hpp"
#include "mdd.hpp"


//
// TSP MDD constructor
//
class MDDTSPConstructor {
    public:
        // Constructor
        MDDTSPConstructor(TSPInstance* _inst);

        // Generate exact 
        MDD* generate_exact();

    private:
        TSPInstance* inst;
};

#endif