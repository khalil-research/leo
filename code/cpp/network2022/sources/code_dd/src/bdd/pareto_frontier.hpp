// ----------------------------------------------------------
// Pareto Frontier classes
// ----------------------------------------------------------

#ifndef PARETO_FRONTIER_HPP_
#define PARETO_FRONTIER_HPP_

#define DOMINATED -9999999
#define EPS 0.0001

#include <algorithm>
#include <iostream>
#include <fstream>
#include <limits>
#include <vector>
#include <list>

#include "../util/util.hpp"

using namespace std;


//
// Pareto Frontier struct
//
class ParetoFrontier {
public:
    // (Flat) array of solutions
    vector<ObjType> sols;

    // Add element to set
    void add(ObjType* elem);

    // Merge pareto frontier solutions into existing set
    void merge(const ParetoFrontier& frontier);

    // Merge pareto frontier solutions with shift
    void merge(const ParetoFrontier& frontier, const ObjType* shift);

    // Convolute two nodes from this set to this one
    void convolute(const ParetoFrontier& fA, const ParetoFrontier& fB);

    // Remove pre-set dominated solutions
    void remove_dominated() {
        remove_empty();
    }

    // Get number of solutions
    int get_num_sols() const {
        return sols.size() / NOBJS;
    }

    // Clear pareto frontier
    void clear() {
        sols.resize(0);
    }

    // Print elements in set
    void print() const;

    // Sort array in decreasing order
    void sort_decreasing();

    // Check consistency
    bool check_consistency();

    // Obtain sum of points 
    ObjType get_sum();

    // Check if solution is dominated by any element of this set
    bool is_sol_dominated(const ObjType* sol, const ObjType* shift);

private:
    // Auxiliaries
    ObjType aux[NOBJS];
    ObjType auxB[NOBJS];
    vector<ObjType*> elems;

    // Remove empty elements
    void remove_empty();
};


//
// Pareto frontier manager
//
class ParetoFrontierManager {
    public:
        // Constructor
        ParetoFrontierManager() { }
        ParetoFrontierManager(int size) {
            frontiers.reserve(size);
        }

        // Destructor
        ~ParetoFrontierManager() {
            for (int i = 0; i < frontiers.size(); ++i) {
                delete frontiers[i];
            }
        }

        // Request pareto frontier
        ParetoFrontier* request() {
            if (frontiers.empty()) {
                return new ParetoFrontier;
            }
            ParetoFrontier* f = frontiers.back();
            f->clear();
            frontiers.pop_back();
            return f;
        }

        // Return frontier to allocation
        void deallocate(ParetoFrontier* frontier) {
            frontiers.push_back(frontier);
        }
        
    // Preallocated array set
    vector<ParetoFrontier*> frontiers;
};


//
// Add element to set
//
inline void ParetoFrontier::add(ObjType* elem) {
    bool must_add = true;
    bool dominates;
    bool dominated;
    for (int i = 0; i < sols.size(); i += NOBJS) {
        // check status of foreign solution w.r.t. current frontier solution
        dominates = true;
        dominated = true;
        for (int o = 0; o < NOBJS && (dominates || dominated); ++o) {
            dominates &= (elem[o] >= sols[i+o]);
            dominated &= (elem[o] <= sols[i+o]);
        }
        if (dominated) {
            // if foreign solution is dominated, nothing needs to be done
            return;
        } else if (dominates) {
            // if foreign solution dominates, check if replacement is necessary
            if (must_add) {
                // solution has not been added - just replace current iterate
                std::copy(elem, elem+NOBJS, sols.begin()+i);
                must_add = false;
            } else {
                // if already added, mark array as "to erase"
                sols[i] = DOMINATED;
            }
        }
    }
    // add if still necessary
    if (must_add) {
        sols.insert(sols.end(), elem, elem+NOBJS);
    }
    remove_empty();
}


//
// Check if solution v1 is dominated by v2
//
template <int N>
inline bool AdominatedB(const ObjType* v1, const ObjType* v2) {
    return (v1[N-1] <= v2[N-1]) && AdominatedB<N-1>(v1,v2);
}
template<>
inline bool AdominatedB<0>(const ObjType* v1, const ObjType* v2) {
    return true;
}


//
// Check if solution v1 dominates v2
//
template <int N>
inline bool AdominatesB(const ObjType* v1, const ObjType* v2) {
    return (v1[N-1] >= v2[N-1]) && AdominatesB<N-1>(v1,v2);
}
template<>
inline bool AdominatesB<0>(const ObjType* v1, const ObjType* v2) {
    return true;
}

//
// Merge pareto frontier into existing set
//
inline void ParetoFrontier::merge(const ParetoFrontier& frontier) {
    // last position to check
    int end = sols.size();
    // if current solution set was modified
    bool modified = false;
    // add each solution from frontier set
    bool must_add;
    bool dominates;
    bool dominated;
    for (int j = 0; j < frontier.sols.size(); j += NOBJS) {
        must_add = true; // if solution must be added to set
        for (int i = 0; i < end; i += NOBJS) {
            // check if solution has been removed
            if (sols[i] == DOMINATED) {
                continue;
            }
            // check status of foreign solution w.r.t. current frontier solution
            dominates = true;
            dominated = true;
            for (int o = 0; o < NOBJS && (dominates || dominated); ++o) {
                dominates &= (frontier.sols[j+o] >= sols[i+o]);
                dominated &= (frontier.sols[j+o] <= sols[i+o]);
            }
            if (dominated) {
                // if foreign solution is dominated, just stop loop
                must_add = false;
                break;

            } else if (dominates) {
                // // if foreign solution dominates, replace current solution
                // must_add = false;
                // std::copy(frontier.sols.begin()+j, frontier.sols.begin()+j+NOBJS, &sols[i]);
                // // search for domination in the remaining part of the array
                // for (int k = i+NOBJS; k < end; k += NOBJS) {
                //     if (sols[k] != DOMINATED) {
                //         if (AdominatesB<NOBJS>(&frontier.sols[j], &sols[k])) {
                //            sols[k] = DOMINATED;
                //         }
                //     }
                // }
                // break;

                // if foreign solution dominates, check if replacement is necessary
                if (must_add) {
                    // solution has not been added - just replace current iterate
                    std::copy(frontier.sols.begin()+j, frontier.sols.begin()+j+NOBJS, &sols[i]);
                    must_add = false;
                } else {
                    // if already added, mark array as "to erase"
                    sols[i] = DOMINATED;
                    modified = true;
                }
            }
        }
        // if solution has not been added already, append element to the end
        if (must_add) {
            sols.insert(sols.end(), frontier.sols.begin()+j, frontier.sols.begin()+j+NOBJS);
        }
    }
    if (modified) {
        remove_empty();
    }
}


//
// Merge pareto frontier into existing set considering shift
//
inline void ParetoFrontier::merge(const ParetoFrontier& frontier, const ObjType* shift) {
    // last position to check
    int end = sols.size();
    // if current solution set was modified
    // bool modified = false;
    // add each solution from frontier set
    bool must_add;
    bool dominates;
    bool dominated;
    for (int j = 0; j < frontier.sols.size(); j += NOBJS) {
        // update auxiliary
        for (int o = 0; o < NOBJS; ++o) {
            aux[o] = frontier.sols[j+o] + shift[o];
        }
        must_add = true; // if solution must be added to set
        for (int i = 0; i < end; i += NOBJS) {
            // check if solution has been removed
            if (sols[i] == DOMINATED) {
                continue;
            }
            // check status of foreign solution w.r.t. current frontier solution
            dominates = true;
            dominated = true;
            for (int o = 0; o < NOBJS && (dominates || dominated); ++o) {
                dominates &= (aux[o] >= sols[i+o]);
                dominated &= (aux[o] <= sols[i+o]);
            }
            if (dominated) {
                // if foreign solution is dominated, just stop loop
                must_add = false;
                break;

            } else if (dominates) {            
                // // if foreign solution dominates, replace current solution
                // must_add = false;
                // std::copy(aux, aux+NOBJS, &sols[i]);
                // // search for domination in the remaining part of the array
                // for (int k = i+NOBJS; k < end; k += NOBJS) {
                //     if (sols[k] != DOMINATED) {
                //         if (AdominatesB<NOBJS>(aux, &sols[k])) {
                //            sols[k] = DOMINATED;
                //         }
                //     }
                // }
                // break;

                // if foreign solution dominates, check if replacement is necessary
                if (must_add) {
                    // solution has not been added - just replace current iterate
                    std::copy(aux, aux+NOBJS, &sols[i]);
                    must_add = false;
                } else {
                    // if already added, mark array as "to erase"
                    sols[i] = DOMINATED;
                    //modified = true;
                }
            }
        }
        // if solution has not been added already, append element to the end
        if (must_add) {
            sols.insert(sols.end(), aux, aux+NOBJS);
        }
    }
    //if (modified) {
        remove_empty();
    //}
}


//
// Print elements in set
//
inline void ParetoFrontier::print() const {
    for (int i = 0; i < sols.size(); i += NOBJS) {
        cout << "(";
        for (int o = 0; o < NOBJS-1; ++o) {
            cout << sols[i+o] << ",";
        }
        cout << sols[i+NOBJS-1] << ")";
        cout << endl;
    }
}


//
// Remove empty elements
//
inline void ParetoFrontier::remove_empty() {
    if (sols.empty()) {
        return;
    }
    // find first non-dominated element
    int last = sols.size() - NOBJS;    
    while (last >= 0 && sols[last] == DOMINATED) {
        last -= NOBJS;
    }
    // if there is no such element, all array can be removed
    if (last < 0) {
        sols.resize(0);
        return;
    }
    // otherwise, erase last components
    for (int i = 0; i < last; i += NOBJS) {
        if (sols[i] == DOMINATED) {
            std::copy(sols.begin()+last, sols.begin()+last+NOBJS, sols.begin()+i);
            last -= NOBJS;
            while (sols[last] == DOMINATED) {
                last -= NOBJS;
            }
        }
    }
    assert( last >= 0 );
    sols.resize( last + NOBJS );
}


//
// Convolute two nodes from this set to this one
//
inline void ParetoFrontier::convolute(const ParetoFrontier& fA, const ParetoFrontier& fB) {
    if (fA.sols.size() < fB.sols.size()) {
        for (int j = 0; j < fA.sols.size(); j += NOBJS) {
            std::copy(fA.sols.begin() + j, fA.sols.begin() + j + NOBJS, auxB);
            merge( fB, auxB );
        }
    } else {
        for (int j = 0; j < fB.sols.size(); j += NOBJS) {
            std::copy(fB.sols.begin() + j, fB.sols.begin() + j + NOBJS, auxB);
            merge( fA, auxB );
        }        
    }
}


//
// Auxiliary comparator
//
struct SolComp {
	bool operator()(const ObjType* solA, const ObjType* solB) {
		for (int i = 0; i < NOBJS; ++i) {
			if (solA[i] != solB[i]) {
				return (solA[i] > solB[i]);
			}
		}  
		return (solA[0] > solB[0]);
	}
};


//
// Sort array in decreasing order
//
inline void ParetoFrontier::sort_decreasing() {
    const int num_sols = get_num_sols();
    while (elems.size() < num_sols) {
        elems.push_back( new ObjType[NOBJS] );
    }
    int ct = 0;
    for (int i = 0; i < sols.size(); i += NOBJS) {              
        std::copy( sols.begin()+i, sols.begin()+i+NOBJS, elems[ct++] );
    }
    sort(elems.begin(), elems.begin()+num_sols, SolComp());
    ct = 0;
    for (int i = 0; i < num_sols; ++i) {
        std::copy( elems[i], elems[i]+NOBJS, sols.begin()+ct );
        ct += NOBJS;
    }
}


//
// Check consistency
//
inline bool ParetoFrontier::check_consistency() {
    for (int i = 0; i < sols.size(); i += NOBJS) {
        assert( sols[i] != DOMINATED );
        for (int j = i+NOBJS; j < sols.size(); j += NOBJS) {
            // check status of foreign solution w.r.t. current frontier solution
            bool dominates = true;
            bool dominated = true;
            for (int o = 0; o < NOBJS && (dominates || dominated); ++o) {
                dominates &= (sols[i+o] >= sols[j+o]);
                dominated &= (sols[i+o] <= sols[j+o]);
            }
            assert( !dominates );
            assert( !dominated );
            if (dominates || dominated) {
                return false;
            }
        }
    }
    return true;
}


//
// Obtain sum of points
//
inline ObjType ParetoFrontier::get_sum() {
    ObjType sum = 0;
    for (int i = 0; i < sols.size(); ++i) {
        sum += sols[i];
    }
    return sum;
}


//
// Check if solution is dominated by any element of this set
//
inline bool ParetoFrontier::is_sol_dominated(const ObjType* sol, const ObjType* shift) {
    bool dominated = false;
    for (int i = 0; i < sols.size() && !dominated; i += NOBJS) {
        dominated = AdominatedB<NOBJS>(sol, &sols[i]);
    }
    return dominated;
} 


#endif
