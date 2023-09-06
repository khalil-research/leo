// ----------------------------------------------------------
// Pareto Set utils
// ----------------------------------------------------------

#ifndef PARETO_UTIL_HPP_
#define PARETO_UTIL_HPP_

#include <algorithm>
#include <iostream>
#include <fstream>
#include <limits>

using namespace std;

//
// Solution representation
//
struct Solution
{
	vector<int> x;	 // solution
	vector<int> obj; // objectives

	// Constructors
	Solution() {}
	Solution(vector<int> &_x, vector<int> &_obj)
		: x(_x), obj(_obj) {}

	// Print solution
	void print(ostream &out)
	{
		// out << "(";
		for (size_t i = 0; i < x.size(); ++i)
		{
			// if (i > 0)
			// out << ", ";
			out << x[i] << " ";
		}
		// out << ") --> ";
		print_objective(out);
	}

	// Print only objective
	void print_objective(ostream &out)
	{
		for (size_t i = 0; i < obj.size(); ++i)
		{
			if (i > 0)
				out << " ";
			out << obj[i];
		}
	}

	// Return if solution dominates other
	bool dominates(Solution &sol)
	{
		assert(obj.size() == sol.obj.size());
		bool sol_dominates = true;
		for (size_t i = 0; i < obj.size() && sol_dominates; ++i)
		{
			sol_dominates = (obj[i] >= sol.obj[i]);
		}
		return sol_dominates;
	}
};

//
// Solution list
//
typedef list<Solution> SolutionList;

//
// Solution comparator
//
struct SolutionComparator
{
	bool operator()(const Solution &solA, const Solution &solB)
	{
		for (size_t i = 0; i < solA.obj.size(); ++i)
		{
			if (solA.x[i] != solB.x[i])
			{
				return (solA.obj[i] > solB.obj[i]);
			}
		}
		return (solA.x[0] > solB.x[0]);
	}
};

//
// Objective comparator
//
struct ObjectiveComparator
{
	bool operator()(const Solution &solA, const Solution &solB)
	{
		for (int p = 0; p < solA.obj.size(); ++p)
		{
			if (solA.obj[p] != solB.obj[p])
			{
				return solA.obj[p] > solB.obj[p];
			}
		}
		return (solA.x[0] > solB.x[0]);
	}
};

//
// Pareto set
//
struct ParetoSet
{
	SolutionList sols;	// Set of solutions
	const int num_objs; // Number of objectives

	// Constructor
	ParetoSet(const int _num_objs) : num_objs(_num_objs)
	{
	}

	// Print all objectives into file
	void print_objectives(string filename)
	{
		SolutionComparator solcomp;
		sols.sort(solcomp);
		ofstream out(filename.c_str());
		for (SolutionList::iterator it = sols.begin(); it != sols.end(); ++it)
		{
			it->print_objective(out);
			out << endl;
		}
		out.close();
	}

	// Print all objectives into file
	void print_objectives()
	{
		SolutionComparator solcomp;
		sols.sort(solcomp);
		for (SolutionList::iterator it = sols.begin(); it != sols.end(); ++it)
		{
			it->print_objective(cout);
			cout << endl;
		}
	}

	void print_sols()
	{
		SolutionComparator solcomp;
		sols.sort(solcomp);
		for (SolutionList::iterator it = sols.begin(); it != sols.end(); ++it)
		{
			it->print(cout);
			cout << endl;
		}
	}

	void print_sols(string filename)
	{
		SolutionComparator solcomp;
		sols.sort(solcomp);
		ofstream out(filename.c_str(), ios_base::app);
		for (SolutionList::iterator it = sols.begin(); it != sols.end(); ++it)
		{
			it->print(out);
			out << endl;
		}
		out.close();
	}

	vector<vector<int>> get_x_sols()
	{
		vector<vector<int>> x_sols;
		for (SolutionList::iterator it = sols.begin(); it != sols.end(); ++it)
		{
			x_sols.push_back(it->x);
		}
		return x_sols;
	}

	vector<vector<int>> get_z_sols()
	{
		vector<vector<int>> z_sols;
		for (SolutionList::iterator it = sols.begin(); it != sols.end(); ++it)
		{
			z_sols.push_back(it->obj);
		}
		return z_sols;
	}

	//
	// Add solution
	//
	void add(Solution &sol)
	{
		assert(sol.obj.size() == num_objs);
		bool dominates;
		bool dominated;
		for (SolutionList::iterator it = sols.begin(); it != sols.end();)
		{
			dominates = true; // if sol. dominates iterate
			dominated = true; // if sol. is dominated by iterate
			for (int i = 0; i < num_objs && (dominates || dominated); ++i)
			{
				dominates &= (sol.obj[i] >= it->obj[i]);
				dominated &= (sol.obj[i] <= it->obj[i]);
			}

			if (dominates)
			{
				// solution dominates iterate
				it = sols.erase(it);
			}
			else if (dominated)
			{
				// solution is dominated by iterate
				return;
			}
			else
			{
				++it;
			}
		}
		// add solution to list
		sols.insert(sols.end(), sol);
	}

	//
	// Add solution, considering a new primal value to be appended at 'x'
	// and its objective function coefficients. Solutions at or after
	// position 'pos' of the list are ignored
	//
	size_t add(Solution &sol, int val, vector<int> &obj, SolutionList::iterator &pos)
	{
		assert(sol.obj.size() == num_objs);
		// bool active = (sol.obj[0]+obj[0] == 10171);
		// if (active) {
		//   cout << "** [";
		//   for (int i = 0; i < num_objs; ++i) {
		// 	cout << " " << (sol.obj[i]+obj[i]);
		//   }
		//   cout << " ]" << endl;
		// }

		size_t num_comparisons = 0;
		bool dominates;
		bool dominated;
		for (SolutionList::iterator it = sols.begin(); it != pos;)
		{
			num_comparisons += 1;
			dominates = true; // if sol. dominates iterate
			dominated = true; // if sol. is dominated by iterate
			for (int i = 0; i < num_objs && (dominates || dominated); ++i)
			{
				dominates &= (sol.obj[i] + obj[i] >= it->obj[i]);
				dominated &= (sol.obj[i] + obj[i] <= it->obj[i]);
			}
			// if (active) {
			// 	cout << "\t" << dominates << " :: " << dominated << endl;
			// 	cout << "\t\t";
			// 	it->print_objective(cout);
			// 	cout << endl;
			// }

			if (dominates)
			{
				// solution dominates iterate
				it = sols.erase(it);
			}
			else if (dominated)
			{
				// solution is dominated by iterate
				return num_comparisons;
			}
			else
			{
				// just move to next iterate
				++it;
			}
		}
		// add solution to list
		Solution new_sol(sol.x, sol.obj);
		new_sol.x.push_back(val);
		for (int i = 0; i < num_objs; ++i)
		{
			new_sol.obj[i] += obj[i];
		}
		sols.insert(sols.end(), new_sol);

		return num_comparisons;
	}

	//
	// Add solution, considering a new primal value to be appended at 'x'
	// and its objective function coefficients. Solutions at or after
	// position 'pos' of the list are ignored
	//
	void add(Solution &sol, SolutionList::iterator &pos)
	{
		assert(sol.obj.size() == num_objs);
		// bool active = (sol.obj[0]+obj[0] == 10171);
		// if (active) {
		//   cout << "** [";
		//   for (int i = 0; i < num_objs; ++i) {
		// 	cout << " " << (sol.obj[i]+obj[i]);
		//   }
		//   cout << " ]" << endl;
		// }

		bool dominates;
		bool dominated;
		for (SolutionList::iterator it = sols.begin(); it != pos;)
		{
			dominates = true; // if sol. dominates iterate
			dominated = true; // if sol. is dominated by iterate
			for (int i = 0; i < num_objs && (dominates || dominated); ++i)
			{
				dominates &= (sol.obj[i] >= it->obj[i]);
				dominated &= (sol.obj[i] <= it->obj[i]);
			}
			// if (active) {
			// 	cout << "\t" << dominates << " :: " << dominated << endl;
			// 	cout << "\t\t";
			// 	it->print_objective(cout);
			// 	cout << endl;
			// }

			if (dominates)
			{
				// solution dominates iterate
				it = sols.erase(it);
			}
			else if (dominated)
			{
				// solution is dominated by iterate
				return;
			}
			else
			{
				// just move to next iterate
				++it;
			}
		}
		// add solution to list
		sols.insert(sols.end(), sol);
	}

	//
	// Merge with another pareto set list, considering a new solution primal value
	// that is appended at the end of 'x' and its objective function coefficient
	//
	size_t merge(ParetoSet &set, int val, vector<int> &obj)
	{
		size_t num_comparisons = 0;
		if (set.sols.size() == 0)
		{
			return 0;
		}

		// add artificial solution to avoid rechecking dominance between elements in the
		// set to be merged
		Solution sol;
		SolutionList::iterator pos = sols.insert(sols.end(), sol);
		for (SolutionList::iterator it = set.sols.begin(); it != set.sols.end(); ++it)
		{
			num_comparisons += add((*it), val, obj, pos);
		}

		// remove artificial solution
		sols.erase(pos);
		return num_comparisons;
	}

	//
	// Merge with another pareto set list
	//
	void merge(ParetoSet &set)
	{
		if (set.sols.size() == 0)
		{
			return;
		}
		// add artificial solution to avoid rechecking dominance between elements in the
		// set to be merged
		Solution sol;
		SolutionList::iterator pos = sols.insert(sols.end(), sol);
		for (SolutionList::iterator it = set.sols.begin(); it != set.sols.end(); ++it)
		{
			add((*it), pos);
		}

		// remove artificial solution
		sols.erase(pos);
	}

	//
	// Merge with another pareto set list, checking for dominance
	//
	void merge(ParetoSet *set)
	{
		for (SolutionList::iterator it = set->sols.begin(); it != set->sols.end(); ++it)
		{
			add((*it));
		}
	}

	//
	// Just copy content from another pareto set, replacing current one
	//
	void copy(ParetoSet &set)
	{
		assert(num_objs == set.num_objs);
		sols = set.sols;
	}

	//
	// Clear
	//
	void clear()
	{
		sols.clear();
	}

	//
	// Returns true if empty
	//
	bool empty()
	{
		return sols.empty();
	}

	//
	// Check if pareto set completely dominates another pareto set
	//
	bool dominates(ParetoSet *pset)
	{
		if (sols.empty())
		{
			return false;
		}
		bool dominates;
		for (SolutionList::iterator it = pset->sols.begin(); it != pset->sols.end(); ++it)
		{
			dominates = false;
			for (SolutionList::iterator s = sols.begin(); s != sols.end() && !dominates; ++s)
			{
				dominates = s->dominates(*it);
			}
			if (!dominates)
			{
				return false;
			}
		}
		return true;
	}

	//
	// Check validity of solutions
	//
	void check_validity()
	{
		// check dominance
		for (SolutionList::iterator itA = sols.begin(); itA != sols.end(); ++itA)
		{
			for (SolutionList::iterator itB = sols.begin(); itB != sols.end(); ++itB)
			{
				if (itA == itB)
					continue;
				bool dominates = true;
				for (int t = 0; t < num_objs && dominates; ++t)
				{
					dominates = (itA->obj[t] >= itB->obj[t]);
				}
				if (dominates)
				{
					cout << "Error: ";
					cout << "(";
					itA->print_objective(cout);
					cout << ")";
					cout << " dominates ";
					cout << "(";
					itB->print_objective(cout);
					cout << ")" << endl;
					exit(1);
				}
			}
		}
	}
};

#endif
