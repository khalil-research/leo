// ----------------------------------------------------------
// BDD Utilities
// ----------------------------------------------------------

// TODO:
// - value extraction for int length
// - optimize value extraction for cases (positive costs, empty BDD while compiling...)

#ifndef BDD_UTIL_HPP_
#define BDD_UTIL_HPP_

#include <algorithm>
#include <iostream>
#include <limits>
// #include <ilcplex/ilocplex.h>

#include "bdd.hpp"
#include "pareto_util.hpp"

using namespace std;


//
// Multiobjective optimization result
//
struct MultiobjResult{
	int num_layers;
	ParetoSet * pareto_set;	
	unsigned long int * num_pareto_sol;	

	MultiobjResult(int _num_layers){
		num_layers = _num_layers;
		pareto_set = NULL;
		num_pareto_sol = new unsigned long int[num_layers];
		// Zero-initialize
		for(int i=0; i < num_layers; i++){
			num_pareto_sol[i] = 0;
		}
	}

	~MultiobjResult(){
		// Free memory
		delete pareto_set;
		delete num_pareto_sol;
	}

	void print_num_pareto_sol(){
		for (int i=0; i<num_layers; i++){
			cout << " " << i << ":" << num_pareto_sol[i];
		}
		cout << endl;
	}
};


//
// BDD Algorithms
//
class BDDAlg
{
public:
	// Compute shortest path
	static int shortest_path(BDD *bdd);

	// Compute longest path
	static int longest_path(BDD *bdd);

	// Compute the longest path less than a particular length
	static int longest_path_less_than(BDD *bdd, int length);

	// Reduce BDD
	static void reduce(BDD *bdd);

	// Remove dangling nodes
	static void remove_dangling_nodes(BDD *bdd);

	// Intersect BDDs
	static BDD *intersect(BDD *bddA, BDD *bddB);

	// Compute pareto-set solution of BDD given 'n' objective functions
	// Assume zero-arc lenghts are zero and one-arc lenghts are fixed per layer
	static MultiobjResult *pareto_set(BDD *bdd, const vector<vector<int>> &obj_coeffs);

	// Compute pareto-set solution of BDD given 'n' objective functions, with delayed states
	// Assume zero-arc lenghts are zero and one-arc lenghts are fixed per layer
	static ParetoSet *pareto_set_delayed(BDD *bdd,
										 const vector<vector<int>> &obj_coeffs,
										 const int max_delayed_states,
										 const int delay_heuristic);

	// Remove BDD node, updating neighbor references
	static void remove_node(Node *node);

	// Create flow model from BDD
	// static void flow_model(BDD *bdd, IloModel &model, IloBoolVarArray &x, vector<int> &var_layer);

	// // Create flow model from BDD. Amount of flow is controlled by a variable f
	// static void flow_model_f(BDD *bdd, IloModel &model, IloBoolVarArray &x, vector<int> &var_layer,
	// 						 IloNumVar &f);

	// Export BDD to file
	static void exportToFile(BDD *bdd, char *filename);

	// Import BDD from file
	static BDD *importFromFile(char *filename);

	// Create value BDD where all paths have the same length
	// Assume lengths of arcs are nonnegative are the one-arc lenghts are fixed per layer
	static BDD *create_value_BDD(int length, vector<int> &layer_length);

	// Create value-extracted BDDB through DD intersection.
	// Assume lengths of arcs are nonnegative are the one-arc lenghts are fixed per layer
	static BDD *value_extraction_intersection(BDD *bdd, int length, vector<int> &layer_length);

	// Create new reduced BDD contaning the paths from the input BDD with a given length
	static BDD *value_extraction(BDD *bdd, int length);

	// Create new reduced BDD contaning the paths from the input BDD with a given length
	// Use this if all lengths are non-negative
	static BDD *value_extraction_nonnegative(BDD *bdd, int length);

	// Compute Nadir and Ideal points for multi-objective program represented by the input BDD
	static vector<vector<int>> extreme_points(BDD *bdd, const vector<vector<int>> &obj_coeffs,
											  vector<vector<int>> &solutions);

	// Compute longest path for multi-objective problem for given objective function
	static vector<int> longest_path_multi_obj(BDD *bdd, int obj, const vector<vector<int>> &obj_coeffs);

	/*
  // Compute Nadir and Ideal points for multi-objective program starting from a given node
  static vector<vector<int> > extreme_points_from_node(BDD* bdd, const vector< vector<int> >& obj_coeffs,
							Node* node, int layer);

  // Compute longest path for multi-objective problem for objective function 'obj' starting from a given node
  static vector<int> longest_path_multi_obj_from_node(BDD* bdd, int obj, const vector< vector<int> >& obj_coeffs,
						      Node* node, int layer);
  */

	// Extract upper bound for all nodes w.r.t. to function 'obj'
	static void compute_ub_longest_path(BDD *bdd, int obj, const vector<vector<int>> &obj_coeffs);
};

// ------------------------------------------------------------------
// Inline implementations
// ------------------------------------------------------------------

//
// Compute shortest path
//
inline int BDDAlg::shortest_path(BDD *bdd)
{
	bdd->layers[0][0]->length = 0;
	for (int l = 1; l < bdd->num_layers; ++l)
	{
		for (size_t i = 0; i < bdd->layers[l].size(); ++i)
		{
			Node *node = bdd->layers[l][i];
			assert(node->zero_prev.size() > 0 || node->one_prev.size() > 0);

			// initialization of node length
			if (node->zero_prev.size() == 0)
			{
				node->length = node->one_prev[0]->length +
							   node->one_prev[0]->one_arc_length;
			}
			else
			{
				node->length = node->zero_prev[0]->length +
							   node->zero_prev[0]->zero_arc_length;
			}

			// compute minimum length to path
			for (size_t j = 0; j < node->zero_prev.size(); ++j)
			{
				node->length = std::min(node->length,
										node->zero_prev[j]->length +
											node->zero_prev[j]->zero_arc_length);
			}

			for (size_t j = 0; j < node->one_prev.size(); ++j)
			{
				node->length = std::min(node->length,
										node->one_prev[j]->length +
											node->one_prev[j]->one_arc_length);
			}
		}
	}
	return bdd->layers[bdd->num_layers - 1][0]->length;
}

//
// Compute longest path
//
inline int BDDAlg::longest_path(BDD *bdd)
{
	bdd->layers[0][0]->length = 0;
	for (int l = 1; l < bdd->num_layers; ++l)
	{
		//cout << "Layer " << l << endl;
		for (size_t i = 0; i < bdd->layers[l].size(); ++i)
		{
			Node *node = bdd->layers[l][i];
			assert(node != NULL);
			assert(node->zero_prev.size() > 0 || node->one_prev.size() > 0);

			// initialization of node length
			if (node->zero_prev.size() == 0)
			{
				node->length = node->one_prev[0]->length +
							   node->one_prev[0]->one_arc_length;
			}
			else
			{
				node->length = node->zero_prev[0]->length +
							   node->zero_prev[0]->zero_arc_length;
			}

			// compute minimum length to path
			for (size_t j = 0; j < node->zero_prev.size(); ++j)
			{
				node->length = std::max(node->length,
										node->zero_prev[j]->length +
											node->zero_prev[j]->zero_arc_length);
			}

			for (size_t j = 0; j < node->one_prev.size(); ++j)
			{
				node->length = std::max(node->length,
										node->one_prev[j]->length +
											node->one_prev[j]->one_arc_length);
			}
		}
	}
	return bdd->layers[bdd->num_layers - 1][0]->length;
}

//
// Compute the longest path less than a particular length
//
inline int BDDAlg::longest_path_less_than(BDD *bdd, int length)
{

	// Since number of hops is the same for all paths, we can shift numbers
	// by a constant to account for negative lenghts
	int shift = 0;
	for (int l = 0; l < bdd->num_layers; ++l)
	{
		for (size_t i = 0; i < bdd->layers[l].size(); ++i)
		{
			if (bdd->layers[l][i]->zero_arc != NULL)
			{
				shift = std::min(shift, bdd->layers[l][i]->zero_arc_length);
			}
			if (bdd->layers[l][i]->one_arc != NULL)
			{
				shift = std::min(shift, bdd->layers[l][i]->one_arc_length);
			}
			// initialize node length
			bdd->layers[l][i]->length = (-1) * std::numeric_limits<int>::max();
		}
	}
	if (shift < 0)
	{
		shift *= -1;
		length += shift * (bdd->num_layers - 1);
	}

	// Compute longest path
	bdd->layers[0][0]->length = 0;
	for (int l = 0; l < bdd->num_layers - 1; ++l)
	{
		for (size_t i = 0; i < bdd->layers[l].size(); ++i)
		{
			Node *node = bdd->layers[l][i];
			// zero arc
			if (node->zero_arc != NULL && (node->length + node->zero_arc_length + shift < length))
			{
				node->zero_arc->length = std::max(node->zero_arc->length,
												  node->length + node->zero_arc_length + shift);
			}
			// one arc
			if (node->one_arc != NULL && (node->length + node->one_arc_length + shift < length))
			{
				node->one_arc->length = std::max(node->one_arc->length,
												 node->length + node->one_arc_length + shift);
			}
		}
	}
	return (bdd->layers[bdd->num_layers - 1][0]->length - shift * (bdd->num_layers - 1));
}

//
// Remove BDD node, updating neighbor references
//
void inline BDDAlg::remove_node(Node *node)
{
	assert(node->active);
	if (node->zero_arc != NULL)
	{
		for (size_t i = 0; i < node->zero_arc->zero_prev.size(); ++i)
		{
			if (node->zero_arc->zero_prev[i] == node)
			{
				node->zero_arc->zero_prev[i] = node->zero_arc->zero_prev.back();
				node->zero_arc->zero_prev.pop_back();
				break;
			}
		}
	}
	if (node->one_arc != NULL)
	{
		for (size_t i = 0; i < node->one_arc->one_prev.size(); ++i)
		{
			if (node->one_arc->one_prev[i] == node)
			{
				node->one_arc->one_prev[i] = node->one_arc->one_prev.back();
				node->one_arc->one_prev.pop_back();
				break;
			}
		}
	}

	for (size_t i = 0; i < node->zero_prev.size(); ++i)
	{
		node->zero_prev[i]->zero_arc = NULL;
	}
	for (size_t i = 0; i < node->one_prev.size(); ++i)
	{
		node->one_prev[i]->one_arc = NULL;
	}

	// TODO
	delete node;
}

//
// Create value-extracted BDDB through DD intersection.
// Assume lengths of layers are fixed per layer
//
inline BDD *BDDAlg::value_extraction_intersection(BDD *bdd, int length,
												  vector<int> &layer_length)
{

	// create value BDD
	BDD *valueBDD = create_value_BDD(length, layer_length);

	// create BDD intersection BDD
	BDD *intersection = intersect(bdd, valueBDD);

	delete valueBDD;
	return intersection;
}

//
// Remove dangling nodes
//
inline void BDDAlg::remove_dangling_nodes(BDD *bdd)
{
	for (int l = bdd->num_layers - 2; l >= 0; --l)
	{
		//cout << "Layer " << l << endl;
		for (size_t i = 0; i < bdd->layers[l].size();)
		{
			Node *node = bdd->layers[l][i];
			if (node->zero_arc == NULL && node->one_arc == NULL)
			{
				bdd->layers[l][i] = bdd->layers[l].back();
				bdd->layers[l][i]->index = i;
				bdd->layers[l].pop_back();
				remove_node(node);
			}
			else
			{
				i++;
			}
		}
	}
}

//
// Export BDD to file
//
inline void BDDAlg::exportToFile(BDD *bdd, char *filename)
{
	bdd->repair_node_indices();
	ofstream out(filename);
	out << bdd->num_layers << endl;
	for (int l = 0; l < bdd->num_layers; ++l)
	{
		out << bdd->layers[l].size() << " ";
	}
	out << endl;
	for (int l = 0; l < bdd->num_layers; ++l)
	{
		for (size_t i = 0; i < bdd->layers[l].size(); ++i)
		{
			Node *node = bdd->layers[l][i];
			out << node->index << " ";
			if (node->zero_arc != NULL)
			{
				out << node->zero_arc->index << " ";
			}
			else
			{
				out << -1 << " ";
			}
			if (node->one_arc != NULL)
			{
				out << node->one_arc->index << " ";
			}
			else
			{
				out << -1 << " ";
			}
			out << endl;
		}
	}
	out.close();
}

//
// Import BDD from file
//
inline BDD *BDDAlg::importFromFile(char *filename)
{
	ifstream bddin(filename);
	int num_layers = -1;
	bddin >> num_layers;
	BDD *bdd = new BDD(num_layers);
	for (int l = 0; l < num_layers; ++l)
	{
		int size = -1;
		bddin >> size;
		for (int i = 0; i < size; ++i)
		{
			bdd->add_node(l);
		}
	}
	int node;
	for (int l = 0; l < num_layers; ++l)
	{
		for (size_t i = 0; i < bdd->layers[l].size(); ++i)
		{
			bddin >> node;
			assert(node == i);
			int zero_arc = -1;
			bddin >> zero_arc;
			if (zero_arc != -1)
			{
				bdd->add_arc(bdd->layers[l][i], bdd->layers[l + 1][zero_arc], false, 0);
			}
			int one_arc = -1;
			bddin >> one_arc;
			if (one_arc != -1)
			{
				bdd->add_arc(bdd->layers[l][i], bdd->layers[l + 1][one_arc], true, 0);
			}
		}
	}
	bddin.close();
	return bdd;
}

#endif /* BDD_UTIL_HPP */
