#include "knapsack_bdd.hpp"
#include "../bdd_util.hpp"

using namespace boost;

//
// Create exact BDD
//
BDD *KnapsackBDDConstructor::generate_exact()
{

	// cout << "Generating exact Knapsack BDD...\n";

	// Knapsack BDD
	BDD *bdd = new BDD(inst->n_vars + 1);

	// State maps
	int iter = 0;
	int next = 1;

	// create root node
	Node *root_node = bdd->add_node(0);
	states[iter].clear();
	states[iter][0] = root_node;
	root_node->intState = 0;

	// create terminal node
	Node *terminal_node = bdd->add_node(inst->n_vars);

	for (int l = 0; l < inst->n_vars; ++l)
	{
		states[next].clear();
		BOOST_FOREACH (StateNodeMap::value_type i, states[iter])
		{
			int weight = i.first;
			Node *node = i.second;

			if (l < inst->n_vars - 1)
			{

				// zero arc
				StateNodeMap::iterator it = states[next].find(weight);
				if (it == states[next].end())
				{
					Node *new_node = bdd->add_node(l + 1);
					states[next][weight] = new_node;
					new_node->intState = weight;
					node->add_out_arc(new_node, false, 0);
				}
				else
				{
					node->add_out_arc(it->second, false, 0);
				}

				// one arc
				weight += inst->coeffs[l];
				if (weight <= inst->rhs)
				{
					StateNodeMap::iterator it = states[next].find(weight);
					if (it == states[next].end())
					{
						Node *new_node = bdd->add_node(l + 1);
						states[next][weight] = new_node;
						new_node->intState = weight;
						node->add_out_arc(new_node, true, inst->obj_coeffs[l]);
					}
					else
					{
						node->add_out_arc(it->second, true, inst->obj_coeffs[l]);
					}
				}
			}
			else
			{
				// if last layer, just add arcs to the terminal node

				// zero arc
				node->add_out_arc(terminal_node, false, 0);
				// one arc
				if (weight + inst->coeffs[l] <= inst->rhs)
				{
					node->add_out_arc(terminal_node, true, inst->obj_coeffs[l]);
				}
			}
		}

		// invert iter and next
		next = !next;
		iter = !iter;
	}

	// reduce BDD
	// cout << "\twidth = " << bdd->get_width() << endl;
	// cout << "\tremoving dangling nodes..." << endl;
	BDDAlg::remove_dangling_nodes(bdd);
	bdd->repair_node_indices();
	// cout << "\twidth = " << bdd->get_width() << endl;
	return bdd;
};

KnapsackBDDConstructor::~KnapsackBDDConstructor()
{
	delete inst;
}