#include <boost/unordered_map.hpp>
#include <boost/foreach.hpp>
#include "bdd_util.hpp"

//
// Reduce BDD
//
void BDDAlg::reduce(BDD *bdd)
{

	// State for equivalence test
	typedef pair<int, int> state;

	// State map
	typedef boost::unordered_map<state, Node *> node_map;
	node_map states;
	state new_state;
	vector<Node *> inactive;

	// Merge equivalent nodes
	for (int l = bdd->num_layers - 2; l >= 0; --l)
	{
		states.clear();

		while (!bdd->layers[l].empty())
		{
			// remove node from layer and add to map
			Node *node = bdd->layers[l].back();
			bdd->layers[l].pop_back();

			if (!node->active)
			{
				// if node is not active, just save it to add later
				inactive.push_back(node);
			}
			else if (node->zero_arc != NULL || node->one_arc != NULL)
			{

				// create new state
				new_state.first = (node->zero_arc != NULL ? node->zero_arc->index : -1);
				new_state.second = (node->one_arc != NULL ? node->one_arc->index : -1);

				// check if states exists
				node_map::iterator it = states.find(new_state);
				if (it == states.end())
				{
					states[new_state] = node;
				}
				else
				{
					// state exists: merge node with existing one
					Node *original = it->second;
					for (vector<Node *>::iterator it = node->zero_prev.begin();
						 it != node->zero_prev.end(); ++it)
					{
						(*it)->zero_arc = original;
					}
					for (vector<Node *>::iterator it = node->one_prev.begin();
						 it != node->one_prev.end(); ++it)
					{
						(*it)->one_arc = original;
					}
					delete node;
				}
			}
			else
			{
				// node can be removed
				remove_node(node);
			}
		}

		// add nodes back to layer, fixing indices
		BOOST_FOREACH (node_map::value_type it, states)
		{
			bdd->layers[l].push_back(it.second);
			bdd->layers[l].back()->index = bdd->layers[l].size() - 1;
		}

		// add inactive nodes
		for (vector<Node *>::iterator it = inactive.begin(); it != inactive.end(); ++it)
		{
			bdd->layers[l].push_back(*it);
		}
	}

	// fix indices and incoming arcs
	for (int l = bdd->num_layers - 1; l >= 0; --l)
	{
		for (size_t i = 0; i < bdd->layers[l].size(); ++i)
		{
			Node *node = bdd->layers[l][i];
			node->index = i;

			// clear incoming
			node->zero_prev.clear();
			node->one_prev.clear();

			// add list to previous
			if (node->zero_arc != NULL)
			{
				node->zero_arc->zero_prev.push_back(node);
			}
			if (node->one_arc != NULL)
			{
				node->one_arc->one_prev.push_back(node);
			}
		}
	}
}

//
// Create new reduced BDD contaning the paths from the input BDD with the given length
//
BDD *BDDAlg::value_extraction(BDD *bdd, int length)
{

	cout << "Value extraction..." << endl;

	BDD *valBDD = new BDD(bdd->num_layers);

	// Pair for partial equivalence test in value BDD
	typedef pair<int, int> state;

	// State maps
	typedef boost::unordered_map<state, Node *> node_map;
	node_map states[2];
	int iter = 0;
	int next = 1;

	// Create root node in new BDD
	Node *root_node = valBDD->add_node(0);
	states[iter][state(0, 0)] = root_node;

	// Create value BDD
	for (int l = 1; l < bdd->num_layers; ++l)
	{

		cout << "\tLayer " << l << " - num_states = " << states[iter].size() << endl;
		states[next].clear();

		BOOST_FOREACH (node_map::value_type it, states[iter])
		{

			int weight = it.first.first;
			Node *origNode = bdd->layers[l - 1][it.first.second];
			Node *valNode = it.second;

			// zero arc
			if (origNode->zero_arc != NULL && (l != bdd->num_layers - 1 || weight == length))
			{
				// create node
				state new_state(weight, origNode->zero_arc->index);
				node_map::iterator it = states[next].find(new_state);
				if (it == states[next].end())
				{
					Node *new_node = valBDD->add_node(l);
					states[next][new_state] = new_node;
					valNode->add_out_arc(new_node, false, origNode->zero_arc_length);
				}
				else
				{
					valNode->add_out_arc(it->second, false, origNode->zero_arc_length);
				}
			}

			// one arc
			if (origNode->one_arc != NULL && (l != bdd->num_layers - 1 || weight + origNode->one_arc_length == length))
			{
				weight += origNode->one_arc_length;
				state new_state(weight, origNode->one_arc->index);
				node_map::iterator it = states[next].find(new_state);
				if (it == states[next].end())
				{
					Node *new_node = valBDD->add_node(l);
					states[next][new_state] = new_node;
					valNode->add_out_arc(new_node, true, origNode->one_arc_length);
				}
				else
				{
					valNode->add_out_arc(it->second, true, origNode->one_arc_length);
				}
			}
		}

		// invert iter and next
		next = !next;
		iter = !iter;
	}

	// if value BDD does not have a terminal, return empty BDD
	if (valBDD->layers.back().empty())
	{
		delete valBDD;
		return NULL;
	}

	// reduce and return BDD
	cout << "\n\treducing BDD..." << endl;
	reduce(valBDD);

	cout << "\tdone!" << endl
		 << endl;
	return valBDD;
}

//
// Create new reduced BDD contaning the paths from the input BDD with the given length
// Use this if arc lengths are nonnegative
//
BDD *BDDAlg::value_extraction_nonnegative(BDD *bdd, int length)
{

	cout << "Value extraction..." << endl;

	BDD *valBDD = new BDD(bdd->num_layers);

	// compute maximum length from layer l to terminal
	vector<int> max_length(bdd->num_layers, 0);
	for (int l = 0; l < bdd->num_layers; ++l)
	{
		for (size_t i = 0; i < bdd->layers[l].size(); ++i)
		{
			if (max_length[l] < bdd->layers[l][i]->one_arc_length)
			{
				max_length[l] = bdd->layers[l][i]->one_arc_length;
			}
			if (max_length[l] < bdd->layers[l][i]->zero_arc_length)
			{
				max_length[l] = bdd->layers[l][i]->zero_arc_length;
			}
		}
	}
	for (int l = bdd->num_layers - 2; l >= 0; --l)
	{
		max_length[l] += max_length[l + 1];
		// cout << "l=" << l << " - max length=" << max_length[l] << endl;
	}

	// Pair for partial equivalence test in value BDD
	typedef pair<int, int> state;

	// State maps
	typedef boost::unordered_map<state, Node *> node_map;
	node_map states[2];
	int iter = 0;
	int next = 1;

	// Create root node in new BDD
	Node *root_node = valBDD->add_node(0);
	states[iter][state(0, 0)] = root_node;

	// Create value BDD
	for (int l = 1; l < bdd->num_layers; ++l)
	{

		cout << "\tLayer " << l << " - num_states = " << states[iter].size() << endl;

		states[next].clear();

		BOOST_FOREACH (node_map::value_type it, states[iter])
		{

			int weight = it.first.first;
			Node *origNode = bdd->layers[l - 1][it.first.second];
			Node *valNode = it.second;

			// zero arc
			if (origNode->zero_arc != NULL && (l != bdd->num_layers - 1 || weight == length))
			{
				// create node
				state new_state(weight, origNode->zero_arc->index);
				node_map::iterator it = states[next].find(new_state);
				if (it == states[next].end())
				{
					Node *new_node = valBDD->add_node(l);
					states[next][new_state] = new_node;
					valNode->add_out_arc(new_node, false, origNode->zero_arc_length);
				}
				else
				{
					valNode->add_out_arc(it->second, false, origNode->zero_arc_length);
				}
			}

			// one arc
			if (origNode->one_arc != NULL && (l != bdd->num_layers - 1 || weight + origNode->one_arc_length == length))
			{
				weight += origNode->one_arc_length;
				if (weight <= length && weight + max_length[l] >= length)
				{
					state new_state(weight, origNode->one_arc->index);
					node_map::iterator it = states[next].find(new_state);
					if (it == states[next].end())
					{
						Node *new_node = valBDD->add_node(l);
						states[next][new_state] = new_node;
						valNode->add_out_arc(new_node, true, origNode->one_arc_length);
					}
					else
					{
						valNode->add_out_arc(it->second, true, origNode->one_arc_length);
					}
				}
			}
		}

		// invert iter and next
		next = !next;
		iter = !iter;
	}

	// if value BDD does not have a terminal, return empty BDD
	if (valBDD->layers.back().empty())
	{
		delete valBDD;
		return NULL;
	}

	// reduce and return BDD
	cout << "\treducing BDD..." << endl;
	reduce(valBDD);

	cout << "\tdone!" << endl
		 << endl;
	return valBDD;
}

//
// Create reduced value BDD where all paths have the same length
// Assume lengths of layers are nonnegative and fixed per layer
//
BDD *BDDAlg::create_value_BDD(int length, vector<int> &arc_length)
{

	cout << "\nCreating value BDD, length = " << length << endl;

	BDD *bdd = new BDD(arc_length.size() + 1);

	// max length from layer l (non-inclusive) to last layer
	vector<int> max_length(bdd->num_layers, 0);
	for (int l = bdd->num_layers - 2; l >= 0; --l)
	{
		max_length[l] += (max_length[l + 1] + arc_length[l]);
	}

	// State maps
	typedef boost::unordered_map<int, Node *> node_map;
	node_map states[2];
	int iter = 0;
	int next = 1;

	// create root node
	Node *root_node = bdd->add_node(0);
	states[iter][0] = root_node;

	for (int l = 1; l < bdd->num_layers; ++l)
	{
		//cout << "Layer " << l << endl;
		states[next].clear();
		BOOST_FOREACH (node_map::value_type i, states[iter])
		{
			int weight = i.first;
			Node *node = i.second;

			// zero arc
			node_map::iterator it = states[next].find(weight);
			if (it == states[next].end())
			{
				Node *new_node = bdd->add_node(l);
				new_node->length = weight;
				states[next][weight] = new_node;
				node->add_out_arc(new_node, false, 0);
			}
			else
			{
				node->add_out_arc(it->second, false, 0);
			}

			// one arc
			weight += arc_length[l - 1];
			if (weight <= length && weight + max_length[l] >= length)
			{
				node_map::iterator it = states[next].find(weight);
				if (it == states[next].end())
				{
					Node *new_node = bdd->add_node(l);
					new_node->length = weight;
					states[next][weight] = new_node;
					node->add_out_arc(new_node, true, arc_length[l - 1]);
				}
				else
				{
					node->add_out_arc(it->second, true, arc_length[l - 1]);
				}
			}
		}

		// invert iter and next
		next = !next;
		iter = !iter;
	}

	// remove invalid terminal nodes
	//cout << "removing terminals..." << endl;
	vector<Node *> &terminals = bdd->layers.back();
	size_t node_count = 0;
	while (node_count < terminals.size())
	{
		int i = node_count;
		if (terminals[i]->length != length)
		{
			Node *node = terminals[i];
			terminals[i] = terminals.back();
			terminals[i]->index = i;
			terminals.pop_back();
			remove_node(node);
		}
		else
		{
			node_count++;
		}
	}
	//cout << "\tsize = " << terminals.size() << endl;
	//cout << endl;

	// merge terminal nodes
	int prev = bdd->num_layers - 2;
	Node *terminal = bdd->layers[prev + 1][0];
	terminal->one_prev.clear();
	terminal->zero_prev.clear();

	for (int i = 0; i < (int)bdd->layers[prev].size(); ++i)
	{
		if (bdd->layers[prev][i]->zero_arc != NULL)
		{
			terminal->add_in_arc(bdd->layers[prev][i], false,
								 bdd->layers[prev][i]->zero_arc_length);
		}
		if (bdd->layers[prev][i]->one_arc != NULL)
		{
			terminal->add_in_arc(bdd->layers[prev][i], true,
								 bdd->layers[prev][i]->one_arc_length);
		}
	}
	for (int i = 1; i < (int)bdd->layers[prev + 1].size(); ++i)
	{
		delete bdd->layers[prev + 1][i];
	}
	bdd->layers[prev + 1].resize(1);

	cout << "\tremoving dangling nodes..." << endl;
	remove_dangling_nodes(bdd);
	cout << "\twidth = " << bdd->get_width() << endl;

	cout << "\treducing..." << endl;
	reduce(bdd);

	cout << "\t";
	cout << "sp = " << BDDAlg::shortest_path(bdd);
	cout << " - lp = " << BDDAlg::longest_path(bdd);
	cout << " - width = " << bdd->get_width();
	cout << endl;

	//bdd->print();

	cout << "\tdone." << endl
		 << endl;

	return bdd;
}

//
// Intersect BDDs
//
BDD *BDDAlg::intersect(BDD *bddA, BDD *bddB)
{

	cout << "\nCreating BDD intersection..." << endl;

	assert(bddA->num_layers == bddB->num_layers);

	// make sure indices are correct in both BDDs
	// TODO: just assert this?
	for (int l = 0; l < bddA->num_layers; ++l)
	{
		for (size_t i = 0; i < bddA->layers[l].size(); ++i)
		{
			bddA->layers[l][i]->index = i;
		}
		for (size_t i = 0; i < bddB->layers[l].size(); ++i)
		{
			bddB->layers[l][i]->index = i;
		}
	}

	// Pair for partial equivalence test in value BDD
	typedef pair<int, int> state;

	// State maps
	typedef boost::unordered_map<state, Node *> node_map;
	node_map states[2];
	int iter = 0;
	int next = 1;

	// Create root node in new BDD
	BDD *bdd = new BDD(bddA->num_layers);
	Node *root_node = bdd->add_node(0);
	states[iter][state(0, 0)] = root_node;

	// Create value BDD
	for (int l = 1; l < bdd->num_layers; ++l)
	{

		//cout << "\tLayer " << l << " - num_states = " << states[iter].size() << endl;
		states[next].clear();

		BOOST_FOREACH (node_map::value_type it, states[iter])
		{
			Node *nodeA = bddA->layers[l - 1][it.first.first];
			Node *nodeB = bddB->layers[l - 1][it.first.second];
			Node *intNode = it.second;

			// zero arc
			if (nodeA->zero_arc != NULL && nodeB->zero_arc != NULL)
			{
				state new_state(nodeA->zero_arc->index, nodeB->zero_arc->index);
				node_map::iterator it = states[next].find(new_state);
				if (it == states[next].end())
				{
					Node *new_node = bdd->add_node(l);
					states[next][new_state] = new_node;
					intNode->add_out_arc(new_node, false, 0);
				}
				else
				{
					intNode->add_out_arc(it->second, false, 0);
				}
			}

			// one arc
			if (nodeA->one_arc != NULL && nodeB->one_arc != NULL)
			{
				state new_state(nodeA->one_arc->index, nodeB->one_arc->index);
				node_map::iterator it = states[next].find(new_state);
				if (it == states[next].end())
				{
					Node *new_node = bdd->add_node(l);
					states[next][new_state] = new_node;
					intNode->add_out_arc(new_node, true, nodeA->one_arc_length);
				}
				else
				{
					intNode->add_out_arc(it->second, true, nodeA->one_arc_length);
				}
			}
		}

		// invert iter and next
		next = !next;
		iter = !iter;
	}

	if (bdd->layers.back().empty())
	{
		delete bdd;
		return NULL;
	}

	cout << "\tremoving dangling nodes..." << endl;
	remove_dangling_nodes(bdd);

	cout << "\t";
	cout << "sp = " << BDDAlg::shortest_path(bdd);
	cout << " - lp = " << BDDAlg::longest_path(bdd);
	cout << " - width = " << bdd->get_width();
	cout << endl;

	//cout << "\treducing..." << endl;
	//reduce(bdd);
	//cout << "\twidth after reduction = " << bdd->get_width() << endl;

	//bdd->print();d

	cout << "\tdone." << endl
		 << endl;
	return bdd;
}

//
// Compute nadir points of BDD given 'n' objective functions
// Assume zero-arc lenghts are zero and one-arc lenghts are fixed per layer
//
vector<vector<int>> BDDAlg::extreme_points(BDD *bdd, const vector<vector<int>> &obj_coeffs, vector<vector<int>> &solutions)
{

	cout << "\nNadir and Ideal points...\n";

	int num_objs = obj_coeffs.size();

	// Pseudonadir point (index 0) and ideal point (index 1)
	vector<vector<int>> points;
	vector<int> nadir(num_objs, 0);
	vector<int> ideal(num_objs, 0);
	points.push_back(nadir);
	points.push_back(ideal);

	for (int o = 0; o < num_objs; ++o)
	{
		vector<int> sol = longest_path_multi_obj(bdd, o, obj_coeffs);
		solutions.push_back(sol);
		if (!o)
		{
			for (int k = 0; k < num_objs; k++)
			{
				points[0][k] = sol[k];
				points[1][k] = sol[k];
			}
		}
		else
		{
			for (int k = 0; k < num_objs; k++)
			{
				points[0][k] = min(points[0][k], sol[k]);
				points[1][k] = max(points[1][k], sol[k]);
			}
		}
	}

	return points;
}

//
// Compute longest path for one specific objective function
// Assume zero-arc lenghts are zero and one-arc lenghts are fixed per layer
//
inline vector<int> BDDAlg::longest_path_multi_obj(BDD *bdd, int obj, const vector<vector<int>> &obj_coeffs)
{

	//  cout << "\nLongest path for objective " << obj << "...\n";

	int num_objs = obj_coeffs.size();

	// Initialize lenghts for each objective function
	for (int o = 0; o < num_objs; ++o)
	{
		bdd->layers[0][0]->multi_length.push_back(0);
	}

	for (int l = 1; l < bdd->num_layers; ++l)
	{
		//cout << "Layer " << l << endl;
		for (size_t i = 0; i < bdd->layers[l].size(); ++i)
		{
			Node *node = bdd->layers[l][i];
			assert(node != NULL);
			assert(node->zero_prev.size() > 0 || node->one_prev.size() > 0);

			// initialization of node length
			for (int o = 0; o < num_objs; ++o)
			{
				if (node->zero_prev.size() == 0)
				{
					node->multi_length[o] = node->one_prev[0]->multi_length[o] + obj_coeffs[o][l - 1];
				}
				else
				{
					node->multi_length[o] = node->zero_prev[0]->multi_length[o];
				}
			}

			// compute maximum path length w.r.t. function 'obj'
			for (size_t j = 0; j < node->zero_prev.size(); ++j)
			{
				int k = node->zero_prev[j]->multi_length[obj];
				if (node->multi_length[obj] < k)
				{
					for (int o = 0; o < num_objs; ++o)
					{
						node->multi_length[o] = node->zero_prev[j]->multi_length[o];
					}
				}
			}

			for (size_t j = 0; j < node->one_prev.size(); ++j)
			{
				int k = node->one_prev[j]->multi_length[obj] + obj_coeffs[obj][l - 1];
				if (node->multi_length[obj] < k)
				{
					for (int o = 0; o < num_objs; ++o)
					{
						node->multi_length[o] = node->one_prev[j]->multi_length[o] + obj_coeffs[o][l - 1];
					}
				}
			}
		}
	}

	// Extract upper bounds on longest path for function 'obj'
	compute_ub_longest_path(bdd, obj, obj_coeffs);

	vector<int> optimum(num_objs);
	for (int o = 0; o < num_objs; o++)
		optimum[o] = bdd->layers[bdd->num_layers - 1][0]->multi_length[o];
	return optimum;
}

//
// Compute longest path for one specific objective function
// Assume zero-arc lenghts are zero and one-arc lenghts are fixed per layer
//
inline void BDDAlg::compute_ub_longest_path(BDD *bdd, int obj, const vector<vector<int>> &obj_coeffs)
{
	int num_objs = obj_coeffs.size();

	for (int l = bdd->num_layers - 1; l >= 0; --l)
	{
		//cout << "Layer " << l << endl;
		for (size_t i = 0; i < bdd->layers[l].size(); ++i)
		{
			Node *node = bdd->layers[l][i];

			// Debug
			/*
      cout << "Node " << node->layer << "-" << node->index << "\n"; 
      cout << "\tNode upper bound: " << node->ub_longest_path[obj] << "\n";
      vector<int> tmp = longest_path_multi_obj_from_node(bdd, obj, obj_coeffs, node, l);
      cout << "\tNode computed bound: " << tmp[obj] << "\n";
      */
			assert(node != NULL);
			if (l)
				assert(node->zero_prev.size() > 0 || node->one_prev.size() > 0);

			// upper bound from one-arcs
			for (size_t j = 0; j < node->one_prev.size(); ++j)
			{
				int k = node->ub_longest_path[obj] + obj_coeffs[obj][l - 1];
				if (node->one_prev[j]->ub_longest_path[obj] < k)
				{
					node->one_prev[j]->ub_longest_path[obj] = k;
				}
			}

			// upper bound from zero-arcs
			for (size_t j = 0; j < node->zero_prev.size(); ++j)
			{
				int k = node->ub_longest_path[obj];
				if (node->zero_prev[j]->ub_longest_path[obj] < k)
				{
					node->zero_prev[j]->ub_longest_path[obj] = k;
				}
			}
		}
	}

	return;
}

/*

//
// Compute nadir points of BDD given 'n' objective functions
// Assume zero-arc lenghts are zero and one-arc lenghts are fixed per layer
//
vector<vector<int> >  BDDAlg::extreme_points_from_node(BDD* bdd, const vector< vector<int> >& obj_coeffs,
						       Node* node, int layer) {

  //  cout << "\nNadir and Ideal points...\n";

  int num_objs = obj_coeffs.size();

  // Pseudonadir point (index 0) and ideal point (index 1)
  vector<vector<int> > points;
  vector<int> nadir(num_objs,0);
  vector<int> ideal(num_objs,0);
  points.push_back(nadir);
  points.push_back(ideal);
  
  
  for (int o = 0; o < num_objs; ++o) {
    vector<int> sol = longest_path_multi_obj_from_node(bdd,o,obj_coeffs,node,layer);
    if (!o) {
      for (int k = 0; k < num_objs; k++) {
	points[0][k] = sol[k];
	points[1][k] = sol[k];
      }
    } else {
      for (int k = 0; k < num_objs; k++) {
	points[0][k] = min(points[0][k],sol[k]);
	points[1][k] = max(points[1][k],sol[k]);
      }
    }
  }
  
  return points;
}



//
// Compute longest path for one specific objective function from a given node
// Assume zero-arc lenghts are zero and one-arc lenghts are fixed per layer
//
inline vector<int> BDDAlg::longest_path_multi_obj_from_node(BDD* bdd, int obj, const vector< vector<int> >& obj_coeffs,
							    Node* node, int layer) {

  //  cout << "\nLongest path for objective " << obj << "...\n";
  int num_objs = obj_coeffs.size();

  // Initialize lenghts with very negative value for all nodes in 'layer' 
  for (size_t i = 0; i < bdd->layers[layer].size(); ++i) {
    for (int o = 0; o < num_objs; ++o) {
      bdd->layers[layer][i]->multi_length[o] = -1e-9;
    } 
  }
  // Initialize lengths for origin node
  for (int o = 0; o < num_objs; ++o) {
    node->multi_length[o] = 0;
  } 
  
	for (int l = layer+1; l < bdd->num_layers; ++l) {
		//cout << "Layer " << l << endl;
		for (size_t i = 0; i < bdd->layers[l].size(); ++i) {
			Node* node = bdd->layers[l][i];
			assert(node != NULL);
			assert(node->zero_prev.size() > 0 || node->one_prev.size() > 0);

			// initialization of node length
			for (int o = 0; o < num_objs; ++o) {
			  if (node->zero_prev.size() == 0) {
			    node->multi_length[o] = node->one_prev[0]->multi_length[o] + obj_coeffs[o][l-1];
			  } else {
			    node->multi_length[o] = node->zero_prev[0]->multi_length[o];
			  }
			}

			// compute maximum path length w.r.t. function 'obj' 
			for (size_t j = 0; j < node->zero_prev.size(); ++j) {
			  int k = node->zero_prev[j]->multi_length[obj];
			  if (node->multi_length[obj] < k) {
			    for (int o = 0; o < num_objs; ++o) {
			      node->multi_length[o] = node->zero_prev[j]->multi_length[o];
			    }
			  }
			}

			for (size_t j = 0; j < node->one_prev.size(); ++j) {
			  int k = node->one_prev[j]->multi_length[obj] + obj_coeffs[obj][l-1];
			  if (node->multi_length[obj] < k) {
			    for (int o = 0; o < num_objs; ++o) {
			      node->multi_length[o] = node->one_prev[j]->multi_length[o] + obj_coeffs[o][l-1];
			    }
			  }
			}
		}
	}

	vector<int> optimum(num_objs);
	for (int o = 0; o < num_objs; o++)
	  optimum[o] = bdd->layers[bdd->num_layers-1][0]->multi_length[o];
	return optimum;
}
*/

//
// Compute pareto-set solution of BDD given 'n' objective functions
// Assume zero-arc lenghts are zero and one-arc lenghts are fixed per layer
//
// ParetoSet *BDDAlg::pareto_set(BDD *bdd, const vector<vector<int>> &obj_coeffs)
MultiobjResult *BDDAlg::pareto_set(BDD *bdd, const vector<vector<int>> &obj_coeffs)
{
	// cout << "\nComputing Pareto Set...\n";

	int width = bdd->get_width();
	int num_objs = obj_coeffs.size();

	// pre-allocate vectors for efficient sets
	// cout << "\tallocating..." << endl;
	vector<ParetoSet *> sets[2];
	for (int o = 0; o < 2; ++o)
	{
		sets[o] = vector<ParetoSet *>(width, NULL);
		for (int w = 0; w < width; ++w)
		{
			sets[o][w] = new ParetoSet(num_objs);
		}
	}

	MultiobjResult* mo_result = new MultiobjResult(bdd->num_layers);

	int bef = 0, cur = 1; // 'before' and 'current' pareto set indices
	vector<int> shift_zero(num_objs, 0);
	vector<int> shift_one(num_objs, 0);
	double avg_size = 0;

	// root node
	vector<int> x;

	Solution rootSolution(x, shift_zero);
	sets[bef][0]->add(rootSolution);
	// Record the number of pareto solutions at layer 0
	mo_result->num_pareto_sol[0] = (unsigned long int) sets[bef][0]->sols.size();

	for (int l = 1; l < bdd->num_layers; ++l)
	{
		// cout << "\tLayer " << l << " - size = " << bdd->layers[l].size();
		// cout << " - avg-pareto-size = " << avg_size << endl;

		// set shift one
		for (int o = 0; o < num_objs; ++o)
		{
			shift_one[o] = obj_coeffs[o][l - 1];
		}

		avg_size = 0;
		for (vector<Node *>::iterator it = bdd->layers[l].begin(); it != bdd->layers[l].end(); ++it)
		{

			int id = (*it)->index;
			// clear efficient set of this node
			sets[cur][id]->clear();

			// add zero arc prev
			for (vector<Node *>::iterator prev = (*it)->zero_prev.begin();
				 prev != (*it)->zero_prev.end(); ++prev)
			{
				sets[cur][id]->merge(*(sets[bef][(*prev)->index]), 0, shift_zero);
			}

			// add one arc prev
			for (vector<Node *>::iterator prev = (*it)->one_prev.begin();
				 prev != (*it)->one_prev.end(); ++prev)
			{
				sets[cur][id]->merge(*(sets[bef][(*prev)->index]), 1, shift_one);
			}

			avg_size += sets[cur][id]->sols.size();
		}
		// Record the number of pareto solutions at layer l
		mo_result->num_pareto_sol[l] = (unsigned long int) avg_size;

		// Find average number of pareto solutions at layer l
		avg_size /= bdd->layers[l].size();
		
		// swap sets
		bef = !bef;
		cur = !cur;
	}

	// erase sets
	// cout << "\tdeallocating..." << endl;
	delete sets[cur][0];
	for (int i = 1; i < width; ++i)
	{
		delete sets[bef][i];
		delete sets[cur][i];
	}
	// cout << "\tdone." << endl;
	// cout << endl;

	mo_result->pareto_set = sets[bef][0];

	// return sets[bef][0];
	return mo_result;
}

//
// Delay states (for pareto set)
//
inline void delay_states(const int max_states, ParetoSet *original, ParetoSet *delayed)
{
	assert(max_states > 0);
	assert(original != NULL);
	assert(delayed != NULL);

	while (original->sols.size() > max_states)
	{
		delayed->add(original->sols.back());
		original->sols.pop_back();
	}
}

//
// Undelay states (for pareto set)
//
inline void undelay_states(const int max_states, ParetoSet *original, ParetoSet *delayed)
{
	assert(max_states > 0);
	assert(original != NULL);
	assert(delayed != NULL);

	while (original->sols.size() < max_states && !delayed->empty())
	{
		original->add(delayed->sols.back());
		delayed->sols.pop_back();
	}
}

//
// Compute pareto-set solution of BDD given 'n' objective functions, with delayed states
// Assume zero-arc lenghts are zero and one-arc lenghts are fixed per layer
//
ParetoSet *BDDAlg::pareto_set_delayed(BDD *bdd, const vector<vector<int>> &obj_coeffs,
									  const int max_delayed_states, const int delay_heuristic)
{

	int width = bdd->get_width();
	int num_objs = obj_coeffs.size();

	// pre-allocate vectors for efficient sets

	cout << "\tallocating..." << endl;

	// sets for node computation
	vector<ParetoSet *> sets[2];
	for (int o = 0; o < 2; ++o)
	{
		sets[o] = vector<ParetoSet *>(width, NULL);
		for (int w = 0; w < width; ++w)
		{
			sets[o][w] = new ParetoSet(num_objs);
		}
	}

	// delay sets for each node
	vector<vector<ParetoSet *>> delayed_states(bdd->num_layers);
	for (int l = 0; l < bdd->num_layers; ++l)
	{
		delayed_states[l].resize(bdd->layers[l].size());
		for (size_t w = 0; w < bdd->layers[l].size(); ++w)
		{
			delayed_states[l][w] = new ParetoSet(num_objs);
		}
	}

	// resulting pareto set
	ParetoSet *frontier = new ParetoSet(num_objs);

	vector<int> shift_zero(num_objs, 0);
	vector<int> shift_one(num_objs, 0);
	double avg_size = 0;

	// root node
	vector<int> x;

	// if there are any delayed states to explore
	bool exists_delayed = true; // root node is delayed
	int iteration = 0;

	int bef, cur; // 'before' and 'current' pareto set indices

	cout << endl;

	while (true)
	{

		// initialize
		bef = 0;
		cur = 1;
		int initial_layer;

		if (iteration == 0)
		{
			Solution rootSolution(x, shift_zero);
			sets[0][0]->add(rootSolution);
			initial_layer = 0;
		}
		else
		{
			for (int w = 0; w < width; ++w)
			{
				sets[0][w]->clear();
				sets[1][w]->clear();
			}

			// search for first layer from bottom with delayed states
			bool undelayed = false;
			for (initial_layer = bdd->num_layers - 1; initial_layer >= 0 && !undelayed; --initial_layer)
			{
				for (vector<Node *>::iterator it = bdd->layers[initial_layer].begin();
					 it != bdd->layers[initial_layer].end(); ++it)
				{
					int id = (*it)->index;
					if (!delayed_states[initial_layer][id]->empty())
					{
						//cout << "\tfound at: " << initial_layer << " - node=" << id << " - size=" << delayed_states[initial_layer][id]->sols.size() << endl;
						undelay_states(max_delayed_states, sets[0][id], delayed_states[initial_layer][id]);
						undelayed = true;
						//cout << "\tafter: " << initial_layer << " - node=" << id << " - size=" << delayed_states[initial_layer][id]->sols.size() << endl;
					}
				}
			}
			//cout << "\nInitial layer: " << initial_layer << endl;

			// if no node was undelayed, search is done
			if (!undelayed)
			{
				break;
			}
			initial_layer++; // adjust position
		}

		//if (iteration > 10) {
		//exit(1);
		//}

		for (int l = initial_layer + 1; l < bdd->num_layers; ++l)
		{
			//cout << "\tLayer " << l << " - size = " << bdd->layers[l].size();
			//cout << " - avg-pareto-size = " << avg_size << endl;

			// set shift one
			for (int o = 0; o < num_objs; ++o)
			{
				shift_one[o] = obj_coeffs[o][l - 1];
			}

			avg_size = 0;
			for (vector<Node *>::iterator it = bdd->layers[l].begin(); it != bdd->layers[l].end(); ++it)
			{

				int id = (*it)->index;
				// clear efficient set of this node
				sets[cur][id]->clear();

				// add zero arc prev
				for (vector<Node *>::iterator prev = (*it)->zero_prev.begin();
					 prev != (*it)->zero_prev.end(); ++prev)
				{
					if (!sets[bef][(*prev)->index]->empty())
						sets[cur][id]->merge(*(sets[bef][(*prev)->index]), 0, shift_zero);
				}

				// add one arc prev
				for (vector<Node *>::iterator prev = (*it)->one_prev.begin();
					 prev != (*it)->one_prev.end(); ++prev)
				{
					if (!sets[bef][(*prev)->index]->empty())
						sets[cur][id]->merge(*(sets[bef][(*prev)->index]), 1, shift_one);
				}

				// Delay considerations
				if (l < bdd->num_layers - 1 && sets[cur][id]->sols.size() > max_delayed_states)
				{
					// if size of the node is greater than max states, delay some states
					delay_states(max_delayed_states, sets[cur][id], delayed_states[l][id]);
				}

				avg_size += sets[cur][id]->sols.size();
			}
			avg_size /= bdd->layers[l].size();

			// swap sets
			bef = !bef;
			cur = !cur;
		}

		// add resulting set to frontier
		frontier->merge(sets[bef][0]);

		cout << "Iteration " << iteration << " - frontier size = " << frontier->sols.size();
		cout << " - initial layer = " << initial_layer << endl;
		iteration++;
	}

	cout << endl;

	// erase sets
	cout << "\tdeallocating..." << endl;
	for (int i = 0; i < width; ++i)
	{
		delete sets[bef][i];
		delete sets[cur][i];
	}
	cout << "\tdone." << endl;
	cout << endl;

	return frontier;
}

//
// Create flow model from BDD
//

// void BDDAlg::flow_model(BDD* bdd, IloModel &model, IloBoolVarArray& x, vector<int>& var_layer) {
// 	bdd->repair_node_indices();
// 	IloEnv env = model.getEnv();

// 	// create one expression for node
// 	vector< vector<IloExpr> >flow_nodes(bdd->num_layers);
// 	for (int l = 0; l < bdd->num_layers; ++l) {
// 		for (size_t i = 0; i < bdd->layers[l].size(); ++i) {
// 			flow_nodes[l].push_back(IloExpr(env));
// 		}
// 	}

// 	// create one expression for variable
// 	vector<IloExpr> var_expr(x.getSize());
// 	for (int v = 0; v < x.getSize(); ++v) {
// 		var_expr[v] = IloExpr(env);
// 	}

// 	// root node
// 	flow_nodes[0][0] += 1;

// 	// terminal node
// 	flow_nodes[bdd->num_layers-1][0] += -1;

// 	// add expression constraints
// 	for (int l = 0; l < bdd->num_layers-1; ++l) {
// 		for (size_t u = 0; u < bdd->layers[l].size(); ++u) {
// 			Node* node = bdd->layers[l][u];

// 			if (node->zero_arc != NULL) {
// 				IloNumVar zero_arc(env, 0, 1);
// 				//IloIntVar zero_arc(env, 0, 1);
// 				flow_nodes[l][u] -= zero_arc;
// 				flow_nodes[l+1][node->zero_arc->index] += zero_arc;
// 			}

// 			if (node->one_arc != NULL) {
// 				IloNumVar one_arc(env, 0, 1);
// 				//IloIntVar one_arc(env, 0, 1);
// 				flow_nodes[l][u] -= one_arc;
// 				flow_nodes[l+1][node->one_arc->index] += one_arc;

// 				var_expr[var_layer[l]] += one_arc;
// 			}
// 		}
// 	}

// 	// add balance constraints
// 	for (int l = 0; l < bdd->num_layers-1; ++l) {
// 		for (size_t u = 0; u < bdd->layers[l].size(); ++u) {
// 			model.add( flow_nodes[l][u] == 0 );
// 		}
// 	}

// 	// add variable constraints
// 	for (int v = 0; v < x.getSize(); ++v) {
// 		model.add( x[v] == var_expr[v] );
// 	}

// }

// //
// // Create flow model from BDD. Amount of flow is controlled by a variable f
// //
// void BDDAlg::flow_model_f(BDD* bdd, IloModel &model, IloBoolVarArray& x, vector<int>& var_layer,
// 		IloNumVar& z) {
// 	bdd->repair_node_indices();
// 	IloEnv env = model.getEnv();

// 	char varname[256];

// 	// create one expression for node
// 	vector< vector<IloExpr> >flow_nodes(bdd->num_layers);
// 	for (int l = 0; l < bdd->num_layers; ++l) {
// 		for (size_t i = 0; i < bdd->layers[l].size(); ++i) {
// 			flow_nodes[l].push_back(IloExpr(env));
// 		}
// 	}

// 	// create one expression for variable
// 	vector<IloExpr> var_expr(x.getSize());
// 	for (int v = 0; v < x.getSize(); ++v) {
// 		var_expr[v] = IloExpr(env);
// 	}

// 	// root node
// 	flow_nodes[0][0] += z;

// 	// terminal node
// 	flow_nodes[bdd->num_layers-1][0] -= z;

// 	// negative terminal node
// 	//IloExpr negflow(env);
// 	//negflow -= (1-z);

// 	// add expression constraints
// 	for (int l = 0; l < bdd->num_layers-1; ++l) {
// 		for (size_t u = 0; u < bdd->layers[l].size(); ++u) {
// 			Node* node = bdd->layers[l][u];

// 			if (node->zero_arc != NULL) {
// 				sprintf(varname, "f[%d][%d][%d]", l, node->index, 0);
// 				IloNumVar zero_arc(env, 0, 1, varname);
// 				//IloIntVar zero_arc(env, 0, 1);
// 				flow_nodes[l][u] -= zero_arc;
// 				flow_nodes[l+1][node->zero_arc->index] += zero_arc;
// 			} else {
// 				/*sprintf(varname, "f[%d][%d][%d]", l, node->index, 0);
//                 IloNumVar zero_arc(env, 0, 1, varname);
//                 flow_nodes[l][u] -= zero_arc;
//                 negflow += zero_arc;*/
// 			}

// 			if (node->one_arc != NULL) {
// 				sprintf(varname, "f[%d][%d][%d]", l, node->index, 1);
// 				IloNumVar one_arc(env, 0, 1, varname);
// 				//IloIntVar one_arc(env, 0, 1);
// 				flow_nodes[l][u] -= one_arc;
// 				flow_nodes[l+1][node->one_arc->index] += one_arc;

// 				//cout << "l = " << l << " " << bdd->num_layers << " --> " << var_layer.size() << endl;

// 				var_expr[var_layer[l]] += one_arc;
// 			} else {
// 				/*
//                 sprintf(varname, "f[%d][%d][%d]", l, node->index, 1);
//                 IloNumVar one_arc(env, 0, 1, varname);
//                 //IloIntVar one_arc(env, 0, 1);
//                 flow_nodes[l][u] -= one_arc;
//                 negflow += one_arc;
//                 var_expr[var_layer[l]] += one_arc;
// 				 */
// 			}
// 		}
// 	}

// 	// add balance constraints
// 	for (int l = 0; l < bdd->num_layers; ++l) {
// 		for (size_t u = 0; u < bdd->layers[l].size(); ++u) {
// 			model.add( flow_nodes[l][u] == 0 );
// 		}
// 	}
// 	//model.add( negflow == 0 );

// 	// add variable constraints
// 	for (int v = 0; v < x.getSize(); ++v) {
// 		model.add( x[v] >= var_expr[v] );
// 	}

// }
