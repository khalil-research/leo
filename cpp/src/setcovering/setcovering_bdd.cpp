#include "setcovering_bdd.hpp"
#include "../bdd_util.hpp"

//
// Preprocess 
//
void SetCoveringBDDConstructor::preprocess() {
	cout << "\tPreprocessing for BDD compilation..." << endl;

	// Alternate representation of variables per constraint
	vector<State> vars_in_cons(inst->n_cons);
	for (int i = 0; i < inst->n_cons; ++i) {
		vars_in_cons[i].resize(inst->n_vars, false);
		for (vector<int>::iterator it = inst->vars_cons[i].begin(); 
				it != inst->vars_cons[i].end(); ++it) 
		{
			vars_in_cons[i][*it] = true;
		}
	}

	// check which clauses are eliminated by each constraints in each layer/variable
	// (assumes variables are ordered lexicographicaly in the BDD)

	clauses_var_cons.resize(inst->n_vars);
	cons_needs_checking = new bool*[inst->n_vars];

	for (int j = 0; j < inst->n_vars; ++j) {

		// eliminate variable from constraints it belongs to
		for (vector<int>::iterator it = inst->cons_var[j].begin(); 
				it != inst->cons_var[j].end(); ++it) 
		{
			// eliminate variables from constraints
			vars_in_cons[*it][j] = false;
		}

		clauses_var_cons[j].resize(inst->n_cons);
		cons_needs_checking[j] = new bool[inst->n_cons];
		for (int i = 0; i < inst->n_cons; ++i) {
			clauses_var_cons[j][i].resize(inst->n_cons, true);
			for (int k = 0; k < inst->n_cons; ++k) {
				if (k != i && vars_in_cons[i].is_subset_of(vars_in_cons[k])) {
					clauses_var_cons[j][i][k] = false;
				}
			}
			cons_needs_checking[j][i] = (clauses_var_cons[j][i].count() < inst->n_cons);
			//cout << "Var=" << j << " - Cons=" << i << " --> " << clauses_var_cons[j][i];
			//cout << " - " << cons_needs_checking[j][i] << endl;
		}
	}




	// Masks for state setting
	mask_set_zero.resize(inst->n_vars);
	mask_set_one.resize(inst->n_vars);

	for (int j = 0; j < inst->n_vars; ++j) {
		//mask_set_zero[j] = clauses_var[j];
		//mask_set_one[j] = clauses_var[j];
		mask_set_zero[j].resize(inst->n_cons, true);
		mask_set_one[j].resize(inst->n_cons, true);

		for (vector<int>::iterator it = inst->cons_var[j].begin(); 
				it != inst->cons_var[j].end(); ++it) 
		{
			mask_set_one[j][*it] = false;
		}
	}

	// cout << "Masks:" << endl;
	// for (int j = 0; j < inst->n_vars; ++j) {
	//   cout << "\tj=" << j << ": " << mask_set_zero[j] << " " << mask_set_one[j] << endl;
	// }

	// Compute last variable for each constraint
	last_cons.resize(inst->n_vars);
	for (int j = 0; j < inst->n_vars; ++j) {
		last_cons[j].resize(inst->n_cons, false);
	}

	for (int i = 0; i < inst->n_cons; ++i) {
		sort(inst->vars_cons[i].begin(), inst->vars_cons[i].end());
		last_cons[inst->vars_cons[i].back()][i] = true;
	}

	// cout << "Last vars:" << endl;
	// for (int i = 0; i < inst->n_cons; ++i) {
	//   cout << "\ti=" << i << ": " << last_var[i] << endl;
	// }

	cout << "\tdone." << endl;
}


//
// Generate exact BDD
//
BDD* SetCoveringBDDConstructor::generate_exact() {

	cout << "\nCreating exact Set Covering BDD..." << endl;

	// preprocess for equivalence testing
	preprocess();

	// IndepSet BDD
	BDD* bdd = new BDD(inst->n_vars+1);

	// State maps
	int iter = 0;
	int next = 1;

	// create root node
	Node* root_node = bdd->add_node(0);
	states[iter].clear();

	// state of root node
	root_node->setState.resize( inst->n_cons, true );
	states[iter][&root_node->setState] = root_node;

	StateNodeMap::iterator it;
	int var;    
	var_layer.resize(inst->n_vars+1);

	for (int l = 1; l <= inst->n_vars; ++l) {   
		states[next].clear();

		// select next vertex
		var = l-1;
		var_layer[l-1] = var;

		//cout << "\tLayer " << l << " - var=" << var << " - size=" << states[iter].size() << endl;

		BOOST_FOREACH(StateNodeMap::value_type i, states[iter]) {
			const State& parent = *(i.first);
			Node* node = i.second;

			// zero arc
			if (!last_cons[var].intersects(*i.first)) {
				State state = parent;
				// remove absorption clauses
				for (int c = i.first->find_first(); c != state_end; c = i.first->find_next(c)) {
					if (cons_needs_checking[var][c]) {
						state &= clauses_var_cons[var][c];
					}
				}
				state &= mask_set_zero[var];
				//cout << "\tst = " << state << endl;
				it = states[next].find(&state);
				if (it == states[next].end()) {
					Node* new_node = bdd->add_node(l);
					new_node->setState = state;
					states[next][&new_node->setState] = new_node;
					node->add_out_arc(new_node, false, 0);
				} else {
					node->add_out_arc(it->second, false, 0);
				}
			}

			// one arc
			State state = parent & mask_set_one[var];
			//cout << "\tst = " << state << endl;
			it = states[next].find(&state);
			if (it == states[next].end()) {
				Node* new_node = bdd->add_node(l);
				new_node->setState = state;
				states[next][&new_node->setState] = new_node;
				node->add_out_arc(new_node, true, 0);
			} else {
				node->add_out_arc(it->second, true, 0);
			}
		}

		iter = !iter;
		next = !next;
	}

	BDDAlg::remove_dangling_nodes(bdd);
	bdd->repair_node_indices();

	cout << "\tdone." << endl << endl;
	//bdd->print();

	return bdd;
}
