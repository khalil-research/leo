#include "indepset_bdd.hpp"
#include "../bdd_util.hpp"

using namespace boost;


//
// Create BDD
//
BDD* IndepSetBDDConstructor::generate_exact(bool new_order_provided) {
	cout << "\nCreating IndepSet BDD..." << endl;

	// IndepSet BDD
	BDD* bdd = new BDD(inst->graph->n_vertices+1);

	// State maps
	int iter = 0;
	int next = 1;

	// initialize internal structures for variable ordering
	var_layer.resize(inst->graph->n_vertices);
	active_vertices.resize(inst->graph->n_vertices);
	for (int v = 0; v < inst->graph->n_vertices; ++v) {
		in_state_counter[v] = 1;
		active_vertices[v] = v;
	}

	// create root node
	Node* root_node = bdd->add_node(0);

	// initialize states
	states[iter].clear();

	// state of root node
	root_node->setState.resize( inst->graph->n_vertices, true );
	states[iter][&root_node->setState] = root_node;

	StateNodeMap::iterator it;
	int vertex;    

	for (int l = 1; l <= inst->graph->n_vertices; ++l) {   
		states[next].clear();

		// select next vertex
		if (new_order_provided){
			// Use the default order if a new_order was provided as an input
			vertex = l-1;
		}
		else{
			vertex = choose_next_vertex_min_size_next_layer(states[iter]);
		}				
		var_layer[l-1] = vertex;

		//cout << "\tLayer " << l << " - vertex=" << vertex << " - size=" << states[iter].size() << endl;

		BOOST_FOREACH(StateNodeMap::value_type i, states[iter]) {
			State& state = *(i.first);
			Node* node = i.second;
			bool was_set = state[vertex];

			// zero arc
			state.set(vertex, false);
			it = states[next].find(&state);
			if (it == states[next].end()) {
				Node* new_node = bdd->add_node(l);
				new_node->setState = state; 
				states[next][&new_node->setState] = new_node;
				node->add_out_arc(new_node, false, 0);
			} else {
				node->add_out_arc(it->second, false, 0);
			}

			// one arc
			if (was_set) {
				state &= inst->adj_mask_compl[vertex];
				it = states[next].find(&state);
				if (it == states[next].end()) {
					Node* new_node = bdd->add_node(l);
					new_node->setState = state;
					states[next][&new_node->setState] = new_node;					
					node->add_out_arc(new_node, true, objs[0][l-1]);
				} else {
					node->add_out_arc(it->second, true, objs[0][l-1]);
				}
			}
		}

		// invert iter and next
		next = !next;
		iter = !iter;
	}

	//cout << "\tdone." << endl << endl;
	return bdd;
};


//
// Create BDD for trees
//
BDD* IndepSetBDDConstructor::generate_exact_tree() {
	//cout << "\nCreating IndepSet BDD..." << endl;

	// IndepSet BDD
	BDD* bdd = new BDD(inst->graph->n_vertices+1);

	// State maps
	int iter = 0;
	int next = 1;

	// create variable ordering
	var_layer.resize(inst->graph->n_vertices);

	CutVertexDecomposition cutv(inst);
	for (int i = 0; i < inst->graph->n_vertices; ++i) {
		var_layer[i] = cutv.vertex_in_layer(bdd, i);
		//cout << "i=" << i << " --> " << var_layer[i] << endl;
	}

	//exit(1);


	active_vertices.resize(inst->graph->n_vertices);
	for (int v = 0; v < inst->graph->n_vertices; ++v) {
		//in_state_counter[v] = 1;
		active_vertices[v] = v;
	}

	// create root node
	Node* root_node = bdd->add_node(0);
	
	// reset states
	states[iter].clear();

	// state of root node	
	root_node->setState.resize( inst->graph->n_vertices, true );
	states[iter][&root_node->setState] = root_node;

	StateNodeMap::iterator it;
	int vertex;    

	for (int l = 1; l <= inst->graph->n_vertices; ++l) {   
		states[next].clear();

		// select next vertex
		//vertex = choose_next_vertex_min_size_next_layer(states[iter]);
		//vertex = l-1;
		//var_layer[l-1] = vertex;
		vertex = var_layer[l-1];

		//cout << "\tLayer " << l << " - vertex=" << vertex << " - size=" << states[iter].size() << endl;

		BOOST_FOREACH(StateNodeMap::value_type i, states[iter]) {
			State state = *(i.first);
			Node* node = i.second;
			bool was_set = state[vertex];

			// zero arc
			state.set(vertex, false);
			it = states[next].find(&state);
			if (it == states[next].end()) {
				Node* new_node = bdd->add_node(l);
				new_node->setState = state;
				states[next][&new_node->setState] = new_node;
				node->add_out_arc(new_node, false, 0);
			} else {
				node->add_out_arc(it->second, false, 0);
			}

			// one arc
			if (was_set) {
				state &= inst->adj_mask_compl[vertex];
				it = states[next].find(&state);
				if (it == states[next].end()) {
					Node* new_node = bdd->add_node(l);
					new_node->setState = state;
					states[next][&new_node->setState] = new_node;
					node->add_out_arc(new_node, true, objs[0][l-1]);
				} else {
					node->add_out_arc(it->second, true, objs[0][l-1]);
				}
			}
		}

		// invert iter and next
		next = !next;
		iter = !iter;
	}

	//cout << "\tdone." << endl << endl;

	return bdd;
};


//
// Create components 
//
void CutVertexDecomposition::identify_components(vector< vector<int> > &comps, 
		vector<bool> &is_in_graph) {

	// initialize union/find
	for (int i = 0; i < inst->graph->n_vertices; ++i) {
		component[i] = i;
	}

	// check components for each vertex included in the graph
	for (int i = 0; i < inst->graph->n_vertices; ++i) {
		if (is_in_graph[i]) {
			for (int j = i+1; j < inst->graph->n_vertices; ++j) {
				if (is_in_graph[j] && inst->graph->is_adj(i,j)) {
					union_f(i,j);
				}
			}
		}
	}


	comps.clear();
	for (int i = 0; i < inst->graph->n_vertices; ++i) {
		component_map[i] = -1;
	}

	int num_comps = 0;
	for (int i = 0; i < inst->graph->n_vertices; ++i) {
		if (is_in_graph[i]) {
			int comp = find(i);
			if (component_map[comp] == -1) {
				component_map[comp] = num_comps++;	
				comps.push_back(vector<int>());
			}
			comps[component_map[comp]].push_back(i);
		}
	}
	//cout << "num comps = " << comps.size() << endl;


	// vector<int> label(inst->graph->n_vertices, -1);
	// int num_comps = -1;

	// vector<int> stack;

	// vector<bool> visited(inst->graph->n_vertices, false);
	// for( int i = 0; i < inst->graph->n_vertices; i++ ) {

	//   if( is_in_graph[i] && !visited[i]) {

	//     num_comps++;
	//     stack.push_back(i);

	//     while( !stack.empty() ) {

	// 	int v = stack.back();
	// 	stack.pop_back();

	// 	label[v] = num_comps;
	// 	visited[v] = true;

	// 	for( int w = 0; w < inst->graph->n_vertices; w++ ) {
	// 	  if( w == v ) continue;
	// 	  if( is_in_graph[w] && inst->graph->is_adj(v,w) && !visited[w]) {
	// 	    stack.push_back(w);
	// 	  }
	// 	}
	//     }    
	//   }
	// }

	// comps.clear();
	// comps.resize(num_comps+1);

	// for( int v = 0; v < inst->graph->n_vertices; v++ ) {
	//   if( label[v] != -1 ) {
	//     comps[label[v]].push_back(v);
	//   }
	// }
}


//
// Find orderings relative to a particular subgraph 
//
vector<int> CutVertexDecomposition::find_ordering(vector<bool> is_in_graph) {

	int size = 0;
	for( int i = 0; i < inst->graph->n_vertices; i++ ) {
		size += (is_in_graph[i] ? 1 : 0 );
	}

	// find vertex with all components less than half the size of the graph
	vector< vector<int> > comps;
	for( int i = 0; i < inst->graph->n_vertices; i++ ) {
		if( is_in_graph[i] ) {

			// try removing vertex
			is_in_graph[i] = false;
			identify_components(comps, is_in_graph);

			//cout << "components when removing vertex " << i << endl;

			bool all_valid = true;
			for( int j = 0; j < (int)comps.size() && all_valid; j++ ) {
				all_valid = ((int)comps[j].size() <= size/2);
				//cout << "\t" << comps[j].size() << endl;
			}

			//cout << i << " -- " << all_valid << endl;

			if( all_valid ) {

				vector<int> ordering;

				// compose ordering for each component separately
				vector<bool> is_in_graph_new(inst->graph->n_vertices, false);
				for( int c = 0; c < (int)comps.size(); c++ ) {

					//cout << "*** Component " << c << endl;

					for( int v = 0; v < (int)comps[c].size(); v++ ) {
						is_in_graph_new[comps[c][v]] = true;
					}

					vector<int> order_bck = find_ordering(is_in_graph_new);

					for( int v = 0; v < (int)comps[c].size(); v++ ) {
						is_in_graph_new[comps[c][v]] = false;
						ordering.push_back(order_bck[v]);
					}
				}
				ordering.push_back(i);
				return ordering;	
			}

			// put vertex back again
			is_in_graph[i] = true;
		}
	}

	cout << "Error: graph is not a tree (it might general or a forest)" << endl;
	exit(1);
	return( vector<int>(1,-1) );
}


//
// Construct final ordering
//
void CutVertexDecomposition::construct_ordering() {

	vector<bool> is_in_graph(inst->graph->n_vertices, true);
	vector<int> ordering = find_ordering(is_in_graph);

	//cout << endl;
	// for( int i = 0; i < (int)ordering.size(); i++ ) {
	//   cout << ordering[i] << " ";
	// }
	// cout << endl;
	//cout << "ordering size: " << ordering.size() << endl;

	for( int i = 0; i < (int)ordering.size(); i++ ) {
		v_in_layer[i] = ordering[i];
	}
}
