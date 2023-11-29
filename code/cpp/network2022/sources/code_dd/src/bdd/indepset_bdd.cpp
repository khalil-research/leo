// ----------------------------------------------------------
// Indepset BDD Constructor - Implementation
// ----------------------------------------------------------

#include <deque>

#include "indepset_bdd.hpp"
#include "bdd_alg.hpp"

#include "../util/util.hpp"

using namespace boost;



//
// State allocator
//
struct StateAllocator {
	// State type
	typedef IndepSetBDDConstructor::State State;
	// Allocated states
	deque<State*> alloc_states;
	// Free states
	deque<State*> free_states;
	
	// Request state
	inline State* request() {
		if (free_states.empty()) {
			alloc_states.push_back( new State );
			return alloc_states.back();
		}
		State* st = free_states.back();
		free_states.pop_back();
		return st;
	}

	// Deallocate state
	inline void deallocate(State* state) {
		free_states.push_back(state);
	}

	// Destructor
	~StateAllocator() {
		for (deque<State*>::iterator it = alloc_states.begin(); it != alloc_states.end(); ++it) {
			delete *it;
		}
	}
};


//
// Create BDD
//
BDD* IndepSetBDDConstructor::generate_exact() {
	//cout << "\nCreating IndepSet BDD..." << endl;

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

	// initialize allocator
	StateAllocator alloc;

	// initialize state map 
	states[iter].clear();

	// create root node
	Node* root_node = bdd->add_node(0);
	State* root_state = alloc.request();
	root_state->resize(inst->graph->n_vertices, true);
	states[iter][root_state] = root_node;
	root_node->setpack_state = *root_state;

	StateNodeMap::iterator it;
	int vertex;    

	// weights for zero arc
	ObjType* zero_weights = new ObjType[NOBJS];
	memset(zero_weights, 0, sizeof(ObjType)*NOBJS);

	for (int l = 1; l <= inst->graph->n_vertices; ++l) {   
		states[next].clear();

		// select next vertex
		vertex = choose_next_vertex_min_size_next_layer(states[iter]);
		//vertex = l-1;
		var_layer[l-1] = vertex;

		// set weights for one arc
		ObjType* one_weights = new ObjType[NOBJS];
		for (int p = 0; p < NOBJS; ++p) {
			one_weights[p] = objs[p][vertex];
		}

	 	//cout << "\tLayer " << l << " - vertex=" << vertex << " - size=" << states[iter].size() << '\n';

	 	BOOST_FOREACH(StateNodeMap::value_type i, states[iter]) {
			State& state = *(i.first);
			Node* node = i.second;
			bool was_set = state[vertex];

			// zero arc
			state.set(vertex, false);
			it = states[next].find(&state);
			if (it == states[next].end()) {
				Node* new_node = bdd->add_node(l);
				State* new_state = alloc.request();
				(*new_state) = state; 
				states[next][new_state] = new_node;
				new_node->setpack_state = state;
				node->add_out_arc(new_node, 0);
				node->set_arc_weights(0, zero_weights);
			} else {
				node->add_out_arc(it->second, 0);
				node->set_arc_weights(0, zero_weights);
			}

			// one arc
			if (was_set) {
				state &= inst->adj_mask_compl[vertex];
				it = states[next].find(&state);
				if (it == states[next].end()) {
					Node* new_node = bdd->add_node(l);
					State* new_state = alloc.request();
					(*new_state) = state; 
					states[next][new_state] = new_node;
					new_node->setpack_state = state;					
					node->add_out_arc(new_node, 1);
					node->set_arc_weights(1, one_weights);
				} else {
					node->add_out_arc(it->second, 1);
					node->set_arc_weights(1, one_weights);
				}
			}

			// deallocate node state
			alloc.deallocate(i.first);
		}

		// invert iter and next
		next = !next;
		iter = !iter;
	}

	// for (int l = 0; l < bdd->num_layers; ++l) {
	// 	for (int i = 0; i < bdd->layers[l].size(); ++i) {
	// 		Node* node = bdd->layers[l][i];
	// 		if (node->setpack_state.size() == 0) {
	// 			cout << "Error -- " << l << " " << i << endl;
	// 			exit(1);
	// 		}
	// 	}
	// }

	//cout << "\tdone." << endl << endl;
	return bdd;
}


