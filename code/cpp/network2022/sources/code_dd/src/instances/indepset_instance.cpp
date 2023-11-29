/**
 * -------------------------------------------------
 * Independent Set structure - Implementation
 * -------------------------------------------------
 */

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "indepset_instance.hpp"
#include "../util/util.hpp"

#include <ilcp/cpext.h>

using namespace std;



void Graph::export_to_dimacs(){

	ofstream new_graph_file;
	new_graph_file.open("new_graph.dat");
	new_graph_file << "p e " << n_vertices  << " " << n_edges << endl;
	for(int i = 0 ; i < n_vertices ; ++i){
		for(int j = i+1 ; j < n_vertices ; ++j){
			if( adj_m[i][j] ){
				new_graph_file << "e " << i+1 << " " << j+1 << endl;
			}
		}
	}

	new_graph_file.close();

}

//
// Read graph in DIMACS format
//
void Graph::read_dimacs(const char* filename) {

	string buffer;
	char   command;

	ifstream input(filename);

	if (!input.is_open()) {
		cerr << "Error: could not open DIMACS graph file " << filename << endl << endl;
		exit(1);
	}

	int read_edges = 0;
	n_edges = -1;

	int source, target;

	while ( read_edges != n_edges && !input.eof() ) {

		input >> command;

		if (command == 'c') {
			// read comment
			getline(input, buffer);

		} else if (command == 'n') {
			// read weight
			input >> source;
			source--;
			input >> weights[source];

		} else if (command == 'p') {
			// read 'edge' or 'col'
			input >> buffer;

			// read number of vertices and edges
			input >> n_vertices;
			input >> n_edges;

			// allocate adjacent matrix
			adj_m = new bool*[n_vertices];
			for (int i = 0; i < n_vertices; i++) {
				adj_m[i] = new bool[n_vertices];
				memset(adj_m[i], false, sizeof(bool)*n_vertices);
			}

			// allocate adjacent list
			adj_list.resize(n_vertices);

			// initialize weights
			weights = new double[n_vertices];
			for (int i = 0; i < n_vertices; ++i) {
				weights[i] = 1.0;
			}

		} else if (command == 'e') {

			if (input.eof()) {
				break;
			}

			// read edge
			input >> source;
			source--;

			input >> target;
			target--;

			set_adj(source, target);
			set_adj(target, source);

			read_edges++;
		}

	}

	input.close();

	//    // we assume a vertex is adjacent to itself
	//    for( int i = 0; i < n_vertices; i++ ) {
	//        set_adj(i, i);
	//    }

	int count_edges = 0;
	for (int i = 0; i < n_vertices; i++) {
		for (int j = i+1; j < n_vertices; j++) {
			if (is_adj(i,j)) {
				count_edges++;
			}
		}
	}

	n_edges = count_edges;
}




//
// Export to gml
//
void Graph::export_to_gml(const char* output) {

	ofstream file(output);
	file << "graph [\n";

	for (int i = 0; i < n_vertices; i++) {
		file << "node [\n";
		file << "\tid " << i << "\n";
		file << "\tlabel \"" << i << "\"\n";

		file << "\t graphics [ \n";
		file << "\t\t type \"ellipse\"\n";
		file << "\t\t hasFill 0 \n";
		file << "\t\t ] \n";

		file << "\t]\n" << endl;
	}
	int total_edges = 0;
	for (int i = 0; i < n_vertices; ++i) {
		for (int j = i+1; j < n_vertices; ++j) {
			if ( !is_adj(i, j) )
				continue;
			file << "edge [\n";
			file << "\t source " << i << "\n";
			file << "\t target " << j << "\n";
			file << "\t]\n";
			total_edges++;
		}
	}
	file << "\n]";
	file.close();
	cout << "TOTAL EDGES: " << total_edges << endl;
}


//
// Create an isomorphic graph according to a vertex mapping
// Mapping description: mapping[i] = position where vertex i is in new ordering
//
//
Graph::Graph(Graph* graph, vector<int>& mapping)
: n_vertices(graph->n_vertices), n_edges(graph->n_edges)
{
	// allocate adjacent matrix
	adj_m = new bool*[n_vertices];
	for (int i = 0; i < n_vertices; ++i) {
		adj_m[i] = new bool[n_vertices];
		memset(adj_m[i], false, sizeof(bool)*n_vertices);
	}

	// allocate adjacent list
	adj_list.resize(n_vertices);

	// construct graph according to mapping
	for (int i = 0; i < graph->n_vertices; ++i) {
		for (vector<int>::iterator it = graph->adj_list[i].begin();
				it != graph->adj_list[i].end();
				it++)
		{
			set_adj(mapping[i], mapping[*it]);
		}
	}
}


//
// Print graph
//
void Graph::print() {
	cout << "Graph" << endl;
	for (int v = 0; v < n_vertices; ++v) {
		if (adj_list[v].size() != 0) {
			cout << "\t" << v << " --> ";
			for (vector<int>::iterator it = adj_list[v].begin();
					it != adj_list[v].end();
					++it)
			{
				cout << *it << " ";
			}
			cout << endl;
		}
	}
	cout << endl;
}


//
// Extract tree from a graph
//
Graph* Graph::extract_clique(Graph* graph, vector<int>& clique) {
	// find vertex with largest degree
	int vertex = -1;
	int max_degree = 0;
	for( int v = 0; v < graph->n_vertices; ++v ) {
		if( graph->degree(v) > max_degree ) {
			vertex = v;
			max_degree = graph->degree(v);
		}
	}

	// if maximum degree is zero, graph has no edges
	if( max_degree == 0 ) {
		return NULL;
	}

	// vertices already in clique (using bool* might be slightly more efficient)
	vector<bool> in_clique( graph->n_vertices, false );
	clique.clear();
	clique.push_back( vertex );
	in_clique[vertex] = true;

	// grow clique by taking vertices that are adjacent to 'vertex'
	// with maximum degree
	while ( true ) {

		vector<int> subgraph;
		for( vector<int>::iterator u = graph->adj_list[vertex].begin();
				u != graph->adj_list[vertex].end();
				++u )
		{
			// check if vertex is adjacent to all vertices in clique
			bool is_adj_all = !in_clique[ *u ];
			for( int j = 1; j < (int)clique.size() && is_adj_all; ++j ) {
				is_adj_all = graph->is_adj( clique[j], *u );
			}
			if( is_adj_all ) {
				subgraph.push_back( *u );
			}
		}

		if( subgraph.empty() ) {
			break;
		}

		int max_degree = -1;
		int selected_u = -1;
		for( int i = 0; i < (int)subgraph.size(); ++i ) {
			int degree = 0;
			for( int j = 0; j < (int)subgraph.size(); ++j ) {
				if( i == j ) continue;
				if( graph->is_adj( subgraph[i], subgraph[j] ) ) {
					degree++;
				}
			}
			if( degree > max_degree ) {
				selected_u = subgraph[i];
				max_degree = degree;
			}
		}

		// add vertex to clique
		clique.push_back( selected_u );
		in_clique[selected_u] = true;
	}

	// build graph representing clique, and remove clique from original graph
	Graph* clique_graph = new Graph( graph->n_vertices );
	for (int i = 0; i < (int)clique.size(); ++i) {
		for (int j = i+1; j < (int)clique.size(); ++j) {
			clique_graph->add_edge(clique[i], clique[j]);
			graph->remove_edge( clique[i], clique[j] );
		}
	}

	return clique_graph;
}



//
// Extract lifted clique from a graph
//
Graph* Graph::extract_lifted_clique(Graph* graph, vector<int>& clique) {
	// find vertex with largest degree
	int vertex = -1;
	int max_degree = 0;
	for( int v = 0; v < graph->n_vertices; ++v ) {
		if( graph->degree(v) > max_degree ) {
			vertex = v;
			max_degree = graph->degree(v);
		}
	}

	// if maximum degree is zero, graph has no edges
	if( max_degree == 0 ) {
		return NULL;
	}

	// vertices already in clique (using bool* might be slightly more efficient)
	vector<bool> in_clique( graph->n_vertices, false );
	clique.clear();
	clique.push_back( vertex );
	in_clique[vertex] = true;

	// grow clique by taking vertices that are adjacent to 'vertex'
	// with maximum degree
	while ( true ) {

		vector<int> subgraph;
		for( vector<int>::iterator u = graph->adj_list[vertex].begin();
				u != graph->adj_list[vertex].end();
				++u )
		{
			// check if vertex is adjacent to all vertices in clique
			bool is_adj_all = !in_clique[ *u ];
			for( int j = 1; j < (int)clique.size() && is_adj_all; ++j ) {
				is_adj_all = graph->is_adj( clique[j], *u );
			}
			if( is_adj_all ) {
				subgraph.push_back( *u );
			}
		}

		if( subgraph.empty() ) {
			break;
		}

		int max_degree = -1;
		int selected_u = -1;
		for( int i = 0; i < (int)subgraph.size(); ++i ) {
			int degree = 0;
			for( int j = 0; j < (int)subgraph.size(); ++j ) {
				if( i == j ) continue;
				if( graph->is_adj( subgraph[i], subgraph[j] ) ) {
					degree++;
				}
			}
			if( degree > max_degree ) {
				selected_u = subgraph[i];
				max_degree = degree;
			}
		}

		// add vertex to clique
		clique.push_back( selected_u );
		in_clique[selected_u] = true;
	}

	// cout << "clique: ";
	// for (int i = 0; i < clique.size(); ++i) {
	//   cout << clique[i] << " ";
	// }
	// cout << endl;

	// search for vertices that are now 'almost' adjancent to clique
	while (true) {

		int max_v = -1;
		int max_adj = 0;

		for (int v = 0; v < graph->n_vertices; ++v) {
			int adj_count = 0;
			if (!in_clique[v]) {
				for (int i = 0; i < clique.size(); ++i) {
					if (graph->is_adj(clique[i], v)) {
						adj_count++;
					}
				}
				if (adj_count > max_adj) {
					max_v = v;
					max_adj = adj_count;
				}
			}
		}

		double gap = (double)max_adj/(double)clique.size();

		//cout << gap << endl;

		if (gap <= 0.3) {
			break;
		}

		clique.push_back(max_v);
		in_clique[max_v] = true;
		//cout << max_v << ": " << max_adj << " - size=" << gap << endl;
		//exit(1);
	}



	// build graph representing clique, and remove clique from original graph
	Graph* clique_graph = new Graph( graph->n_vertices );
	for (int i = 0; i < (int)clique.size(); ++i) {
		for (int j = i+1; j < (int)clique.size(); ++j) {
			if (graph->is_adj(clique[i], clique[j])) {
				clique_graph->add_edge(clique[i], clique[j]);
				graph->remove_edge( clique[i], clique[j] );
			}
		}
	}

	return clique_graph;
}




//
// Union Find
//
vector<int> parent;
int find(int i) {
	if (parent[i] == i) return i;
	return find(parent[i]);
}
void union_find(int i, int j) {
	parent[find(i)] = find(j);
}


//
// Extract tree from a graph
//
Graph* Graph::extract_tree(Graph* graph, vector<int>& tree) {

	// check if graph is empty
	if (graph->n_edges == 0) {
		return NULL;
	}

	// initialize union/find
	for (int i = 0; i < graph->n_vertices; ++i) {
		parent[i] = i;
	}

	Graph* tree_graph = new Graph(graph->n_vertices);

	// create forest based on edges in the graph
	for (int v = 0; v < graph->n_vertices; ++v) {
		for (int u = v+1; u < graph->n_vertices; ++u) {
			if (graph->is_adj(u,v)) {
				if (find(u) != find(v)) {
					tree_graph->add_edge(u,v);
					//tree.push_back(u);
					//tree.push_back(v);

					graph->remove_edge(u,v);
					union_find(u,v);
				}
			}
		}
	}

	return tree_graph;
}


//
// Decompose graphs into cluster cliques
//

Graph* Graph::decompose_into_cluster_cliques(vector<Graph*>& cliques,
		vector< vector<int> >& vertex_ind)
{
	Graph* graph = NULL;

	try {

		IloEnv env;
		IloModel model(env);

		int num_clusters = 6;

		IloIntVarArray x(env, n_vertices, 0, num_clusters-1);

		IloExpr obj(env);
		for (int i = 0; i < this->n_vertices; ++i) {
			for (int j = i+1; j < this->n_vertices; ++j) {
				if (!this->is_adj(i,j)) {
					obj += (x[i] == x[j]);
				} else {
					obj += (x[i] != x[j]);
				}
			}
		}

		model.add( IloMinimize(env, obj) );

		IloCP cp(model);
		cp.setParameter(IloCP::Workers, 1);
		cp.setParameter(IloCP::TimeLimit, 30);

		cp.solve();


		int t_clusters = 0;
		vector<int> cluster_map(num_clusters, -1);
		for (int i = 0; i < this->n_vertices; ++i) {
			if (cluster_map[cp.getValue(x[i])] == -1) {
				cluster_map[cp.getValue(x[i])] = t_clusters++;
			}
		}

		cout << "Total number of clusters: " << t_clusters << endl;

		// collect cliques
		vertex_ind.clear();
		vertex_ind.resize(t_clusters);
		for (int i = 0; i < this->n_vertices; ++i) {
			vertex_ind[cluster_map[cp.getValue(x[i])]].push_back(i);
		}

		cliques.clear();
		cliques.resize(t_clusters);
		for (int c = 0; c < t_clusters; ++c) {
			cliques[c] = new Graph(this->n_vertices);
			cout << "Clique " << c << ": ";
			for (size_t i = 0; i < vertex_ind[c].size(); ++i) {
				cout << vertex_ind[c][i] << " ";
				for (size_t j = i+1; j < vertex_ind[c].size(); ++j) {
					int u = vertex_ind[c][i];
					int v = vertex_ind[c][j];
					if (is_adj(u,v)) {
						cliques[c]->add_edge(u,v);
					}
				}
			}
			cout << endl;
		}




	} catch (IloException& ex) {
		cout << "ILOG Error: " << ex << endl;
		exit(1);
	}

	return graph;
}


//
// Decompose graphs into trees
//
void Graph::decompose_into_trees(vector<Graph*>& trees,
		vector< vector<int> >& vertex_ind)
{
	cout << "\nDecomposing into trees..." << endl;

	// initialize parents for union/find
	parent.resize(this->n_vertices);

	// create copy of current graph
	Graph* graph = new Graph( this->n_vertices );
	for (int v = 0; v < graph->n_vertices; ++v ) {
		if (this->adj_list[v].size() != 0) {
			for( vector<int>::iterator it = this->adj_list[v].begin();
					it != this->adj_list[v].end();
					++it )
			{
				graph->add_edge( v, *it );
			}
		}
	}

	trees.clear();
	vertex_ind.clear();

	// Extract trees
	vector<int> tree;
	Graph* tree_graph = extract_tree(graph, tree);

	int sum_edges = 0;

	while (tree_graph != NULL) {
		//cout << "Tree - size=" << tree_graph->n_edges << endl;
		sum_edges += tree_graph->n_edges;

		trees.push_back(tree_graph);
		vertex_ind.push_back(tree);

		tree_graph = extract_tree(graph, tree);
	}

	cout << "\tnum trees: " << trees.size() << endl;
	cout << "\ttotal edges: " << sum_edges << endl;
	cout << endl;

}

//
// Export negated graph to DIMACS
//
void Graph::export_negated_dimacs(string filename) {
	ofstream out(filename.c_str());

	// number of edges in the negated graph
	long int num_edges = 0;
	for (int u = 0; u < n_vertices; ++u) {
		for (int v = u+1; v < n_vertices; ++v) {
			if (!is_adj(u,v)) {
				num_edges++;
			}
		}
	}

	out << "c negated dimacs" << endl;
	out << "p edge " << n_vertices << " " << num_edges << endl;
	for (int u = 0; u < n_vertices; ++u) {
		for (int v = u+1; v < n_vertices; ++v) {
			if (!is_adj(u,v)) {
				out << "e " << (u+1) << " " << (v+1) << endl;
			}
		}
	}

	out.close();
}


