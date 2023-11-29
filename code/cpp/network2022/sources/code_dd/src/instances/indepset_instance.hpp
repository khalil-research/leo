/*
 * --------------------------------------------------------
 * Independent set data structure
 * --------------------------------------------------------
 */

#ifndef INSTANCE_HPP_
#define INSTANCE_HPP_

#include <cstring>
#include <fstream>
#include <iostream>
#include <boost/dynamic_bitset.hpp>

using namespace std;


//
// Graph structure
//
struct Graph {

	int                         n_vertices;         /**< |V| */
	int                         n_edges;            /**< |E| */
	double*		      weights;		  /**< weight of each vertex */

	bool**                      adj_m;              /**< adjacent matrix */
	vector< vector<int> >       adj_list;           /**< adjacent list */


	/** Set two vertices as adjacents */
	void set_adj(int i, int j);

	/** Check if two vertices are adjancent */
	bool is_adj(int i, int j);

	/** Empty constructor */
	Graph();

	/** Create an isomorphic graph according to a vertex mapping */
	Graph(Graph* graph, vector<int>& mapping);

	/** Read graph from a DIMACS format */
	void read_dimacs(const char* filename);

	/** Export graph from a DIMACS format */
	void export_to_dimacs();

	/** Export to GML format */
	void export_to_gml(const char* output);

	/** Constructor with number of vertices */
	Graph(int num_vertices);

	/** Add edge */
	void add_edge(int i, int j);

	/** Remove edge */
	void remove_edge(int i, int j);

	/** Return degree of a vertex */
	int degree( int v ) { return adj_list[v].size(); }

	/** Print graph */
	void print();

	/** Export negated graph to dimacs format */
	void export_negated_dimacs(string filename);

	/** Decompose graphs into cliques */
	void decompose_into_cliques(vector<Graph*>& cliques, vector< vector<int> >& vertex_ind);

	/** Decompose graphs into cliques greater than 'k' */
	Graph* decompose_into_lifted_cliques_k(vector<Graph*>& cliques,
			vector< vector<int> >& vertex_ind,
			int k);

	/** Decompose graphs into lifted cliques */
	Graph* decompose_into_lifted_cliques(vector<Graph*>& cliques,
			vector< vector<int> >& vertex_ind);

	/** Decompose graphs into cluster cliques */
	Graph* decompose_into_cluster_cliques(vector<Graph*>& cliques,
			vector< vector<int> >& vertex_ind);


	/** Decompose graphs into trees */
	void decompose_into_trees(vector<Graph*>& trees, vector< vector<int> >& vertex_ind);


private:
	// Extract a clique from a graph
	Graph* extract_clique(Graph* graph, vector<int>& indices);

	// Extract a lifted clique from a graph
	Graph* extract_lifted_clique(Graph* graph, vector<int>& indices);

	// Extract a tree from a graph
	Graph* extract_tree(Graph* graph, vector<int>& indices);
};



//
// Independent set instance structure
//
struct IndepSetInst {

	Graph*              				graph;             	// independent set graph
	vector< boost::dynamic_bitset<> >	adj_mask_compl;	 	// complement mask of adjacencies


	/** Read DIMACS independent set instance */
	void read_DIMACS(const char* filename);

	/** Create empty instance */
	IndepSetInst() { }

	/** Create from file */
	IndepSetInst(const char* filename) { read_DIMACS(filename); }

	/** Create from graph */
	IndepSetInst(Graph* _graph);
};



/*
 * -----------------------------------------------------
 * Inline implementations: Graph
 * -----------------------------------------------------
 */


/**
 * Empty constructor
 */
inline Graph::Graph() : n_vertices(0), n_edges(0), weights(NULL), adj_m(NULL) {
}


/**
 * Constructor with number of vertices
 **/
inline Graph::Graph(int num_vertices)
: n_vertices(num_vertices), n_edges(0), weights(NULL)
{
	adj_m = new bool*[ num_vertices ];
	for (int i = 0; i < num_vertices; ++i) {
		adj_m[i] = new bool[ num_vertices ];
		memset( adj_m[i], false, sizeof(bool) * num_vertices );
	}
	adj_list.resize(num_vertices);
}


/**
 * Check if two vertices are adjacent
 */
inline bool Graph::is_adj(int i, int j) {
	assert(i >= 0);
	assert(j >= 0);
	assert(i < n_vertices);
	assert(j < n_vertices);
	return adj_m[i][j];
}


/**
 * Set two vertices as adjacent
 */
inline void Graph::set_adj(int i, int j) {
	assert(i >= 0);
	assert(j >= 0);
	assert(i < n_vertices);
	assert(j < n_vertices);

	// check if already adjacent
	if (adj_m[i][j])
		return;

	// add to adjacent matrix and list
	adj_m[i][j] = true;
	adj_list[i].push_back(j);
}



/**
 * Add edge
 **/
inline void Graph::add_edge(int i, int j) {
	assert(i >= 0);
	assert(j >= 0);
	assert(i < n_vertices);
	assert(j < n_vertices);

	// check if already adjacent
	if (adj_m[i][j])
		return;

	// add to adjacent matrix and list
	adj_m[i][j] = true;
	adj_m[j][i] = true;
	adj_list[i].push_back(j);
	adj_list[j].push_back(i);

	n_edges++;
}

/**
 * Remove edge
 **/
inline void Graph::remove_edge(int i, int j) {
	assert(i >= 0);
	assert(j >= 0);
	assert(i < n_vertices);
	assert(j < n_vertices);

	// check if already adjacent
	if (!adj_m[i][j])
		return;

	// add to adjacent matrix and list
	adj_m[i][j] = false;
	adj_m[j][i] = false;

	for (int v = 0; v < (int)adj_list[i].size(); ++v) {
		if ( adj_list[i][v] == j ) {
			adj_list[i][v] = adj_list[i].back();
			adj_list[i].pop_back();
			break;
		}
	}

	for (int v = 0; v < (int)adj_list[j].size(); ++v) {
		if ( adj_list[j][v] == i ) {
			adj_list[j][v] = adj_list[j].back();
			adj_list[j].pop_back();
			break;
		}
	}
	n_edges--;
}


/*
 * -----------------------------------------------------
 * Inline implementations: Independent Set
 * -----------------------------------------------------
 */


//
// Read DIMACS independent set instance with no costs
//
inline void IndepSetInst::read_DIMACS(const char *filename) {

	cout << "\nReading instance " << filename << endl;

	// read graph
	graph = new Graph;
	graph->read_dimacs(filename);

	cout << "\tnumber of vertices: " << graph->n_vertices << endl;
	cout << "\tnumber of edges: " << graph->n_edges << endl;

	// create complement mask of adjacencies
	adj_mask_compl.resize(graph->n_vertices);
	for( int v = 0; v < graph->n_vertices; v++ ) {

		adj_mask_compl[v].resize(graph->n_vertices, true);
		for( int w = 0; w < graph->n_vertices; w++ ) {
			if( graph->is_adj(v,w) ) {
				adj_mask_compl[v].set(w, false);
			}
		}

		// we assume here a vertex is adjacent to itself
		adj_mask_compl[v].set(v, false);

	}
	cout << "\tdone.\n" << endl;
}




//
// Decompose graphs into cliques
//
inline void Graph::decompose_into_cliques(vector<Graph*>& cliques,
		vector< vector<int> >& vertex_ind)
{
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

	cliques.clear();
	vertex_ind.clear();

	// Extract cliques
	vector<int> clique;
	Graph* clique_graph = extract_clique(graph, clique);

	while (clique_graph != NULL) {
		// cout << "Clique - size=" << clique.size() << ": ";
		// for( int i = 0; i < (int)clique.size(); ++i ) {
		//   cout << clique[i] << " ";
		// }
		// cout << endl;

		cliques.push_back(clique_graph);
		vertex_ind.push_back(clique);

		clique_graph = extract_clique(graph, clique);
	}

}

//
// Decompose graphs into cliques greater than k
//
inline Graph* Graph::decompose_into_lifted_cliques_k(vector<Graph*>& cliques,
		vector< vector<int> >& vertex_ind,
		int k)
{
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

	cliques.clear();
	vertex_ind.clear();

	// Extract cliques
	vector<int> clique;
	Graph* clique_graph = extract_lifted_clique(graph, clique);

	while (clique_graph != NULL) {
		cout << "Clique - size=" << clique.size() << ": ";
		for( int i = 0; i < (int)clique.size(); ++i ) {
			cout << clique[i] << " ";
		}
		cout << endl;

		cliques.push_back(clique_graph);
		vertex_ind.push_back(clique);

		if (clique.size() <= k) {
			return graph;
		}

		clique_graph = extract_lifted_clique(graph, clique);
	}

	return NULL;
}



/** Create IndepsetInst from graph */
inline IndepSetInst::IndepSetInst(Graph* _graph) : graph(_graph) {
	// create complement mask of adjacencies
	adj_mask_compl.resize(graph->n_vertices);
	for( int v = 0; v < graph->n_vertices; v++ ) {

		adj_mask_compl[v].resize(graph->n_vertices, true);
		for( int w = 0; w < graph->n_vertices; w++ ) {
			if( graph->is_adj(v,w) ) {
				adj_mask_compl[v].set(w, false);
			}
		}

		// we assume here a vertex is adjacent to itself
		adj_mask_compl[v].set(v, false);
	}
}


//
// Decompose graphs into cliques
//
inline Graph* Graph::decompose_into_lifted_cliques(vector<Graph*>& cliques,
		vector< vector<int> >& vertex_ind)
{
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

	cliques.clear();
	vertex_ind.clear();

	// Extract cliques
	vector<int> clique;
	Graph* clique_graph = extract_lifted_clique(graph, clique);

	while (clique_graph != NULL) {
		cout << "Clique - size=" << clique.size() << ": ";
		for( int i = 0; i < (int)clique.size(); ++i ) {
			cout << clique[i] << " ";
		}
		cout << endl;

		cliques.push_back(clique_graph);
		vertex_ind.push_back(clique);

		clique_graph = extract_lifted_clique(graph, clique);
	}

	return NULL;
}




#endif /* THISANCE_HPP_ */



