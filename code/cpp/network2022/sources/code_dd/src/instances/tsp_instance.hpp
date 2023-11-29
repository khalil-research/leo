// ----------------------------------------------------------
// TSP Instance
// ----------------------------------------------------------

#ifndef TSP_INSTANCE_HPP_
#define TSP_INSTANCE_HPP_

#include <iostream>
#include <vector>

using namespace std;

//
// TSP Instance
//
struct TSPInstance
{
	// Number of cities
	int n_cities;
	// Number of objective functions
	int n_objs;
	// Objective functions  (indexed by objective/city/city)
	vector<vector<vector<int>>> objs;

	vector<vector<vector<int>>> objs_canonical;

	// Empty Constructor
	TSPInstance() {}

	// Read instance based on our model
	void read(const char *filename);

	void reset_order(vector<int> new_order);
};

#endif