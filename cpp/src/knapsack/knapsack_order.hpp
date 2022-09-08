#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

const int WEIGHT = 0;
const int AVG_VALUE = 1;
const int MAX_VALUE = 2;
const int MIN_VALUE = 3;
const int AVG_VALUE_BY_WEIGHT = 4;
const int MAX_VALUE_BY_WEIGHT = 5;
const int MIN_VALUE_BY_WEIGHT = 6;
const vector<int> feature_types = {WEIGHT, AVG_VALUE, MAX_VALUE, MIN_VALUE,
                                   AVG_VALUE_BY_WEIGHT,
                                   MAX_VALUE_BY_WEIGHT,
                                   MIN_VALUE_BY_WEIGHT};

struct IndexValue
{
    int idx;
    float val;
    IndexValue(int, float);
};

bool descending(IndexValue, IndexValue);
vector<int> get_index(vector<IndexValue>);

// Get features for a given feature type
vector<float> get_feature(int feature_type,
                          vector<int> kp_weight,
                          vector<vector<int>> kp_values);

// Get variable order based on weighted feature scores
vector<int> get_order(vector<float> feature_weights,
                      vector<int> kp_weight,
                      vector<vector<int>> values);
