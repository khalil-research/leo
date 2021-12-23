#include <cstring>
#include <vector>
#include <algorithm>
#include <assert.h>
#include "knapsack_order.hpp"

using namespace std;

IndexValue::IndexValue(int ival, float vval)
{
    idx = ival;
    val = vval;
}

bool descending(IndexValue a, IndexValue b)
{
    return (a.val > b.val);
}

vector<int> get_index(vector<IndexValue> index_value)
{
    vector<int> index;
    for (const auto &iv : index_value)
    {
        index.push_back(iv.idx);
    }

    return index;
}

vector<float> get_feature(int feature_type,
                          vector<int> weight,
                          vector<vector<int>> values)
{
    vector<float> feature;
    int i, j;
    float sum;
    float max_val;
    float min_val;

    switch (feature_type)
    {
    case WEIGHT:
        for (i = 0; i < weight.size(); i++)
        {
            feature.push_back(weight[i]);
        }
        break;

    case AVG_VALUE:
        for (i = 0; i < weight.size(); i++)
        {
            sum = 0;
            for (j = 0; j < values.size(); j++)
            {
                sum += values[j][i];
            }
            feature.push_back(sum / values.size());
        }
        break;

    case MAX_VALUE:
        for (i = 0; i < weight.size(); i++)
        {
            max_val = 0;
            for (j = 0; j < values.size(); j++)
            {
                if (max_val < values[j][i])
                {
                    max_val = values[j][i];
                }
            }
            feature.push_back(max_val);
        }
        break;

    case MIN_VALUE:
        for (i = 0; i < weight.size(); i++)
        {
            min_val = 1001;
            for (j = 0; j < values.size(); j++)
            {
                if (min_val > values[j][i])
                {
                    min_val = values[j][i];
                }
            }
            feature.push_back(min_val);
        }
        break;

    case AVG_VALUE_BY_WEIGHT:
        for (i = 0; i < weight.size(); i++)
        {
            sum = 0;
            for (j = 0; j < values.size(); j++)
            {
                sum += values[j][i];
            }
            feature.push_back((sum / values.size()) / weight[i]);
        }
        break;

    case MAX_VALUE_BY_WEIGHT:
        for (i = 0; i < weight.size(); i++)
        {
            max_val = 0;
            for (j = 0; j < values.size(); j++)
            {
                if (max_val < values[j][i])
                {
                    max_val = values[j][i];
                }
            }
            feature.push_back(max_val / weight[i]);
        }
        break;

    case MIN_VALUE_BY_WEIGHT:
        for (i = 0; i < weight.size(); i++)
        {
            min_val = 1001;
            for (j = 0; j < values.size(); j++)
            {
                if (min_val > values[j][i])
                {
                    min_val = values[j][i];
                }
            }
            feature.push_back(min_val / weight[i]);
        }
        break;
    }
    assert(feature.size() == weight.size());
    // cout << feature_type << " ";
    // for (int i = 0; i < feature.size(); i++)
    // {
    //     cout << feature[i] << " ";
    // }
    // cout << endl;
    return feature;
}

vector<int> get_order(vector<float> features_weights,
                      vector<int> kp_weight,
                      vector<vector<int>> kp_values)
{
    int p = kp_values.size();
    int n = kp_values[0].size();

    // cout << p << " " << n;

    // Get features
    vector<vector<float>> features;
    for (const auto &feature_type : feature_types)
    {
        features.push_back(get_feature(feature_type, kp_weight, kp_values));
    }

    // Calculate variables score
    vector<IndexValue> scores;
    for (int i = 0; i < n; i++)
    {
        scores.push_back(IndexValue(i, 0));
        for (int j = 0; j < features_weights.size(); j++)
        {
            scores[i].val += (features_weights[j] * features[j][i]);
        }
        // cout << scores[i].val << endl;
    }

    // Sort descending based on scores
    stable_sort(scores.begin(), scores.end(), descending);

    // Fetch item ids
    vector<int> order;
    for (int i = 0; i < n; i++)
    {
        order.push_back(scores[i].idx);
    }

    return order;
}