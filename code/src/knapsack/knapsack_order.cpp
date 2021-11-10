#include <vector>
#include <algorithm>

#include "knapsack_order.hpp"

using namespace std;

namespace kp
{
    IndexValue::IndexValue(int ival, float vval)
    {
        i = ival;
        v = vval;
    }

    bool descending(IndexValue &i, IndexValue &j)
    {
        return (i.v > j.v);
    }

    bool ascending(IndexValue &i, IndexValue &j)
    {
        return (i.v < j.v);
    }

    vector<int> get_index(vector<IndexValue> index_value)
    {
        vector<int> index;
        for (const auto &iv : index_value)
        {
            index.push_back(iv.i);
        }

        return index;
    }

    vector<int> get_order(OrderType order_type,
                          vector<int> wt,
                          vector<vector<int>> val)
    {
        vector<int> order;
        vector<IndexValue> idx_val;
        int p = val.size();
        int n = val[0].size();

        switch (order_type)
        {
        case max_weight:
            // Select items based on descending order of weight
            for (int i = 0; i < wt.size(); i++)
            {
                idx_val.push_back(IndexValue(i, wt[i]));
            }

            sort(idx_val.begin(), idx_val.end(), descending);
            order = get_index(idx_val);
            break;

        case min_weight:
            // Select items based on ascending order of weight
            for (int i = 0; i < wt.size(); i++)
            {
                idx_val.push_back(IndexValue(i, wt[i]));
            }

            sort(idx_val.begin(), idx_val.end(), ascending);
            order = get_index(idx_val);
            break;

        case max_avg_value:
            // Calculate the average of value for each item and then select
            // items based on the descending order of it.
            for (int i = 0; i < n; i++)
            {
                float avg = 0;
                for (int j = 0; j < p; j++)
                {
                    avg += val[j][i];
                }
                avg /= p;
                idx_val.push_back(IndexValue(i, avg));
            }
            sort(idx_val.begin(), idx_val.end(), descending);
            order = get_index(idx_val);
            break;

        case min_avg_value:
            // Calculate the average of value for each item and then select
            // items based on the ascending order of it.
            for (int i = 0; i < n; i++)
            {
                float avg = 0;
                for (int j = 0; j < p; j++)
                {
                    avg += val[j][i];
                }
                avg /= p;
                idx_val.push_back(IndexValue(i, avg));
            }
            sort(idx_val.begin(), idx_val.end(), ascending);
            order = get_index(idx_val);
            break;

        case max_max_value:
            // Calculate the maximum of value for each item and then select
            // items based on the descending order of it.
            for (int i = 0; i < n; i++)
            {
                int maximum = 0;
                for (int j = 0; j < p; j++)
                {
                    if (maximum < val[j][i])
                        maximum = val[j][i];
                }
                idx_val.push_back(IndexValue(i, maximum));
            }
            sort(idx_val.begin(), idx_val.end(), descending);
            order = get_index(idx_val);
            break;

        case min_max_value:
            // Calculate the maximum of value for each item and then select
            // items based on the ascending order of it.
            for (int i = 0; i < n; i++)
            {
                int maximum = 0;
                for (int j = 0; j < p; j++)
                {
                    if (maximum < val[j][i])
                        maximum = val[j][i];
                }
                idx_val.push_back(IndexValue(i, maximum));
            }
            sort(idx_val.begin(), idx_val.end(), ascending);
            order = get_index(idx_val);
            break;

        case max_min_value:
            // Calculate the minimum of value for each item and then select
            // items based on the descending order of it.
            for (int i = 0; i < n; i++)
            {
                int minimum = 1002;
                for (int j = 0; j < p; j++)
                {
                    if (minimum > val[j][i])
                        minimum = val[j][i];
                }
                idx_val.push_back(IndexValue(i, minimum));
            }
            sort(idx_val.begin(), idx_val.end(), descending);
            order = get_index(idx_val);
            break;

        case min_min_value:
            // Calculate the minimum of value for each item and then select
            // items based on the ascending order of it.
            for (int i = 0; i < n; i++)
            {
                int minimum = 1002;
                for (int j = 0; j < p; j++)
                {
                    if (minimum > val[j][i])
                        minimum = val[j][i];
                }
                idx_val.push_back(IndexValue(i, minimum));
            }
            sort(idx_val.begin(), idx_val.end(), ascending);
            order = get_index(idx_val);
            break;

        case max_avg_value_by_weight:
            // Calculate the average value and divide by weight.
            // Select items based on the descending order of this ratio
            for (int i = 0; i < n; i++)
            {
                float avg = 0;
                for (int j = 0; j < p; j++)
                {
                    avg += val[j][i];
                }
                avg /= p;
                idx_val.push_back(IndexValue(i, (float)avg / wt[i]));
            }
            sort(idx_val.begin(), idx_val.end(), descending);
            order = get_index(idx_val);
            break;

        case max_max_value_by_weight:
            // Calculate the max value and divide by weight.
            // Select items based on the descending order of this ratio
            for (int i = 0; i < n; i++)
            {
                int maximum = 0;
                for (int j = 0; j < p; j++)
                {
                    if (maximum < val[j][i])
                        maximum = val[j][i];
                }
                // cout << maximum << " ";
                idx_val.push_back(IndexValue(i, (float)maximum / wt[i]));
                // cout << idx_val[idx_val.size() - 1].v << " :";
            }
            sort(idx_val.begin(), idx_val.end(), descending);
            order = get_index(idx_val);
            break;
        }
        return order;
    }

    string get_order_name(OrderType order_type)
    {
        string order_name;
        switch (order_type)
        {
        case max_weight:
            order_name = "max_weight";
            break;
        case min_weight:
            order_name = "min_weight";
            break;
        case max_avg_value:
            order_name = "max_avg_value";
            break;
        case min_avg_value:
            order_name = "min_avg_value";
            break;
        case max_max_value:
            order_name = "max_max_value";
            break;
        case min_max_value:
            order_name = "min_max_value";
            break;
        case max_min_value:
            order_name = "max_min_value";
            break;
        case min_min_value:
            order_name = "min_min_value";
            break;
        case max_avg_value_by_weight:
            order_name = "max_avg_value_by_weight";
            break;
        case max_max_value_by_weight:
            order_name = "max_max_value_by_weight";
            break;
        }

        return order_name;
    }
}