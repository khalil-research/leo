#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

namespace kp
{
    enum OrderType
    {
        max_weight,
        min_weight,
        max_avg_value,
        min_avg_value,
        max_max_value,
        min_max_value,
        max_min_value,
        min_min_value,
        max_avg_value_by_weight,
        max_max_value_by_weight
    };
    constexpr std::initializer_list<OrderType> all_OrderType = {
        max_weight,
        min_weight,
        max_avg_value,
        min_avg_value,
        max_max_value,
        min_max_value,
        max_min_value,
        min_min_value,
        max_avg_value_by_weight,
        max_max_value_by_weight};

    struct IndexValue
    {
        int i;
        float v;
        IndexValue(int, float);
    };

    bool descending(IndexValue &, IndexValue &);
    bool ascending(IndexValue &, IndexValue &);
    vector<int> get_index(vector<IndexValue>);
    vector<int> get_order(OrderType,
                          vector<int>,
                          vector<vector<int>>);

    string get_order_name(OrderType);
    OrderType get_order_type(int);
}