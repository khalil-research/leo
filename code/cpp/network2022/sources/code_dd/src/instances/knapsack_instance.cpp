// ----------------------------------------------------------
// Knapsack Instance - Implementations
// ----------------------------------------------------------

#include <algorithm>
#include <fstream>
#include <iostream>
#include <cstdlib>

#include "knapsack_instance.hpp"

using namespace std;

//
// Read instance based on our format
//
void KnapsackInstance::read(char *filename)
{
    ifstream input(filename);
    if (!input.is_open())
    {
        cout << "Error - file " << filename << " could not be found" << endl;
        exit(1);
    }
    // Read basic info
    input >> n_vars;
    input >> n_cons;
    input >> num_objs;
    // Allocate memory
    obj_coeffs.resize(n_vars, vector<int>(num_objs, 0));
    coeffs.resize(n_cons, vector<int>(n_vars));
    rhs.resize(n_cons, 0);
    // Read objective coefficients
    for (int o = 0; o < num_objs; ++o)
    {
        for (int i = 0; i < n_vars; ++i)
        {
            input >> obj_coeffs[i][o];
        }
    }
    // Read coefficients and right-hand side
    for (int c = 0; c < n_cons; ++c)
    {
        for (int i = 0; i < n_vars; ++i)
        {
            input >> coeffs[c][i];
        }
        input >> rhs[c];
    }

    obj_coeffs_canonical = obj_coeffs;
    coeffs_canonical = coeffs;

    // Print instance
    // print();
}

//
// Print instance
//
void KnapsackInstance::print()
{
    cout << "\nKnapsack Instance: " << endl;
    cout << "\tnum variables: " << n_vars << endl;
    cout << "\tnum constraints: " << n_cons << endl;
    cout << "\tnum objs: " << num_objs << endl;
    cout << endl;

    for (int o = 0; o < num_objs; ++o)
    {
        cout << "Objective " << o << endl;
        cout << "\t";
        for (int i = 0; i < n_vars; ++i)
        {
            cout << obj_coeffs[i][o] << " ";
        }
        cout << endl;
    }
    for (int c = 0; c < n_cons; ++c)
    {
        cout << "Constraint " << c << endl;
        cout << "\t";
        for (int i = 0; i < n_vars; ++i)
        {
            cout << coeffs[c][i] << " ";
        }
        cout << " <= " << rhs[c] << endl;
    }
}

//
// Comparator based on largest constraint coefficients, in order of input
//
struct LargestCoeffComp
{
    // Array of coefficients
    const vector<vector<int>> &coeffs;

    // Constructor
    LargestCoeffComp(const vector<vector<int>> &_coeffs) : coeffs(_coeffs)
    {
    }

    // Comparator
    bool operator()(const int i, const int j) const
    {
        int sumA = 0;
        int sumB = 0;
        // for (int c = 0; c < coeffs.size(); ++c) {
        for (int c = coeffs.size() - 1; c >= 0; --c)
        {
            if (coeffs[c][i] != coeffs[c][j])
            {
                return coeffs[c][i] < coeffs[c][j];
            }
            sumA += coeffs[c][i];
            sumB += coeffs[c][j];
        }
        if (sumA != sumB)
        {
            return sumA < sumB;
        }
        return i < j;
    }
};

//
// Reorder variables based on constraint coefficients
//
void KnapsackInstance::reorder_coefficients()
{
    // Mapping of variable indices
    vector<int> map(n_vars);
    for (int i = 0; i < n_vars; ++i)
    {
        map[i] = i;
    }

    // Sort
    LargestCoeffComp comp(coeffs);
    sort(map.begin(), map.end(), comp);

    // Rearrange arrays
    vector<vector<int>> new_coeffs = coeffs;
    for (int c = 0; c < n_cons; ++c)
    {
        for (int i = 0; i < n_vars; ++i)
        {
            new_coeffs[c][i] = coeffs[c][map[i]];
        }
    }
    coeffs = new_coeffs;
    vector<vector<int>> new_obj_coeffs = obj_coeffs;
    for (int i = 0; i < n_vars; ++i)
    {
        for (int o = 0; o < NOBJS; ++o)
        {
            new_obj_coeffs[i][o] = obj_coeffs[map[i]][o];
        }
    }
    obj_coeffs = new_obj_coeffs;

    // for (int i = 0; i < n_vars; ++i) {
    //     cout << map[i] << " ";
    // }
    // cout << endl;
    // exit(1);
}

void KnapsackInstance::reset_order(vector<int> new_order)
{
    order = new_order;
    for (int c = 0; c < n_cons; ++c)
    {
        for (int i = 0; i < n_vars; ++i)
        {
            coeffs[c][i] = coeffs_canonical[c][new_order[i]];
        }
    }

    for (int i = 0; i < n_vars; i++)
    {

        for (int o = 0; o < num_objs; o++)
        {
            obj_coeffs[i][o] = obj_coeffs_canonical[new_order[i]][o];
        }
    }
}

void KnapsackInstance::reset_order()
{
    for (int i = 0; i < n_vars; ++i)
    {
        order[i] = i;
    }

    coeffs = coeffs_canonical;
    obj_coeffs = obj_coeffs_canonical;
}