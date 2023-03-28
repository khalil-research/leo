#include <iostream>
#include <vector>
#include <fstream>

// #include "../order.hpp"

using namespace std;

struct MultiObjKnapsackInstanceOrdered
{
    //
    // Multiobjective knapsack problem
    //

    // Number of variables
    int n_vars;
    // Number of objective functions
    int num_objs;
    // Objective function coefficients
    vector<vector<int>> obj_coeffs;
    vector<vector<int>> obj_coeffs_canonical;

    // Constraint coefficients
    vector<int> coeffs;
    vector<int> coeffs_canonical;

    // Right-hand side
    int rhs;
    // Order of the variables
    vector<int> order;

    //
    // Constructor
    //
    MultiObjKnapsackInstanceOrdered(char *inputfile)
    {
        ifstream knapsack(inputfile);
        if (!knapsack.is_open())
        {
            cout << "Error: could not open file " << inputfile << endl;
            exit(1);
        }

        // Read Data
        knapsack >> n_vars;
        knapsack >> num_objs;

        int val;

        obj_coeffs.resize(num_objs);
        for (int o = 0; o < num_objs; ++o)
        {
            obj_coeffs[o].resize(n_vars);
            for (int i = 0; i < n_vars; ++i)
            {
                knapsack >> val;
                obj_coeffs[o][i] = val;
            }
        }

        coeffs.resize(n_vars);
        for (int i = 0; i < n_vars; ++i)
        {
            knapsack >> val;
            coeffs[i] = val;
            order.push_back(i);
        }
        knapsack >> rhs;
        knapsack.close();

        // Copy variables in canonical order for future use
        obj_coeffs_canonical = obj_coeffs;
        coeffs_canonical = coeffs;

        // cout << "\nMultiobjective Knapsack Instance: " << endl;
        // cout << "\tnum_vars = " << n_vars << endl;
        // cout << "\trhs = " << rhs << endl;
        // cout << endl;
    }

    void reset_order(vector<int> new_order)
    {
        for (int i = 0; i < n_vars; i++)
        {
            order[i] = new_order[i];
            coeffs[i] = coeffs_canonical[new_order[i]];
            for (int o = 0; o < num_objs; o++)
            {
                obj_coeffs[o][i] = obj_coeffs_canonical[o][new_order[i]];
            }
        }
    }

    void reset_order()
    {
        for (int i = 0; i < n_vars; i++)
        {
            order[i] = i;
        }
        coeffs = coeffs_canonical;
        obj_coeffs = obj_coeffs_canonical;
    }

    void display()
    {
        cout << "******* Weight ********" << endl;
        for (int i = 0; i < n_vars; i++)
        {
            cout << coeffs[i] << " ";
        }
        cout << endl;
        cout << "******* Value ********" << endl;
        for (int i = 0; i < n_vars; i++)
        {
            for (int o = 0; o < num_objs; o++)
            {
                cout << obj_coeffs[o][i] << " ";
            }
            cout << endl;
        }
    }
};
