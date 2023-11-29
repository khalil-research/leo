/*
 * --------------------------------------------------------
 * AbsVal Instance
 * --------------------------------------------------------
 */

#ifndef ABSVAL_INSTANCE_HPP_
#define ABSVAL_INSTANCE_HPP_

#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

//
// AbsVal Instance
//
struct AbsValInstance
{
    // We read a and b vectors
    // Original problem: min{ |A_1^T x - b_1| , |A_2^T x - b_2| , ... , |A_P^T x - b_P| }
    //                   s.t.   1^T x <= floor(n * cardinality_ratio)
    //                          x in {0,1}^n

    // Reformulation: min{ y_1 , y_2 , ... , y_P}
    //                s.t. y_j >= A_j^T x - b_j,    j=1,...,P
    //                     y_j >= -(A_j^T x - b_j), j=1,...,P
    //                     1^T x <= floor(n * cardinality_ratio)
    //                     x binary
    //                     y integer

    // Number of x variables
    int n_xvars;
    // Number of y variables
    int n_yvars;

    // The ratio for the cardinality constraint
    double cardinality_ratio;

    // Constraint coefficients (indexed by constraint/variable)
    vector<vector<int>> A;
    vector<vector<int>> A_canonical;

    // Right-hand sides
    vector<int> b;

    // Constructors
    AbsValInstance() {}

    // Read instance based on BDD format
    void read_BDD(const char *inputfile);

    void reset_order(vector<int> new_order);

    // Write instance based on Kirlik format
    void write_Kirlik(char *filename);

    // Write instance based on Ozlen format
    void write_Ozlen(char *filename);

    // Write instance based on the format of the Rectangle Splitting algorithm
    void write_Biobj(char *filename);
};

//
// Read BDD input format
//
inline void AbsValInstance::read_BDD(const char *inputfile)
{
    ifstream input(inputfile);
    if (!input.is_open())
    {
        cout << "Error: could not open file " << inputfile << endl;
        exit(1);
    }

    input >> n_yvars; // i.e., # objs
    input >> n_xvars; // i.e., # binary variables
    input >> cardinality_ratio;

    int val;

    for (int p = 0; p < n_yvars; p++)
    {
        A.push_back(vector<int>(n_xvars));
        for (int i = 0; i < n_xvars; ++i)
        {
            input >> val;
            A[p][i] = val;
        }
        input >> val;
        b.push_back(val);
    }
    input.close();

    A_canonical = A;

    // cout << "\nMultiobjective AbsVal Instance" << endl;
    // cout << "\tnum of y variables = " << n_yvars << endl;
    // cout << "\tnum of x variables = " << n_xvars << endl;
    // cout << endl;
}

inline void AbsValInstance::reset_order(vector<int> new_order)
{
    for (int i = 0; i < n_xvars; ++i)
    {
        for (int p = 0; p < n_yvars; p++)
        {
            A[p][i] = A_canonical[p][new_order[i]];
        }
    }

    for (int p = 0; p < n_yvars; p++)
    {
        for (int i = 0; i < n_xvars; ++i)
        {
            cout << A_canonical[p][i] << " ";
        }
        cout << endl;
    }
    cout << endl;

    for (int p = 0; p < n_yvars; p++)
    {
        for (int i = 0; i < n_xvars; ++i)
        {
            cout << A[p][i] << " ";
        }
        cout << endl;
    }
}

#endif