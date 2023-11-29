// ----------------------------------------------------------
// Knapsack Instance - Implementations
// ----------------------------------------------------------

#include <algorithm>
#include <fstream>
#include <iostream>
#include <cstdlib>

#include "../bdd/portfolio_opt_bdd.hpp"

using namespace std;


//
// Read instance based on our format
//
void PortfolioInstance::read_BDD(const char* inputfile) {
    ifstream input(inputfile);
    if (!input.is_open()) {
        cout << "Error: could not open file " << inputfile << endl;
        exit(1);
    }
    
    input >> n_vars;
    
    int val;
    
    a.resize(n_vars);
    for (int i = 0; i < n_vars; ++i) {
        input >> val;
        a[i] = val;
    }
    
    input >> b;
    
    mu.resize(n_vars);
    for (int i = 0; i < n_vars; ++i) {
        input >> val;
        mu[i] = val;
    }
    
    sigma_sq.resize(n_vars);
    for (int i = 0; i < n_vars; ++i) {
        input >> val;
        sigma_sq[i] = val;
    }
    
    gamma_cube.resize(n_vars);
    for (int i = 0; i < n_vars; ++i) {
        input >> val;
        gamma_cube[i] = val;
    }
    
    beta_fourth.resize(n_vars);
    for (int i = 0; i < n_vars; ++i) {
        input >> val;
        beta_fourth[i] = val;
    }
    
    // Print instance
    print();
    
    cout << "\nMultiobjective Portfolio Instance" << endl;
    cout << "\tnum of x variables = " << n_vars << endl;
    cout << endl;
}

//
// Print instance
//
void PortfolioInstance::print() {
    cout << "\nPortfolio Instance: " << endl;
    cout << "\tnum variables: " << n_vars << endl;
    cout << "\tnum objs: " << NOBJS << endl;
    cout << endl;
    
    cout << "mu: ";
    for(int i=0; i < n_vars; i++)
        cout << mu[i] << " ";
    cout << endl;
   
    if(NOBJS >= 2) {
        cout << "sigma square: ";
        for(int i=0; i < n_vars; i++)
            cout << sigma_sq[i] << " ";
        cout << endl;
    }
    
    if(NOBJS >= 3) {
        cout << "gamma cube: ";
        for(int i=0; i < n_vars; i++)
            cout << gamma_cube[i] << " ";
        cout << endl;
    }
    
    if(NOBJS >= 4) {
        cout << "beta fourth: ";
        for(int i=0; i < n_vars; i++)
            cout << beta_fourth[i] << " ";
        cout << endl;
    }
    
    cout << "Constraint: ";
    for (int i = 0; i < n_vars; ++i)
        cout << a[i] << " ";
    cout << " <= " << b << endl;
}





