// ----------------------------------------------------------
// Portfolio Optimization Instance
// ----------------------------------------------------------

#ifndef PORTFOLIO_OPT_INSTANCE_HPP_
#define PORTFOLIO_OPT_INSTANCE_HPP_

#include <vector>
#include <ilcplex/ilocplex.h>

using namespace std;


//
// Round numbers up to three decimal places
//
inline double roundup3(double n) {
    double t;
    t = n - floor(n);
    if (t >= 0.5) {
        n *= 1000; //1000 since we want two decimal points
        n = ceil(n);
        n /= 1000;
    }
    else {
        n *= 1000;
        n = floor(n);
        n /= 1000;
    }
    return n;
}



//
// Portfolio Instance
//
struct PortfolioInstance {
    // We read mu and sigma_sq vectors
    // Original problem: max{ mu^T x , - ((sigma^2)^T x})^{1/2} , ((gamma^3)^T x)^{1/2} , - ((beta^4)^T x})^{1/4} } s.t. a^T x <= b, x binary

    // Number of x variables
    int n_vars;
    // mu
    vector<int> mu;
    // sigma_sq
    vector<int> sigma_sq;
    // gamma_cube
    vector<int> gamma_cube;
    // beta_fourth
    vector<int> beta_fourth;
    // Constraint coefficients
    vector<double> a;
    // Right-hand sides
    double b;
    
    // Constructors
    PortfolioInstance() {
        assert(NOBJS <= 4);
    }
    
    // Read instance based on BDD format
    void read_BDD(const char* inputfile);
    
    // Print Instance
    void print();
};

#endif

