// ----------------------------------------------------------
// Knapsack Instance
// ----------------------------------------------------------

#ifndef KNAPSACK_INSTANCE_HPP_
#define KNAPSACK_INSTANCE_HPP_

#include <cassert>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <fstream>

#include "../bdd.hpp"
#include "knapsack_instance.hpp"

using namespace std;

//
// Multiobjective knapsack problem
//
struct MultiObjKnapsackInstance
{
	// Number of variables
	int n_vars;
	// Number of objective functions
	int num_objs;
	// Objective function coefficients
	vector<vector<int>> obj_coeffs;
	// Constraint coefficients
	vector<int> coeffs;
	// Right-hand side
	int rhs;

	//
	// Constructor
	//
	MultiObjKnapsackInstance(char *inputfile)
	{
		ifstream knapsack(inputfile);
		if (!knapsack.is_open())
		{
			cout << "Error: could not open file " << inputfile << endl;
			exit(1);
		}

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
		}
		knapsack >> rhs;
		knapsack.close();

		// cout << "\nMultiobjective Knapsack Instance: " << endl;
		// cout << "\tnum_vars = " << n_vars << endl;
		// cout << "\trhs = " << rhs << endl;
		// cout << endl;
	}

	//
	// Generate instance LP
	//
	void createLP_Melihozen(string lpfilename)
	{
		ofstream out(lpfilename.c_str());
		out << "maximize 0";
		out << endl;
		out << "subject to" << endl;
		out << "\\Capacity Constraint" << endl;
		out << coeffs[0] << " x" << 0;
		for (int i = 1; i < n_vars; ++i)
		{
			out << " + " << coeffs[i] << " x" << i;
		}
		out << " <= " << rhs << endl;
		out << endl;

		for (int o = 0; o < num_objs; ++o)
		{
			out << "\\New objective is defined" << endl;
			out << obj_coeffs[o][0] << " x" << 0;
			for (int i = 1; i < n_vars; ++i)
			{
				out << " + " << obj_coeffs[o][i] << " x" << i << " ";
			}
			out << " > " << (o + 1) << endl
				<< endl;
		}

		out << "\\Integer constraints" << endl;
		out << "binaries" << endl;
		out << " ";
		for (int i = 0; i < n_vars; ++i)
		{
			out << "x" << i << " ";
		}
		out << endl;

		out << "end" << endl;

		out.close();
	}

	//
	// Generate instance LP - Kirlik model
	//
	void createLP_Kirlik(string lpfilename)
	{
		ofstream out(lpfilename.c_str());
		out << "\\" << num_objs << endl;
		out << "Minimize\n obj: z";
		out << endl;
		out << "Subject To" << endl;

		for (int o = 0; o < num_objs; ++o)
		{
			out << "\\New objective is defined" << endl;
			for (int i = 0; i < n_vars; ++i)
			{
				out << " - " << obj_coeffs[o][i] << " x" << i << " ";
			}
			out << " <= " << 1 << endl
				<< endl;
		}

		out << "\\Capacity Constraint" << endl;
		out << coeffs[0] << " x" << 0;
		for (int i = 1; i < n_vars; ++i)
		{
			out << " + " << coeffs[i] << " x" << i;
		}
		out << " <= " << rhs << endl;
		out << endl;

		out << "Bounds" << endl;
		out << " z = 1" << endl;
		for (int i = 0; i < n_vars; ++i)
		{
			out << "0 <= x" << i << " <= 1" << endl;
		}

		out << "\\Integer constraints" << endl;
		out << "binaries" << endl;
		out << " ";
		for (int i = 0; i < n_vars; ++i)
		{
			out << "x" << i << " ";
		}
		out << endl;

		out << "end" << endl;

		out.close();
	}
};

//
// Knapsack Instance
//
struct KnapsackInstance
{
	// Number of variables
	int n_vars;
	// Objective function coefficients
	vector<int> obj_coeffs;
	// Constraint coefficients
	vector<int> coeffs;
	// Right-hand side
	int rhs;

	// Create instance from parameters
	KnapsackInstance(int _n_vars, vector<int> &_obj, vector<int> &_coeffs, int _rhs)
		: n_vars(_n_vars), obj_coeffs(_obj), coeffs(_coeffs), rhs(_rhs) {}

	// Read instance from file
	KnapsackInstance(char *inputfile)
	{
		ifstream knapsack(inputfile);
		if (!knapsack.is_open())
		{
			cout << "Error: could not open file " << inputfile << endl;
			exit(1);
		}

		knapsack >> n_vars;
		int val;

		obj_coeffs.resize(n_vars);
		for (int i = 0; i < n_vars; ++i)
		{
			knapsack >> val;
			obj_coeffs[i] = val;
		}

		coeffs.resize(n_vars);
		for (int i = 0; i < n_vars; ++i)
		{
			knapsack >> val;
			coeffs[i] = val;
		}
		knapsack >> rhs;

		knapsack.close();
	}
};

#endif
