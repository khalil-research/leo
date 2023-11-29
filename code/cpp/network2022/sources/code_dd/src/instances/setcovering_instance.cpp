/*
 * --------------------------------------------------------
 * Set Covering instance - Implementations
 * --------------------------------------------------------
 */


#include <ilcp/cpext.h>
#include "setcovering_instance.hpp"


// Apply heuristic to minimize bandwidth
void SetCoveringInstance::minimize_bandwidth() {
	try {
		//cout << "Preprocessing: Minimizing bandwidth..." << endl;

		int total_time = 5;
		if (n_vars >= 100) {
			total_time = n_vars * 0.1;
		}
		//cout << "\tTime limit: " << total_time << endl;


		IloEnv env;
		IloModel model(env);

		IloIntVarArray x(env, n_vars, 0, n_vars-1);
		model.add( IloAllDiff(env, x) );

		IloIntVar obj(env, 0, n_vars*n_vars);
		for (int c = 0; c < n_cons; ++c) {
			for (size_t i = 0; i < vars_cons[c].size(); ++i) {
				for (size_t j = i+1; j < vars_cons[c].size(); ++j) {
					model.add( obj >= IloAbs( x[ vars_cons[c][i] ] - x[ vars_cons[c][j] ]) );
				}
			}
		}
		model.add(IloMinimize(env, obj));

		// set starting point
		IloSolution solution(env);
		for (int i = 0; i < n_vars; ++i) {
			solution.add(x[i]);
			solution.setValue(x[i], i);
		}

		IloCP cp(model);
		cp.setParameter(IloCP::DefaultInferenceLevel, IloCP::Extended);
		cp.setParameter(IloCP::Workers, 1);
		cp.setParameter(IloCP::TimeLimit, total_time);
		cp.setParameter(IloCP::LogVerbosity, IloCP::Quiet);
		cp.setStartingPoint(solution);

		IloSearchPhaseArray phase(env);
		IloIntVarArray objs_v(env);
		objs_v.add(obj);
		phase.add( IloSearchPhase(env, objs_v) );
		//	   IloSelectSmallest(IloVarIndex(env, objs_v)),
		//			      IloSelectSmallest(IloValueSuccessRate(env))
		//			      ));

		//phase.add( IloSearchPhase(env, x) );

		//cp.solve(phase);
		cp.solve();
		bandwidth = cp.getObjValue();

		vector< vector<int> > new_vars_cons(vars_cons.size());
		vector< vector<int> > new_cons_var(n_vars);

		for (int c = 0; c < n_cons; ++c) {
			for (size_t i = 0; i < vars_cons[c].size(); ++i) {
				new_vars_cons[c].push_back( cp.getValue(x[vars_cons[c][i]]) );
				new_cons_var[ cp.getValue(x[vars_cons[c][i]]) ].push_back(c);
			}
		}

		vector< vector<int> > new_objs(n_objs,
				vector<int>(n_vars, 0));
		for (int p = 0; p < n_objs; ++p) {
			for (int j = 0; j < n_vars; ++j) {
				new_objs[p][cp.getValue(x[j])] = objs[p][j];
			}
		}

		objs = new_objs;
		vars_cons = new_vars_cons;
		cons_var = new_cons_var;

		//cout << "\tdone" << endl;

	} catch (IloException &ex) {
		cout << "Error: " << ex << endl;
	}
}
