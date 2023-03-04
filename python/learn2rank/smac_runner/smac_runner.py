from pathlib import Path

import numpy as np
from smac.configspace import ConfigurationSpace
from smac.facade.smac_ac_facade import SMAC4AC
from smac.runhistory.runhistory import RunHistory
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.utils.io.traj_logging import TrajLogger


class SMACRunner:
    def __init__(self, cfg=None):
        assert cfg is not None

        self.cfg = cfg
        self.cs = ConfigurationSpace()
        self.root_pkg = Path(__file__).parent.parent

        # Define scenario
        self.base_scenario_dict = {
            "cs": self.cs,
            "deterministic": "true",
            "run_obj": "runtime",
            "cutoff_time": self.cfg.cutoff_time if self.cfg.restore_run == 0 else self.cfg.new_cutoff_time,
            "wallclock_limit": self.cfg.wallclock_limit if self.cfg.restore_run == 0 else self.cfg.new_wallclock_limit,
            "algo": f"python {str(self.root_pkg)}/scripts/smac_ta.py"
        }

    def initialize_config_space(self):
        raise NotImplementedError

    def set_config_space(self):
        raise NotImplementedError

    def load_old_run(self, scenario, old_run_dir):
        path = Path(self.cfg.res_path[self.cfg.machine])
        path = path / 'smac_output' if self.cfg.mode == 'one' else path / 'smac_all_output'
        path = path / self.cfg.problem.name / f'iinc_{self.cfg.init_incumbent}' / self.cfg.problem.size / self.cfg.split
        path = path / old_run_dir / f'run_{self.cfg.seed}'

        # Update old scenario
        # new_scenario = Scenario(path / 'scenario.txt')
        # scenario.cutoff_time = self.cfg.new_cutoff_time
        # scenario.wallclock_limit = self.cfg.new_wallclock_limit

        # Load the runhistory
        rh_path = path / "runhistory.json"
        runhistory = RunHistory()
        runhistory.load_json(str(rh_path), scenario.cs)

        # And the stats
        stats_path = path / "stats.json"
        stats = Stats(scenario)
        stats.load(stats_path)

        # And the trajectory
        traj_path = path / "traj_aclib2.json"
        trajectory = TrajLogger.read_traj_aclib_format(fn=traj_path, cs=scenario.cs)
        incumbent = trajectory[-1]["incumbent"]

        return runhistory, stats, incumbent

    def run(self, instances):
        assert len(instances)

        # Set instances
        scenario_dict = self.base_scenario_dict.copy()
        scenario_dict['instances'] = instances
        # Set output directory
        output_dir = str(Path(instances[0][0]).stem)
        scenario_dict['output_dir'] = output_dir
        # Create scenario object
        scenario = Scenario(scenario_dict,
                            cmd_options={
                                "wallclock_limit": self.cfg.new_wallclock_limit,  # overwrite these args
                                "cutoff_time": self.cfg.new_cutoff_time,
                                "output_dir": output_dir,
                                "output_dir_for_this_run": f"{output_dir}/run_{self.cfg.seed}"
                            })

        # Load old run, if need be
        if self.cfg.restore_run == 1:
            runhistory, stats, incumbent = self.load_old_run(scenario, output_dir)
        else:
            runhistory, stats, incumbent = None, None, None

        # Create smac object
        smac = SMAC4AC(scenario=scenario,
                       runhistory=runhistory,
                       stats=stats,
                       restore_incumbent=incumbent,
                       rng=np.random.RandomState(self.cfg.seed),
                       run_id=self.cfg.seed + 10)

        # Start optimization
        try:
            incumbent = smac.optimize()
        finally:
            incumbent = smac.solver.incumbent

        print("Optimized configuration %s" % str(incumbent))
