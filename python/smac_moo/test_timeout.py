import numpy as np
import time
from subprocess import Popen, PIPE

sol_list = [
    {"cpu_time": 0.22131657600402832, "wallclock_time": 0.26615333557128906, "evaluations": 1, "cost": 0.22131657600402832, "incumbent": {"max_avg_value": 1e-07, "max_avg_value_by_weight": 1e-07, "max_max_value": 1e-07,
                                                                                                                                          "max_max_value_by_weight": 1e-07, "max_min_value": 1e-07, "max_weight": 1e-07, "min_avg_value": 1e-07, "min_max_value": 1e-07, "min_min_value": 1e-07, "min_weight": 1.0}, "budget": 0, "origin": "Default"},
    {"cpu_time": 0.4305863380432129, "wallclock_time": 1.2671759128570557, "evaluations": 2, "cost": 0.20926976203918457, "incumbent": {"max_avg_value": 0.9864805508053621, "max_avg_value_by_weight": 0.21244808196184573, "max_max_value": 0.976734430531527, "max_max_value_by_weight": 0.30546490857652125,
                                                                                                                                        "max_min_value": 0.2708654061872524, "max_weight": 0.63526869656884, "min_avg_value": 0.2604354580952044, "min_max_value": 0.7357201731145612, "min_min_value": 0.42899697524423613, "min_weight": 0.19638144380709882}, "budget": 0, "origin": "Random Search (sorted)"},
    {"cpu_time": 0.8506476879119873, "wallclock_time": 2.443417549133301, "evaluations": 4, "cost": 0.20915842056274414, "incumbent": {"max_avg_value": 0.7019239526977877, "max_avg_value_by_weight": 0.20873595757184824, "max_max_value": 0.7945671735036132, "max_max_value_by_weight": 0.3472104068547753,
                                                                                                                                       "max_min_value": 0.4159947958977798, "max_weight": 0.6223468041858351, "min_avg_value": 0.13154007581215638, "min_max_value": 0.8001718158169787, "min_min_value": 0.6076638593054593, "min_weight": 0.33436532373249395}, "budget": 0, "origin": "Random Search (sorted)"},
    {"cpu_time": 1.2768123149871826, "wallclock_time": 2.9511256217956543, "evaluations": 6, "cost": 0.2088606357574463, "incumbent": {"max_avg_value": 0.27996767127569194, "max_avg_value_by_weight": 0.6385184555388886, "max_max_value": 0.9046552417580102, "max_max_value_by_weight": 0.1422668304913449,
                                                                                                                                       "max_min_value": 0.5379561689765489, "max_weight": 0.15498784319884262, "min_avg_value": 0.17432992710181405, "min_max_value": 0.6750249669546765, "min_min_value": 0.6330249940198421, "min_weight": 0.6062951558839448}, "budget": 0, "origin": "Random Search (sorted)"},
    {"cpu_time": 1.9043951034545898, "wallclock_time": 4.823843002319336, "evaluations": 9, "cost": 0.2073519229888916, "incumbent": {"max_avg_value": 0.40254215209999084, "max_avg_value_by_weight": 0.06884186902572423, "max_max_value": 0.8646122261007928, "max_max_value_by_weight": 0.2619907601879249,
                                                                                                                                      "max_min_value": 0.14924220754805465, "max_weight": 0.9563215974751369, "min_avg_value": 0.4261669762535863, "min_max_value": 0.3226028221382373, "min_min_value": 0.775700151048032, "min_weight": 0.2036067888405115}, "budget": 0, "origin": "Random Search (sorted)"},
    {"cpu_time": 2.741265296936035, "wallclock_time": 6.8404622077941895, "evaluations": 13, "cost": 0.2069993019104004, "incumbent": {"max_avg_value": 0.3203063989418208, "max_avg_value_by_weight": 0.4577198018631738, "max_max_value": 0.41329857560368416, "max_max_value_by_weight": 0.3745912557371728,
                                                                                                                                       "max_min_value": 0.37112825934008586, "max_weight": 0.9471424531726677, "min_avg_value": 0.26667278848358283, "min_max_value": 0.09955337905262512, "min_min_value": 0.7319656411614134, "min_weight": 0.19906713741636478}, "budget": 0, "origin": "Random Search (sorted)"},
    {"cpu_time": 3.9914889335632324, "wallclock_time": 9.638890504837036, "evaluations": 19, "cost": 0.2063460350036621, "incumbent": {"max_avg_value": 0.534088695938558, "max_avg_value_by_weight": 0.264153995107329, "max_max_value": 0.36808635937838463, "max_max_value_by_weight": 0.3955233864469996,
                                                                                                                                       "max_min_value": 0.017523265700626196, "max_weight": 0.9388707360912683, "min_avg_value": 0.09355109649539622, "min_max_value": 0.5465438811668405, "min_min_value": 0.8117407594998189, "min_weight": 0.25232739196809145}, "budget": 0, "origin": "Local Search"},
    {"cpu_time": 6.733092784881592, "wallclock_time": 15.513006925582886, "evaluations": 32, "cost": 0.2062222957611084, "incumbent": {"max_avg_value": 0.34911272504841157, "max_avg_value_by_weight": 0.7948333251274147, "max_max_value": 0.41454694442865864, "max_max_value_by_weight":
                                                                                                                                       0.6146579259856617, "max_min_value": 0.9063956234015471, "max_weight": 0.30497890770604474, "min_avg_value": 0.8895727435838767, "min_max_value": 0.0460996548847927, "min_min_value": 0.850662739425377, "min_weight": 0.6482629837572936}, "budget": 0, "origin": "Random Search"},
    {"cpu_time": 14.151797533035278, "wallclock_time": 30.85742163658142, "evaluations": 67, "cost": 0.20610785484313965, "incumbent": {"max_avg_value": 0.5780129469911313, "max_avg_value_by_weight": 0.8533235319696483, "max_max_value": 0.8485321685220514, "max_max_value_by_weight": 0.739457317520418,
                                                                                                                                        "max_min_value": 0.23639547175497463, "max_weight": 0.24274972816143622, "min_avg_value": 0.3795244615010901, "min_max_value": 0.1612512216812518, "min_min_value": 0.0023369272941027703, "min_weight": 0.16907747458044986}, "budget": 0, "origin": "Random Search"},
    {"cpu_time": 15.405239343643188, "wallclock_time": 34.127793312072754, "evaluations": 73, "cost": 0.2058699131011963, "incumbent": {"max_avg_value": 0.9987384932151023, "max_avg_value_by_weight": 0.019421143183137275, "max_max_value": 0.21958262394395592, "max_max_value_by_weight":
                                                                                                                                        0.28664238476296905, "max_min_value": 0.2972948138169626, "max_weight": 0.27953343247256357, "min_avg_value": 0.04704127609523253, "min_max_value": 0.6961140387404906, "min_min_value": 0.08758876037176244, "min_weight": 0.6288135792611657}, "budget": 0, "origin": "Random Search"},
    {"cpu_time": 43.63666343688965, "wallclock_time": 96.60679006576538, "evaluations": 207, "cost": 0.20546841621398926, "incumbent": {"max_avg_value": 0.5341919606091811, "max_avg_value_by_weight": 0.2378943598152729, "max_max_value": 0.9345969958936716, "max_max_value_by_weight":
                                                                                                                                        0.2805324979779812, "max_min_value": 0.07096874377438679, "max_weight": 0.8330405741312998, "min_avg_value": 0.1312820732347847, "min_max_value": 0.4554193956824089, "min_min_value": 0.8523228232607121, "min_weight": 0.2229650868263648}, "budget": 0, "origin": "Local Search"},
    {"cpu_time": 76.49684262275696, "wallclock_time": 168.76679110527039, "evaluations": 363, "cost": 0.20504236221313477, "incumbent": {"max_avg_value": 0.6223185321147799, "max_avg_value_by_weight": 0.2312255340716215, "max_max_value": 0.3269251471025832, "max_max_value_by_weight":
                                                                                                                                         0.8868915539921292, "max_min_value": 0.3395812492926487, "max_weight": 0.8358979065396629, "min_avg_value": 0.48629872300612964, "min_max_value": 0.16503705553486256, "min_min_value": 0.25711843965145265, "min_weight": 0.15616449290953216}, "budget": 0, "origin": "Local Search"},
    {"cpu_time": 240.1111409664154, "wallclock_time": 539.5372750759125, "evaluations": 1141, "cost": 0.2042832374572754, "incumbent": {"max_avg_value": 0.4500185954509764, "max_avg_value_by_weight": 0.47900710822884596, "max_max_value": 0.10105274422511748, "max_max_value_by_weight":
                                                                                                                                        0.14390745990629927, "max_min_value": 0.2909215164563521, "max_weight": 0.9182967088730671, "min_avg_value": 0.42199285253811414, "min_max_value": 0.525232096333076, "min_min_value": 0.1087829211786774, "min_weight": 0.9089081579717498}, "budget": 0, "origin": "Random Search"}
]
# instances = [f"./3_20/train/kp_7_3_20_{i}.dat" for i in range(250)]
# print(instances[0])

# cmd = f"./multiobj {instances[0]} ./sample_weighted.csv 1 1 0 0 0 0 0 0 0 0 0"

# io = Popen(cmd.split(" "), stdout=PIPE, stderr=PIPE)
# (stdout_, stderr_) = io.communicate()

# # print(stdout_, stderr_)
# stdout_str = stdout_.decode('utf-8')

# if len(stdout_str):
#     status, result = stdout_str.split(":")
#     print(result)
#     if status == "Solved":
#         print(np.sum(list(map(float, stdout_str.strip().split(',')[-3:]))))


instances = [f"3_20/train/kp_7_3_20_{i}.dat" for i in range(250)]

# wt_dict = {"max_avg_value": 0.4500185954509764,
#            "max_avg_value_by_weight": 0.47900710822884596,
#            "max_max_value": 0.10105274422511748,
#            "max_max_value_by_weight": 0.14390745990629927,
#            "max_min_value": 0.2909215164563521,
#            "max_weight": 0.9182967088730671,
#            "min_avg_value": 0.42199285253811414,
#            "min_max_value": 0.525232096333076,
#            "min_min_value": 0.1087829211786774,
#            "min_weight": 0.9089081579717498}

wt_lst = ['max_weight',
          'min_weight',
          'max_avg_value',
          'min_avg_value',
          'max_max_value',
          'min_max_value',
          'max_min_value',
          'min_min_value',
          'max_avg_value_by_weight',
          'max_max_value_by_weight']
for sol in sol_list:

    # print(sol_list[0]["incumbent"])
    wt_dict = sol["incumbent"]
    # wt_dict = {
    #     "max_avg_value": 0.11549915736265404,
    #     "max_avg_value_by_weight": 0.8621751314406639,
    #     "max_max_value": 0.0822246980956244,
    #     "max_max_value_by_weight": 0.8158566174992625,
    #     "max_min_value": 0.5339267511148671,
    #     "max_weight": 0.18449223611684668,
    #     "min_avg_value": 0.5411655660326893,
    #     "min_max_value": 0.8911638687639566,
    #     "min_min_value": 0.5239021574884885,
    #     "min_weight": 0.9382864770780639
    # }

    wt_str = " ".join([f'{wt_dict[k]}' for k in wt_lst])
    # print(wt_str)

    total_time = []
    for instance in instances:
        cmd = f"./multiobj {instance} ./dummy.csv 1"
        cmd = cmd + " " + wt_str

        # print(cmd)
        io = Popen(cmd.split(" "), stdout=PIPE, stderr=PIPE)
        (stdout_, stderr_) = io.communicate()

        # print(stdout_, stderr_)
        stdout_str = stdout_.decode('utf-8')
        if len(stdout_str):
            status, result = stdout_str.split(":")
            # print(instance, result)
            if status == "Solved":
                total_time.append(
                    np.sum(list(map(float, stdout_str.strip().split(',')[-3:]))))

    print(len(total_time), np.mean(total_time),
          np.std(total_time), np.median(total_time))
