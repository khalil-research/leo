from pathlib import Path
import numpy as np
import pandas as pd
import csv


from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
from operator import itemgetter

order_type = ['max_weight',
              'min_weight',
              'max_avg_value',
              'min_avg_value',
              'max_max_value',
              'min_max_value',
              'max_min_value',
              'min_min_value',
              'max_avg_value_by_weight',
              'max_max_value_by_weight']


def parse_output(filepath):
    result = {}
    with open(filepath) as fp:
        outs = csv.reader(fp, delimiter=',')

        for out in outs:
            out = [o.strip() for o in out]
            prob, s, p, n, inst = out[0].split(
                "/")[-1].split(".")[0].split('_')
            # prob, s, n, p, inst = out[0].split("/")[-1].split(".")[0].split('-')
            otype = out[1]
            iwd = int(out[2])
            rwd = int(out[3])
            isz = int(out[4])
            rsz = int(out[5])
            ctime = float(out[6])
            rtime = float(out[7])
            ptime = float(out[8])
            ttime = ctime + rtime + ptime
            modz = int(out[9])
            order = list(map(int, out[10].split("|")[:-1]))

            if prob not in result:
                result[prob] = {}

            if s not in result[prob]:
                result[prob][s] = {}

            if p not in result[prob][s]:
                result[prob][s][p] = {}

            if n not in result[prob][s][p]:
                result[prob][s][p][n] = {}

            if inst not in result[prob][s][p][n]:
                result[prob][s][p][n][inst] = {}

            if otype not in result[prob][s][p][n][inst]:
                result[prob][s][p][n][inst][otype] = {
                    'iwd': iwd, 'rwd': rwd, 'isz': isz, 'rsz': rsz,
                    'ctime': ctime, 'rtime': rtime, 'ptime': ptime, 'ttime': ttime,
                    'order': order
                }

    return result


def get_scatter_data(data, prob='kp', seed='7', n='20', p='3', order='max_weight', qty='ttime'):
    d = data[prob][seed][p][n]
    # print(d)
    vals = {}

    insts = list(d.keys())
    insts.sort()
    for i in insts:
        if order == 'rnd':
            nruns = [1 if 'rnd'+str(j) in d[i] else 0 for j in range(5)]
            if np.sum(nruns) == 5:
                val = [d[i][f'{order}{j}'][qty] for j in range(5)]
                val = np.mean(val)

                vals[i] = val

        else:
            if order in d[i]:
                val = d[i][order][qty]
                vals[i] = val

    return vals


def main():
    result = parse_output('out_final.csv')

    # Fetch times
    rnd_vals = get_scatter_data(result, prob='kp', seed='7', n='20',
                                p='3', order='rnd', qty='rsz')
    # print(list(rnd_vals.keys))
    rnd_lst = [rnd_vals[str(i)] for i in range(250)]
    ots_lst = []
    order_rankings = []

    for ot in order_type:
        ot_vals = get_scatter_data(result, prob='kp', seed='7', n='20',
                                   p='3', order=ot, qty='rsz')
        # ckeys = set(rnd_vals.keys()).intersection(set(ot_vals.keys()))
        # ckeys = list(ckeys)
        # ckeys.sort()

        ots_lst.append([ot_vals[str(i)] for i in range(250)])
        order_rankings.append((ot, np.mean(rnd_lst)/np.mean(ots_lst[-1])))

    order_rankings = sorted(order_rankings, key=itemgetter(1))
    for t in order_rankings:
        print(t[0], "\t", t[1])

    # # Add traces
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=ots_lst[0], y=rnd_lst,
                             mode='markers',
                             name=order_type[0]))

    fig.add_trace(go.Scatter(x=ots_lst[1], y=rnd_lst,
                             mode='markers',
                             name=order_type[1]))

    fig.add_trace(go.Scatter(x=ots_lst[2], y=rnd_lst,
                             mode='markers',
                             name=order_type[2]))

    fig.add_trace(go.Scatter(x=ots_lst[3], y=rnd_lst,
                             mode='markers',
                             name=order_type[3]))

    fig.add_trace(go.Scatter(x=ots_lst[4], y=rnd_lst,
                             mode='markers',
                             name=order_type[4]))

    fig.add_trace(go.Scatter(x=ots_lst[5], y=rnd_lst,
                             mode='markers',
                             name=order_type[5]))

    fig.add_trace(go.Scatter(x=ots_lst[6], y=rnd_lst,
                             mode='markers',
                             name=order_type[6]))

    fig.add_trace(go.Scatter(x=ots_lst[7], y=rnd_lst,
                             mode='markers',
                             name=order_type[7]))

    fig.add_trace(go.Scatter(x=ots_lst[8], y=rnd_lst,
                             mode='markers',
                             name=order_type[8]))

    fig.add_trace(go.Scatter(x=ots_lst[9], y=rnd_lst,
                             mode='markers',
                             name=order_type[9]))

    fig.show()


if __name__ == '__main__':
    main()
