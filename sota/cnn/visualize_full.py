import sys
import genotypes
import numpy as np
from graphviz import Digraph


supernet_dict = {
    0:  ('c_{k-2}', '0'),
    1:  ('c_{k-1}', '0'),
    2:  ('c_{k-2}', '1'),
    3:  ('c_{k-1}', '1'),
    4:  ('0', '1'),
    5:  ('c_{k-2}', '2'),
    6:  ('c_{k-1}', '2'),
    7:  ('0', '2'),
    8:  ('1', '2'),
    9:  ('c_{k-2}', '3'),
    10: ('c_{k-1}', '3'),
    11: ('0', '3'),
    12: ('1', '3'),
    13: ('2', '3'),
}
steps = 4

def plot_space(primitives, filename):
    g = Digraph(
        format='pdf',
        edge_attr=dict(fontsize='20', fontname="times"),
        node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
        engine='dot')
    g.body.extend(['rankdir=LR'])
    g.body.extend(['ratio=50.0'])

    g.node("c_{k-2}", fillcolor='darkseagreen2')
    g.node("c_{k-1}", fillcolor='darkseagreen2')

    steps = 4

    for i in range(steps):
        g.node(str(i), fillcolor='lightblue')

    n = 2
    start = 0
    nodes_indx = ["c_{k-2}", "c_{k-1}"]
    for i in range(steps):
        end = start + n
        p = primitives[start:end]
        v = str(i)
        for node, prim in zip(nodes_indx, p):
            u = node
            for op in prim:
                g.edge(u, v, label=op, fillcolor="gray")

    start = end
    n += 1
    nodes_indx.append(v)

    g.node("c_{k}", fillcolor='palegoldenrod')
    for i in range(steps):
        g.edge(str(i), "c_{k}", fillcolor="gray")

    g.render(filename, view=False)


def plot(genotype, filename):
    g = Digraph(
        format='pdf',
        edge_attr=dict(fontsize='100', fontname="times"),
        node_attr=dict(style='filled', shape='rect', align='center', fontsize='100', height='0.5', width='0.5', penwidth='2', fontname="times"),
        engine='dot')
    g.body.extend(['rankdir=LR'])
    g.body.extend(['ratio=0.3'])

    g.node("c_{k-2}", fillcolor='darkseagreen2')
    g.node("c_{k-1}", fillcolor='darkseagreen2')
    num_edges = len(genotype)

    for i in range(steps):
        g.node(str(i), fillcolor='lightblue')

    for eid in range(num_edges):
        op = genotype[eid]
        u, v = supernet_dict[eid]
        if op != 'skip_connect':
            g.edge(u, v, label=op, fillcolor="gray", color='red', fontcolor='red')
        else:
            g.edge(u, v, label=op, fillcolor="gray")

    g.node("c_{k}", fillcolor='palegoldenrod')
    for i in range(steps):
        g.edge(str(i), "c_{k}", fillcolor="gray")

    g.render(filename, view=False)



# def plot(genotype, filename):
#     g = Digraph(
#         format='pdf',
#         edge_attr=dict(fontsize='100', fontname="times", penwidth='3'),
#         node_attr=dict(style='filled', shape='rect', align='center', fontsize='100', height='0.5', width='0.5',
#                        penwidth='2', fontname="times"),
#         engine='dot')
#     g.body.extend(['rankdir=LR'])

#     g.node("c_{k-2}", fillcolor='darkseagreen2')
#     g.node("c_{k-1}", fillcolor='darkseagreen2')
#     num_edges = len(genotype)

#     for i in range(steps):
#         g.node(str(i), fillcolor='lightblue')

#     for eid in range(num_edges):
#         op = genotype[eid]
#         u, v = supernet_dict[eid]
#         if op != 'skip_connect':
#             g.edge(u, v, label=op, fillcolor="gray", color='red', fontcolor='red')
#         else:
#             g.edge(u, v, label=op, fillcolor="gray")

#     g.node("c_{k}", fillcolor='palegoldenrod')
#     for i in range(steps):
#         g.edge(str(i), "c_{k}", fillcolor="gray")

#     g.render(filename, view=False)


if __name__ == '__main__':
    #### visualize the supernet ####
    if len(sys.argv) != 2:
        print("usage:\n python {} ARCH_NAME".format(sys.argv[0]))
        sys.exit(1)

    genotype_name = sys.argv[1]
    assert 'supernet' in genotype_name, 'this script only supports supernet visualization'
    try:
        genotype = eval('genotypes.{}'.format(genotype_name))
    except AttributeError:
        print("{} is not specified in genotypes.py".format(genotype_name))
        sys.exit(1)

    path = '../../figs/genotypes/cnn_supernet_cue/'
    plot(genotype.normal, path + genotype_name + "_normal")
    plot(genotype.reduce, path + genotype_name + "_reduce")
