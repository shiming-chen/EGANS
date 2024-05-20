import graphviz
import sys
from graphviz import Digraph
import ast
from operations import *
operation_name_list = list(operation_dict_all.keys())

operation_name_list=["FC+ReLU","FC+LeakyReLU","FC+ReLU+DropOut","FC+LeakyReLU+DropOut","None"]

def plot_genotype(opt, flag, genotype, file_name=None, figure_dir='./network_structure', save_figure=False):
    # Set graph style
    g = Digraph(
        format='pdf',
        edge_attr=dict(fontsize='20', fontname="Times New Roman"),
        node_attr=dict(style='rounded,filled', shape='box', align='center', fontsize='40', fontstyle='italic', height='0.8', width='0.8',
                       penwidth='1.5', fontname="Times New Roman"),
        engine='dot')
    g.body.extend(['rankdir=LR'])

    if flag == 'g':
        input_node_names = ['a', 'z', '[a,z]']
    else:
        input_node_names = ['a', 'x', '[a,x]']
    output_name = ' M4 '

    # All input nodes
    for input_node_name in input_node_names:
        g.node(input_node_name, fillcolor='deepskyblue')

    # Number of inner nodes
    nodes = opt.num_nodes

    # All inner nodes
    for i in range(nodes - 1):
        g.node(" M"+str(i)+" ", fillcolor='darkgoldenrod1')

    # Output node
    g.node(output_name, fillcolor='lightgreen')

    # Add edges
    # Edge direction: u ---> v
    # Genotype: operation, u

    for i in range(nodes):
        for j in range(i+3):
            op = operation_name_list[genotype[int((i+5)*i/2+j)].argmax()]
            if op != 'None':
                if j < len(input_node_names):
                    u = input_node_names[j]
                else:
                    u = " M"+str(j - len(input_node_names))+" "

                if i == nodes - 1:
                    v = output_name
                else:
                    v = " M"+str(i)+" "
                g.edge(u, v, label=op, fillcolor="black")

    # Save the figure
    if save_figure:
        g.render(file_name, view=False, directory=figure_dir)

    return g

if __name__ == '__main__':
  if len(sys.argv) != 3:
    print("usage:\n python {} ARCH_NAME".format(sys.argv[0]))
    sys.exit(1)
  cur_genotype_G = list(ast.literal_eval(sys.argv[1]))
  cur_genotype_D = list(ast.literal_eval(sys.argv[2]))
  print('generator', cur_genotype_G)
  print('discriminator', cur_genotype_D)
  # try:
  #   genotype = eval('genotypes.{}'.format(genotype_name))
  # except AttributeError:
  #   print("{} is not specified in genotypes.py".format(genotype_name))
  #   sys.exit(1)

  plot_genotype('g',
                cur_genotype_G,
                file_name='test_G',
                #          '%s_%s_%s' % \
                # (opt.figure_dir, opt.dataset, timestamp),
                save_figure=True
                )
  plot_genotype('d',
                cur_genotype_D,
                file_name='test_D',
                #          '%s_%s_%s' % \
                # (opt.figure_dir, opt.dataset, timestamp),
                save_figure=True
                )
  print('Figure saved.')