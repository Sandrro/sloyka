import networkx as nx
import hypernetx as hnx
import matplotlib.pyplot as plt

def make_hyper_graph(g):
    dd = nx.to_dict_of_dicts(g)
    H = hnx.Hypergraph(dd)
    return H

def draw_hyper_graph(g):
    kwargs = {'layout_kwargs': {'seed': 42}, 'with_node_counts': False, "node_radius":.1, "with_edge_labels":False, 'with_node_labels':True, 'contain_hyper_edges':False, 
          }
    plt.figure(figsize=(30, 20))
    hnx.draw(g.remove_singletons().dual(), **kwargs, node_labels_kwargs={
            'fontsize': 5
        }
    )