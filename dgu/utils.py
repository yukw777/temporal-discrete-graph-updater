import networkx as nx


def draw_graph(graph: nx.DiGraph, filename: str) -> None:
    agraph = nx.nx_agraph.to_agraph(graph)
    agraph.node_attr.update(
        fontsize="20pt", fontname="helvetica-bold", shape="plaintext"
    )
    agraph.edge_attr.update(
        fontsize="20pt", fontname="helvetica", color="gray", fontcolor="dodgerblue"
    )
    agraph.layout(prog="sfdp")
    agraph.draw(filename)
