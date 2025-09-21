from anytree import Node, RenderTree
from anytree.exporter import DotExporter

def hier_visual(df:pd.DataFrame):
    # Er det noen fellestrekk i friteksten i enkelte grupper som kan beskrives?

    df_sorted = df.sort_values(by="level", ascending=True, inplace=False)

    
    # Creating the tree through a dictionary of nodes
    nodes = {}
    roots = []

    dummy_root = Node("AllRoots")
    for _, row in df_sorted.iterrows():
        if row["parentCode"] is np.nan:
            nodes[row["code"]] = Node(row["code"], parent=dummy_root, alias=row["name"])

            roots.append(nodes[row["code"]])
        else:
            nodes[row["code"]] = Node(
                row["code"], parent=nodes[row["parentCode"]], alias=row["name"]
            )

    roots = sorted(roots, key=lambda r: r.alias)

    # Printing hierarchy
    """
    for root in roots:
        for pre, fill, node in RenderTree(root):
            print(f"{pre}{node.alias}")
    """

    ###### Need to install the graphviz software in the virtual environment

    # Exporting tree to Graphviz and render PNG
    # DotExporter(dummy_root).to_picture("/home/stud-msh/stat-master-nace/src/data/forest.png")
    DotExporter(
        dummy_root,
        nodeattrfunc=lambda node: f'label="{getattr(node, "alias", node.name)}"',
    ).to_dotfile("forest.dot")

    import pydot

    (graph,) = pydot.graph_from_dot_file("forest.dot")
    graph.write_png("forest.png")
    Image("forest.png")  # displays in Jupyter/Colab


    # DotExporter(dummy_root,
    #            nodeattrfunc=lambda node: 'shape=box, style=filled, fillcolor=lightblue' if node.children else 'shape=ellipse, fillcolor=lightgreen',
    #            edgeattrfunc=lambda parent, child: 'color=gray').to_picture("forest_hidden_root.png")
    
    

def dis_visual(df:pd.DataFrame, hier:list[str]):
    ...