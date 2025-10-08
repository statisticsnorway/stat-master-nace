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
    
    for root in roots:
        for pre, fill, node in RenderTree(root):
            print(f"{pre}{node.alias}")
    
    

def dis_visual(df:pd.DataFrame, hier:list[str], n=1):
    for i in range(len(hier)):
        node = hier[i]
        
        div_dist = df[node].value_counts()
        min_idx = div_dist.nsmallest(n).index
        max_idx = div_dist.nlargest(n).index
    
        print(div_dist)
        plt.plot(div_dist.index, div_dist.values, '.')
        plt.plot(min_idx, div_dist.loc[min_idx].values, '.', color='red', markersize=8, label="Lowest")
        plt.plot(max_idx, div_dist.loc[max_idx].values, '.', color='red', markersize=8, label="Highest")
        
        plt.xlabel(node)
        plt.title(f"Distribution of {node} groups")
        plt.show()
        
def dis_table(df:pd.DataFrame, hier:list[str], n=5):
    for i in range(len(hier)):
        node = hier[i]
        
        div_dist = df[node].value_counts()
        df_min = div_dist.nsmallest(n).reset_index()
        df_max = div_dist.nlargest(n).reset_index()
    
        df_min.columns = ['sn', 'nr of instances']
        df_max.columns = ['sn', 'nr of instances']
        
    return df_min, df_max

    
    
    
def group_count_visual(df:pd.DataFrame, hier:list[str], n=1):
    for i in range(len(hier)-1):
        parent = hier[i]
        child = hier[i+1]
        
        cnt_grps = df.groupby(parent)[child].nunique()
        min_idx = cnt_grps.nsmallest(n).index
        max_idx = cnt_grps.nlargest(n).index
        
        print(cnt_grps)
        plt.plot(cnt_grps.index, cnt_grps.values, '.')
        plt.plot(min_idx, cnt_grps.loc[min_idx].values, '.', color='red', markersize=8, label="Lowest")
        plt.plot(max_idx, cnt_grps.loc[max_idx].values, '.', color='red', markersize=8, label="Highest")

        plt.xlabel(parent)
        plt.title(f"Distribution of {node} groups")
        plt.show()
        

def group_count_table(df:pd.DataFrame, hier:list[str], n=5):
     for i in range(len(hier)-1):
        parent = hier[i]
        child = hier[i+1]
        
        cnt_grps = df.groupby(parent)[child].nunique()
        df_min = cnt_grps.nsmallest(n).reset_index()
        df_max = cnt_grps.nlargest(n).reset_index()
    
        df_min.columns = ['sn', 'nr of groups']
        df_max.columns = ['sn', 'nr of groups']
        
    return df_min, df_max

        
        
if __name__=='__main__':
    ...
        
        
        
