from anytree import Node, RenderTree
from anytree.exporter import DotExporter

def dis_visual(df:pd.DataFrame, hier:list[str], n=1):
    for i in range(len(hier)):
        node = hier[i]
        
        div_dist = df[node].value_counts()
        min_idx = div_dist.nsmallest(n).index
        max_idx = div_dist.nlargest(n).index
    
        div_dist = div_dist.sort_index()
        
        print(div_dist)
        plt.scatter(div_dist.index, div_dist.values, label="All values")
        plt.scatter(min_idx, div_dist.loc[min_idx], color='red', label="Lowest")
        plt.scatter(max_idx, div_dist.loc[max_idx], color='green', label="Highest")
        plt.legend()
        plt.xlabel(node)
        plt.title(f"Distribution of {node}")
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
        
        cnt_grps=cnt_grps.sort_index()
        
        print(cnt_grps)
        plt.scatter(cnt_grps.index, cnt_grps.values, label="All values")
        plt.scatter(min_idx, cnt_grps.loc[min_idx], color='red', label="Lowest")
        plt.scatter(max_idx, cnt_grps.loc[max_idx], color='green', label="Highest")
        plt.xlabel(parent)
        plt.title(f"Count of splits in {parent}")
        plt.legend()
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
        
        
        
