import seaborn as sns
import proplot as pplt
from matplotlib.colors import LinearSegmentedColormap


def plot_xy_correlation(ax, data, x=None, y=None, r=None, p=None, corr_type='pearson', 
                        spine_color='black', spine_linewidth=None, colors=None, truncate=False, annot=True):
    # r, p = perm_correlation(np.array(data[x]), np.array(data[y]), 10000, type=corr_type)
    if colors is None:
        colors =  ['#1E90FF', '#EEEEEE', '#FF7F00']
    
    if p > 0.05:
        cmap = pplt.Colormap(LinearSegmentedColormap.from_list("custom_cmap", ['#FFFFFF'] + ['#9C9C9C']))
        linecolor = '#696969'	
    else:
        if r > 0:
            cmap = pplt.Colormap(LinearSegmentedColormap.from_list("custom_cmap", ['#FFFFFF'] + [colors[-1]]))
            linecolor = colors[-1]
        else:
            cmap = pplt.Colormap(LinearSegmentedColormap.from_list("custom_cmap", ['#FFFFFF'] + [colors[0]]))
            linecolor = colors[0]
    
    sns.kdeplot(y=y, x=x, data=data, fill=True, thresh=0.01, levels=100, cmap=cmap, ax=ax)
    sns.regplot(y=y, x=x, data=data, robust=True, 
                line_kws={'color':linecolor, 'lw': 1}, 
                scatter_kws={'s': 0}, ax=ax, truncate=truncate)
    
    if corr_type == 'pearson':
        prefix = 'r'
    elif corr_type == 'spearman':
        prefix = 'rho'
    
    if annot:
        if p < 0.01:
            ax.format(urtitle='${}$ = {:.2f}\n$p$ < 0.01'.format(prefix, r))
        else:
            ax.format(urtitle='${}$ = {:.2f}\n$p$ = {:.2f}'.format(prefix, r, p))

    if spine_linewidth is not None:
        ax.spines['bottom'].set_linewidth(spine_linewidth)
        ax.spines['top'].set_linewidth(spine_linewidth)
        ax.spines['right'].set_linewidth(spine_linewidth) 
        ax.spines['left'].set_linewidth(spine_linewidth) 
    if spine_color is not None:
        ax.spines['bottom'].set_color(spine_color)  
        ax.spines['top'].set_color(spine_color)
        ax.spines['left'].set_color(spine_color)
        ax.spines['right'].set_color(spine_color)
      
