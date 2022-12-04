from ..tools import multiple_line_plot
from numpy import array, arange

def compare_irfs(ARM,
                 FRM,
                 plot_in_deviations = False,
                 T_max = None):
    # houseownership
    H_share_1 = ARM.H_share_vec
    H_share_2 = FRM.H_share_vec
    # average asset holdings
    ave_a_1 = ARM.ave_a_vec
    ave_a_2 = FRM.ave_a_vec
    # average mortgage rate
    ave_rm_1 = ARM.ave_rm_vec
    ave_rm_2 = FRM.ave_rm_vec
    # average default prob
    ave_Pd_1 = ARM.ave_Pd_vec
    ave_Pd_2 = FRM.ave_Pd_vec
    # average purchase prob
    ave_Pp_1 = ARM.ave_Pp_vec
    ave_Pp_2 = FRM.ave_Pp_vec
    # average purchase prob
    ave_Pp_1 = ARM.ave_Pp_vec
    ave_Pp_2 = FRM.ave_Pp_vec
    # inequality in asset holdings
    gini_a_1 = ARM.gini_a_vec
    gini_a_2 = FRM.gini_a_vec
    # inequality in mortgage rate
    gini_rm_1 = ARM.gini_rm_vec
    gini_rm_2 = FRM.gini_rm_vec
    # inequality in default prob
    gini_Pd_1 = ARM.gini_Pd_vec
    gini_Pd_2 = FRM.gini_Pd_vec
    # inequality in purchase prob
    gini_Pp_1 = ARM.gini_Pp_vec
    gini_Pp_2 = FRM.gini_Pp_vec
    # inequality in purchase prob
    gini_Pp_1 = ARM.gini_Pp_vec
    gini_Pp_2 = FRM.gini_Pp_vec
    # time horizon
    T = len(H_share_1)
    T_vec = arange(T) - 1 # T = - 1 corresponds to the pre-shock period
    if type(T_max) == type(None):
        T_max = T - 1
    # Make lists for graphs
    data1 = [H_share_1, ave_a_1, ave_rm_1, ave_Pd_1, ave_Pp_1, gini_a_1, gini_rm_1, gini_Pd_1, gini_Pp_1]
    data2 = [H_share_2, ave_a_2, ave_rm_2, ave_Pd_2, ave_Pp_2, gini_a_2, gini_rm_2, gini_Pd_2, gini_Pp_2]
    titles = ['Homeownership', 'Mean asset holdings', 'Mean mortgage rate', 'Mean default probability', 'Mean purchase probability',
              'Gini: asset holdings', 'Gini: mortgage rare', 'Gini: default probability', 'Gini: purchase probability']
    fnames = ['H_share', 'ave_a', 'ave_rm', 'ave_Pd', 'ave_Pp', 'gini_a', 'gini_rm', 'gini_Pd', 'gini_Pp']
    for i in range(len(data1)):
        if plot_in_deviations:
            data2plot = array([data1[i][1:T_max+2] - data1[i][0:1], data2[i][1:T_max+2] - data2[i][0:1]])
            T_data = T_vec[1:T_max+2]
            y_label = 'deviations from the pre-shock level'
        else:
            data2plot = array([data1[i][0:T_max+2], data2[i][0:T_max+2]])
            T_data = T_vec[0:T_max+2]
            y_label = None
        fname = 'Comparison_' + fnames[i] +'.png'
        multiple_line_plot(T_data, data2plot,
                           x_label = 'periods',
                           y_label = y_label,
                           title = titles[i],
                           labels = ['Adjustable', 'Fixed'],
                           line_width = [2.5, 1.5],
                           line_styles = ['solid', 'dashed'],
                           fname = fname)