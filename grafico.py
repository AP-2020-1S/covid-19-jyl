def grafico(base, t, li, ls, y):
    base.rename(columns={'y': 'y'}, inplace=True)
    base.rename(columns={'li': 'LI'}, inplace=True)
    base.rename(columns={'ls': 'LS'}, inplace=True)
    base.rename(columns={'t': 't'}, inplace=True)
    base['y'] = [int(x) for x in base['y']]
    base['LI'] = [int(x) for x in base['LI']]
    base['LS'] = [int(x) for x in base['LS']]
    plt.plot('t', 'LI', data=base, marker='', color='blue', linewidth=3, linestyle=":")
    plt.plot('t', 'LS', data=base, marker='', color='blue', linewidth=3, linestyle=":")
    plt.plot('t', 'y', data=base, marker='', color='grey', linewidth=3, linestyle="-")
    plt.legend()
