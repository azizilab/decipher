def activate_journal_quality():
    """Activate journal quality settings for plotting.
    It is recommended for high quality figures while keeping the file size small.
    It is recommended if the figures are to be edited in Adobe Illustrator.
    """
    import matplotlib.pyplot as plt
    import scanpy as sc
    import matplotlib as mpl

    sc.settings.set_figure_params(dpi_save=400, vector_friendly=True, fontsize=18)
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42

    plt.rcParams["axes.grid"] = False
