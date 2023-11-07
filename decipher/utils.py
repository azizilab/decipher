import numpy as np
from PIL import Image

DECIPHER_GLOBALS = dict()
DECIPHER_GLOBALS["save_folder"] = "./_decipher_models/"


def create_decipher_uns_key(adata):
    """
    Create the `decipher` uns key if it doesn't exist.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    """
    if "decipher" not in adata.uns:
        adata.uns["decipher"] = dict()
    if "trajectories" not in adata.uns["decipher"]:
        adata.uns["decipher"]["trajectories"] = dict()


def is_notebook() -> bool:
    """Check if the code is running in a Jupyter notebook.

    Returns
    -------
    bool
        True if running in a Jupyter notebook, False otherwise.
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


class GIFMaker:
    """Make a GIF from a list of images."""

    def __init__(self, dpi=100):
        self.images = []
        self.dpi = dpi

    def add_image(self, fig):
        """Add an image to the GIF.

        Parameters
        ----------
        fig : matplotlib.pyplot.figure
            The figure to add.
        """
        fig.set_dpi(self.dpi)
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        self.images.append(Image.fromarray(image))

    def save_gif(self, filename):
        """Make and save a GIF from the images.

        Parameters
        ----------
        filename : str
            The filename to save the GIF to. Add `.gif` if not present.
        """
        images = self.images
        if not filename.endswith(".gif"):
            filename += ".gif"

        images[0].save(
            filename,
            save_all=True,
            append_images=images[1:],
            loop=0,
        )


def load_and_show_gif(filename):
    """Load and show a GIF in a Jupyter notebook.

    Parameters
    ----------
    filename : str
        The filename of the GIF.
    """
    from IPython.display import Image, display

    with open(filename, "rb") as f:
        display(Image(data=f.read(), format="png"))
