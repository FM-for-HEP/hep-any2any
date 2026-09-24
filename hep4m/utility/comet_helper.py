import os
import warnings

import matplotlib.pyplot as plt
import numpy as np


def save_plot(fig, name, comet_logger=None):
    """Log a figure to Comet when a Comet logger is given, else save it to plot_dump/<name>.png."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
    if comet_logger is not None:
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        comet_logger.experiment.log_image(
            image_data=image,
            name=name,
            overwrite=False,
            image_format="png",
        )
    else:
        plot_dir = 'plot_dump'
        os.makedirs(plot_dir, exist_ok=True)
        fig.savefig(f'{plot_dir}/{name}.png', bbox_inches='tight')
    plt.close(fig)
