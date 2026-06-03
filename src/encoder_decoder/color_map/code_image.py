import torch
import logging
import torchvision.transforms.functional as tf

from PIL import Image
from torch.nn.functional import one_hot

from .quantize import encode_guide
from .graph.utils.seg2mask import seg_to_masks
from .graph.utils.img2graph import img_to_graphs
from .graph.coarsening.coarsen import coarsen_graph, coarsen_signal, lift_signal

logger = logging.getLogger(__name__)


def coarsening_ratio(coarsening_strength, preference, threshold=0.2):
    coarsening_strength = 30 - coarsening_strength
    if coarsening_strength <= 15:
        return threshold * preference ** (15 / coarsening_strength)
    else:
        return threshold * preference ** ((30 - coarsening_strength) / 15)


def code_image(image: Image, seg_ts: torch.DoubleTensor, q: float, preferences, coarsening_strength, threshold):
    """
    Code the image into the selective color map with graph coarsening
    :param image: The image to code
    :param seg_ts: the segmentation map as a 1-d tensor
    :param q: quantization step
    :param preferences: user preferences
    :param coarsening_strength: strength of the coarsening
    :param threshold: threshold on maximum signal ratio
    """
    image_ts = tf.to_tensor(image)

    # Get the list of masks and labels from the segmentation map
    mask_list, label_list = seg_to_masks(seg_ts)

    # Get a list of graphs corresponding to the masks and the signal over each graph
    graph_list, signal_list = img_to_graphs(image_ts, mask_list)

    coarsening_matrices = []
    matrix_image = []
    coarsened_signals = []
    graphs_size_in = []
    graphs_size_out = []
    fourier_basis_matrices = []
    pixels_xy = []

    code = False

    logger.info("Number of regions to encode: {}".format(len(graph_list)))
    for graph, signal_on_graph, label in zip(graph_list, signal_list, label_list):
        n_input = graph.N
        r_c = coarsening_ratio(coarsening_strength, preferences[label], threshold=threshold)
        if preferences[label] == 1:
            n_output = n_input
        else:
            # Keep at least one node per graph
            n_output = max(int(n_input * r_c), 1)

        # Coarsen the graph
        logger.debug("Coarsening graph of size {} to {}".format(n_input, n_output))
        graph_c, matrix_c = coarsen_graph(graph, n_output)

        # Coarsen the signal over the graph
        n_output = graph_c.N
        coarsened_signal = coarsen_signal(matrix_c, signal_on_graph)
        logger.debug("Effective coarsening size {} to {}".format(n_input, n_output))

        # Compute the fourier decomposition
        logger.info("Computing fourier basis...")
        if code:
            graph_c.compute_fourier_basis()
            U = graph_c.U
            fourier_basis_matrices.append(torch.tensor(U, dtype=torch.float))
        graphs_size_in.append(n_input)
        graphs_size_out.append(n_output)
        coarsened_signals.append(coarsened_signal)
        pixels_xy.append(torch.tensor(graph.coords, dtype=torch.int))
        coarsening_matrices.append(matrix_c)

        # Update the matrix A for the whole image in $g = A x$
        coord = torch.tensor(graph.coords, dtype=torch.int)
        c_matrix_padded = torch.zeros((n_output, seg_ts.shape[0], seg_ts.shape[1]))
        c_matrix_padded[:, coord[:, 0], coord[:, 1]] = matrix_c.to(torch.float)
        matrix_image.append(c_matrix_padded)
    # Concatenate the coarsening matrices to form A
    image_matrix = torch.concat(matrix_image).flatten(1, 2)
    P = image_matrix.pow(2)

    color_image = color_map(image_ts.shape, coarsening_matrices, coarsened_signals, pixels_xy)

    # Concatenate coarsened signals to form the guide
    g = torch.concat(coarsened_signals)

    # The eigen vectors, graph size in and out are available at both decoder and encoder, only the signal over each ...
    # ... graph concatenated is encoded.
    if code:
        g_coded, bits = encode_guide(coarsened_signals, fourier_basis_matrices, graphs_size_in, graphs_size_out, q)
    else:
        g_coded = g
        bits = g_coded.nelement() * 4
    coarsened_signals_encoded = []
    s = 0
    for nc in graphs_size_out:
        coarsened_signals_encoded.append(g_coded[s:s + nc])
        s += nc
    mse = torch.nn.functional.mse_loss(g, g_coded)
    color_image_encoded = color_map(image_ts.shape, coarsening_matrices, coarsened_signals_encoded, pixels_xy)
    return P, g_coded, bits, color_image, color_image_encoded, mse


def color_map(shape, coarsening_matrices, coarsened_signals, pixels_xy):
    color_image = torch.zeros(shape).permute(1, 2, 0)

    for c_matrix, coarsened_signal, coord in zip(coarsening_matrices, coarsened_signals, pixels_xy):
        # Lift the signal for visualization
        signal_l = lift_signal(c_matrix, coarsened_signal)
        color_image[coord[:, 0], coord[:, 1]] = signal_l

    return color_image.permute(2, 0, 1)


def prompt_from_seg(seg_ts: torch.Tensor, labels_list) -> str:
    prompt = ""

    seg_ts = seg_ts.to(torch.int64)

    masks = one_hot(seg_ts).permute(2, 0, 1).to(torch.bool)
    masks = masks[masks.any(dim=(1, 2))]
    masks = masks.unbind(dim=0)
    labels = [seg_ts[mask][0].item() for mask in masks]

    size_label = [(mask.sum(), labels_list[label]) for mask, label in zip(masks, labels)]
    size_label.sort(reverse=True)
    for _, label in size_label:
        prompt += label + ", "
    return prompt
