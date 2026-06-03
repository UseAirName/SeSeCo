import torch
import pickle
import logging
import numpy as np
import constriction

from math import floor, ceil
from typing import List

logger = logging.getLogger(__name__)

RGB2YUV = torch.tensor([
    [0.29900, 0.58700, 0.11400],
    [-.14713, -.28886, 0.43600],
    [0.61500, -.51498, -.10001]
])

YUV2RGB = torch.tensor([
    [1, 0.00000, 1.13983],
    [1, -.39465, -.58060],
    [1, 2.03211, 0.00000]
])

QUANT_Y = torch.Tensor([[16, 11, 10, 16, 24, 40, 51, 61],
                        [12, 12, 14, 19, 26, 58, 60, 55],
                        [14, 13, 16, 24, 40, 57, 69, 56],
                        [14, 17, 22, 29, 51, 87, 80, 62],
                        [18, 22, 37, 56, 68, 109, 103, 77],
                        [24, 35, 55, 64, 81, 104, 113, 92],
                        [49, 64, 78, 87, 103, 121, 120, 101],
                        [72, 92, 95, 98, 112, 100, 103, 99]])

QUANT_UV = torch.Tensor([[17, 18, 24, 47, 99, 99, 99, 99],
                         [18, 21, 26, 66, 99, 99, 99, 99],
                         [24, 26, 56, 99, 99, 99, 99, 99],
                         [47, 66, 99, 99, 99, 99, 99, 99],
                         [99, 99, 99, 99, 99, 99, 99, 99],
                         [99, 99, 99, 99, 99, 99, 99, 99],
                         [99, 99, 99, 99, 99, 99, 99, 99],
                         [99, 99, 99, 99, 99, 99, 99, 99]])

QUANT_Y_ZIGZAG = np.asarray([16, 11, 12, 14, 12, 10, 16, 14, 13, 14, 18, 17, 16, 19, 24, 40, 26, 24, 22, 22, 24, 49,
                             35, 37, 29, 40, 58, 51, 61, 60, 57, 51, 56, 55, 64, 72, 92, 78, 64, 68, 87, 69, 55, 56,
                             80, 109, 81, 87, 95, 98, 103, 104, 103, 62, 77, 113, 121, 112, 100, 120, 92, 101, 103,
                             99])

QUANT_UV_ZIGZAG = np.asarray([17, 18, 18, 24, 21, 24, 47, 26, 26, 47, 99, 66, 56, 66, 99, 99, 99, 99, 99, 99, 99, 99,
                              99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
                              99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99])

bins_y_std = [0.07656583, 0.033128235, 0.023208119, 0.025801927, 0.018706212, 0.025196781, 0.022785887, 0.047177285,
              0.0047955755, 0.033964608, 0.03923631, 0.0049782842, 0.04459002, 0.0041214684, 0.0053129434, 0.04678877,
              0.0040585482, 0.0032822373, 0.0052494695, 0.003406892, 0.0028228378, 0.053777482, 0.0038981824,
              0.004552435, 0.003663496, 0.0054607675, 0.0033863133, 0.0045244223, 0.0041729975, 0.003226463,
              0.0030089503, 0.08831079, 0.0028293005, 0.0030328992, 0.0031760358, 0.0043370808, 0.0035799055,
              0.0058787786, 0.002776768, 0.003393146, 0.0031605293, 0.0023383233, 0.008379286, 0.0028349867,
              0.0025928635, 0.0035542394, 0.0025834388, 0.0060299216, 0.002118752, 0.0027742307, 0.00392752,
              0.0026354855, 0.003333928, 0.001979688, 0.0034105023, 0.0025611678, 0.0031632779, 0.0021845442,
              0.001954958, 0.0018019335, 0.0015824179, 0.00125767, 0.0008126211, 0.2190087]

bins_u_std = [0.009831959, 0.0045262054, 0.0031666134, 0.0026580095, 0.0026274943, 0.0043556103, 0.0042103627,
              0.0053844415, 0.0009793394, 0.0039708302, 0.0035412614, 0.00091390416, 0.0054451833, 0.00066748617,
              0.0010351305, 0.005676854, 0.0007593535, 0.00063035917, 0.0009945964, 0.00070028397, 0.00048946287,
              0.00640318, 0.00070900953, 0.0011056678, 0.0007337942, 0.00096493395, 0.00065251207, 0.0009573097,
              0.000752588, 0.0005150146, 0.0006306064, 0.012496868, 0.00046447004, 0.00065887556, 0.0007301865,
              0.0005820541, 0.0008567084, 0.0010033606, 0.00047001912, 0.001137104, 0.00040673892, 0.00037964366,
              0.0011601388, 0.0005799496, 0.0005021778, 0.000566561, 0.00038172278, 0.0023139098, 0.0004242128,
              0.0004151404, 0.00066292484, 0.00044177906, 0.00076788914, 0.0003291452, 0.00068856584, 0.00054425886,
              0.00040526764, 0.0003564663, 0.00035033395, 0.00030552634, 0.00020732878, 0.00015499858, 9.088046e-05,
              0.03283475]

bins_v_std = [0.012890416, 0.005977084, 0.0032401765, 0.0030795413, 0.002876952, 0.0033833375, 0.0033734334,
              0.0053358995, 0.0012852136, 0.0072971373, 0.0049968967, 0.0011160726, 0.005859675, 0.0008774494,
              0.0012099461, 0.006921444, 0.0007976085, 0.0010494858, 0.0016230776, 0.0006966489, 0.0006137648,
              0.007228267, 0.0009570972, 0.0009521915, 0.0007661939, 0.0010966596, 0.0006878396, 0.0014359613,
              0.001104683, 0.00089154847, 0.0006713137, 0.011549315, 0.0007054945, 0.0007543214, 0.0009088324,
              0.0007824286, 0.0013947496, 0.0011520606, 0.00056227535, 0.00075581257, 0.00076338, 0.00049126527,
              0.0018663389, 0.00044092178, 0.00047795768, 0.00079327356, 0.0005158206, 0.0021376566, 0.00055982504,
              0.00050897925, 0.0009040824, 0.0008577286, 0.0008970315, 0.0005359236, 0.00078901823, 0.0006375527,
              0.0004992122, 0.0006514243, 0.00034654228, 0.0004505462, 0.00019280617, 0.00014763112, 0.000105649655,
              0.033171397]


def quantize_uniform_gaussian_std(x: torch.Tensor,
                                  stds: torch.Tensor,
                                  p: float,
                                  min_val: float = -1.,
                                  max_val: float = 1) \
        -> [torch.Tensor, torch.Tensor]:
    """
        Uniform quantization, under Gaussian assumption given the stds
        :param x: vector to quantize
        :param stds:
        :param p: precision, scale on the threshold and the quantization step
        :param min_val: range of values
        :param max_val: range of values
    """
    # quantize step
    stds = stds / stds[0]
    q = (max_val - min_val) / 2 * p / stds
    n_symbols = torch.ceil((max_val - min_val) / q)
    clamped = (torch.clamp(x, min=min_val, max=max_val))
    quantize = torch.floor(clamped / q)
    val = (quantize * q + q / 2)
    return quantize.numpy().astype(np.int32), val, n_symbols.numpy().astype(np.int32)


def quantize_uniform_gaussian(x: torch.Tensor,
                              p: float,
                              min_val: float = -1.,
                              max_val: float = 1) \
        -> [torch.Tensor, torch.Tensor]:
    """
        Uniform quantization, under Gaussian assumption given the stds
        :param x: vector to quantize
        :param p: precision, scale on the threshold and the quantization step
        :param min_val: range of values
        :param max_val: range of values
    """
    # quantize step
    q = (max_val - min_val) / 2 * p
    n_symbols = ceil((max_val - min_val) / q)
    clamped = (torch.clamp(x, min=min_val, max=max_val))
    quantize = torch.floor(clamped / q)
    val = (quantize * q + q / 2)
    return quantize.numpy().astype(np.int32), val, n_symbols


def quantize_jpeg(x: torch.Tensor,
                  matrix: np.ndarray,
                  p: float,
                  min_val: float,
                  max_val: float):
    matrix = matrix * p
    x = torch.floor(x * 1024 / (max_val - min_val))
    quantize = torch.floor(x / matrix)
    val = (quantize * matrix + matrix / 2) / 512
    return quantize.numpy().astype(np.int32), val.to(torch.float), np.ceil(1024 / matrix)


def code_quantized(quantized: np.ndarray,
                   n_symbols,
                   stds) -> int:
    # Mean vector is set to 0
    means = np.array([0.] * len(stds))
    stds = n_symbols // 2 * stds
    n_symbol = int(np.asarray(n_symbols).max())
    model = constriction.stream.model.QuantizedGaussian(-n_symbol + n_symbol // 2, n_symbol // 2)
    encoder = constriction.stream.stack.AnsCoder()
    encoder.encode_reverse(quantized, model, means, stds)
    compressed = encoder.get_compressed()
    decoder = constriction.stream.stack.AnsCoder(compressed)
    reconstructed = decoder.decode(model, means, stds)
    assert (reconstructed - quantized).sum() < 0.5
    return len(compressed) * 32


def encode_guide(signals: List[torch.Tensor],
                 eigen_vector_matrices: List[torch.Tensor],
                 sizes_in: List[int],
                 sizes_coarsened: List[int],
                 p: float):
    # Convert rgb to yuv
    for i in range(len(signals)):
        logger.debug("Signals shape: {},  {}".format(i, signals[i].shape))
        signals[i] = torch.matmul(RGB2YUV, signals[i].t())

    # Project the signal on the fourier basis
    fourier_freq = []
    for i, (signal, eigen_vector_matrix) in enumerate(zip(signals, eigen_vector_matrices)):
        logger.debug("Frequencies shape: {},  {}".format(i, signals[i].shape))
        fourier_freq.append(torch.matmul(signal, eigen_vector_matrix))

    # Distribute the frequencies in 64 bins for each channel
    y_bins = [[] for _ in range(64)]
    u_bins = [[] for _ in range(64)]
    v_bins = [[] for _ in range(64)]
    order = [[] for _ in range(64)]

    for count, (freq, n, n_c) in enumerate(zip(fourier_freq, sizes_in, sizes_coarsened)):
        y, u, v = freq
        logger.debug("Y_freq: min: {}, max: {}, norm: {}".format(y.min() / n_c ** 0.5, y.max() / n_c ** 0.5,
                                                                 y.norm() / n_c ** 0.5))
        logger.debug("U_freq: min: {}, max: {}, norm: {}".format(u.min() / n_c ** 0.5, u.max() / n_c ** 0.5,
                                                                 u.norm() / n_c ** 0.5))
        logger.debug("V_freq: min: {}, max: {}, norm: {}".format(v.min() / n_c ** 0.5, v.max() / n_c ** 0.5,
                                                                 v.norm() / n_c ** 0.5))
        for i in range(len(y)):
            bin_val = int((i + 1) / len(y) * int(n_c / n * 63))
            y_bins[bin_val].append(y[i] / n_c ** 0.5)
            u_bins[bin_val].append(u[i] / n_c ** 0.5)
            v_bins[bin_val].append(v[i] / n_c ** 0.5)
            order[bin_val].append(count)

    # Retrieve the std per element
    y_std = [bins_y_std[j] for j in range(len(y_bins)) for _ in range(len(y_bins[j]))]
    u_std = [bins_u_std[j] for j in range(len(u_bins)) for _ in range(len(u_bins[j]))]
    v_std = [bins_v_std[j] for j in range(len(v_bins)) for _ in range(len(v_bins[j]))]

    # Concatenate the all frequencies
    y = torch.tensor([freq for bins in y_bins for freq in bins])
    u = torch.tensor([freq for bins in u_bins for freq in bins])
    v = torch.tensor([freq for bins in v_bins for freq in bins])
    order = torch.tensor([signal_pos for bins in order for signal_pos in bins])

    # Gaussian quantization of the frequency vector
    y_quant, y_val, y_symbols = quantize_uniform_gaussian(y, p, -1, 1)  # Min nad max for y vector
    u_quant, u_val, u_symbols = quantize_uniform_gaussian(u, p, -0.436, 0.436)  # Same for u
    v_quant, v_val, v_symbols = quantize_uniform_gaussian(v, p, -0.615, 0.615)  # Same for v
    logger.debug("Coding symbols, y,u,v: {}, {}, {}".format(y_symbols, -y_symbols + y_symbols // 2, y_symbols // 2))
    logger.debug("Max symbols, y,u,v: {}, {}, {}".format(y_quant.max(), u_quant.max(), v_quant.max()))
    logger.debug("Min symbols, y,u,v: {}, {}, {}".format(y_quant.min(), u_quant.min(), v_quant.min()))

    # Encode the quantized vector with lossless entropy coder
    y_bits = code_quantized(y_quant, y_symbols, np.asarray(y_std))
    u_bits = code_quantized(u_quant, u_symbols, np.asarray(u_std))
    v_bits = code_quantized(v_quant, v_symbols, np.asarray(v_std))

    rate = y_bits + u_bits + v_bits
    logger.debug("Total size before: {}, after {}".format(3 * len(y_quant) * 32, rate))

    # Reorder frequencies per graph and return to rgb
    yuv_val = torch.stack((y_val, u_val, v_val), dim=1)
    signals_coded = [[] for _ in range(len(signals))]
    rgb_val = torch.matmul(YUV2RGB, yuv_val.t()).t()

    for i in range(len(order)):
        pos = order[i]
        signals_coded[pos].append(rgb_val[i])

    # Concatenate per graph
    signals_coded = [torch.matmul(torch.stack(signal_c, dim=0).t(), eigen_vector_matrix.t()).t() * n_c ** 0.5
                     for signal_c, eigen_vector_matrix, n_c in
                     zip(signals_coded, eigen_vector_matrices, sizes_coarsened)]

    # Concatenate all graphs
    guide_coded = torch.concat(signals_coded)
    return guide_coded, rate
