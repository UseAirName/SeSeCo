import torch
import torchvision.transforms.functional as tf

from PIL import Image


def seg2edges(seg_pil: Image):
    seg_ts = tf.to_tensor(seg_pil)[0]
    seg_ts = (seg_ts * 255).to(torch.int64)
    h_edges = torch.zeros_like(seg_ts[:, :-1])
    v_edges = torch.zeros_like(seg_ts[:-1])
    h_edges += seg_ts[:, :-1] - seg_ts[:, 1:]
    v_edges += seg_ts[:-1] - seg_ts[1:]
    h_edges = torch.where(h_edges != 0, 1, 0)
    v_edges = torch.where(v_edges != 0, 1, 0)
    startPt = []
    dcode = []
    maxHRow, maxHCol = h_edges.shape
    maxVRow, maxVCol = v_edges.shape

    while h_edges.max() > 0 or v_edges.max() > 0:
        h, v = False, False
        if h_edges.max() > 0:
            rowh, colh = torch.argwhere(h_edges > 0)[0]
            h = True

        if v_edges.max() > 0:
            rowv, colv = torch.argwhere(v_edges > 0)[0]
            v = True

        if h and v:
            h = (rowh - 1, colh) <= (rowv, colv - 1)
            v = not h

        if h:
            row, col = rowh, colh
            lastDirect = 3
            startPt.append({'h': 1, 'row': row, 'col': col})
            dcode.append([3])  # south
            h_edges[row, col] = 0
        elif v:
            row, col = rowv, colv
            lastDirect = 2
            startPt.append({'h': 0, 'row': row, 'col': col})
            dcode.append([2])  # east
            v_edges[row, col] = 0

        dCount = 1
        done = False

        while not done:
            if lastDirect == 1:  # North
                if row > 1 and h_edges[row-1, col] > 0:
                    h_edges[row-1, col] = 0
                    dCount += 1
                    dcode[-1].append(1)
                    row = row - 1
                    lastDirect = 1
                elif row > 1 and col < maxVCol - 1 and v_edges[row-1, col+1] > 0:
                    v_edges[row-1, col+1] = 0
                    dCount += 1
                    dcode[-1].append(2)
                    row -= 1
                    col += 1
                    lastDirect = 2
                elif row > 1 and v_edges[row-1, col] > 0:
                    v_edges[row-1, col] = 0
                    dCount += 1
                    dcode[-1].append(4)
                    row -= 1
                    lastDirect = 4
                else:
                    done = True

            if lastDirect == 2:  # East
                if col < maxHCol and h_edges[row, col] > 0:
                    h_edges[row, col] = 0
                    dCount += dCount
                    dcode[-1].append(1)
                    lastDirect = 1
                elif col < maxVCol - 1 and v_edges[row, col + 1] > 0:
                    v_edges[row, col + 1] = 0
                    dCount += 1
                    dcode[-1].append(2)
                    col += 1
                elif row < maxHRow - 1 and col < maxHCol and h_edges[row + 1, col] > 0:
                    h_edges[row + 1, col] = 0
                    dCount += 1
                    dcode[-1].append(3)
                    lastDirect = 3
                    row += 1
                else:
                    done = True

            if lastDirect == 3:  # South
                if row < maxHRow - 1 and h_edges[row + 1, col] > 0:
                    h_edges[row + 1, col] = 0
                    dCount = dCount + 1
                    dcode[-1].append(3)
                    row = row + 1
                elif row < maxVRow and col < maxVCol - 1 and v_edges[row, col + 1] > 0:
                    v_edges[row, col + 1] = 0
                    dCount = dCount + 1
                    dcode[-1].append(2)
                    lastDirect = 2
                    col = col + 1
                elif row < maxVRow and v_edges[row, col] > 0:
                    v_edges[row, col] = 0
                    dCount = dCount + 1
                    dcode[-1].append(4)
                    lastDirect = 4
                else:
                    done = True

            if lastDirect == 4:  # West
                if col > 1 and h_edges[row, col - 1] > 0:
                    h_edges[row, col - 1] = 0
                    dCount = dCount + 1
                    dcode[-1].append(1)
                    lastDirect = 1
                    col = col - 1
                elif col > 1 and v_edges[row, col - 1] > 0:
                    v_edges[row, col - 1] = 0
                    dCount = dCount + 1
                    dcode[-1].append(4)
                    col = col - 1
                elif row < maxHRow - 1 and col > 1 and h_edges[row + 1, col - 1] > 0:
                    h_edges[row + 1, col - 1] = 0
                    dCount = dCount + 1
                    dcode[-1].append(3)
                    lastDirect = 3
                    row = row + 1
                    col = col - 1
                else:
                    done = True
        dCount += 1
        dcode[-1].append(0)  # end symbol
    return startPt, dcode

