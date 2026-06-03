import numpy as np


def best_fit_line(full_dcode):
    # Step 0: convert direction code to (x,y) coordinate
    length = len(full_dcode)
    dcode = full_dcode[length - 3:length]  # the last 3 data points

    # To deal with dcode like [1, 2, 3] and [3, 2, 1]
    if (dcode[0] - dcode[1] == dcode[1] - dcode[2]) and abs(dcode[0] - dcode[1]) == 1:
        dcode = dcode[1:]  # just consider the last two codes

    d_length = len(dcode)

    # step 0.1: initialize 1 point
    xd = [0]
    yd = [0]

    for i in range(d_length):  # step 0.2: convert rest of chain code
        if dcode[i] == 1:  # North
            xd.append(xd[i])
            yd.append(yd[i] + 1)
        elif dcode[i] == 2:  # East
            xd.append(xd[i] + 1)
            yd.append(yd[i])
        elif dcode[i] == 3:  # South
            xd.append(xd[i])
            yd.append(yd[i] - 1)
        else:  # West
            xd.append(xd[i] - 1)
            yd.append(yd[i])

    # Step 1: calculate least squares fitting--perpendicular offsets
    if sum(d == 1 for d in dcode) == d_length or sum(d == 3 for d in dcode) == d_length:
        avg_diff = 0
        m = float('inf')
    else:
        xm = np.mean(xd)
        ym = np.mean(yd)
        temp1 = np.dot(yd, yd) - len(xd) * ym ** 2 - (np.dot(xd, xd) - len(xd) * xm ** 2)  # eq.(18)
        temp2 = len(xd) * xm * ym - np.dot(xd, yd)

        if temp2 == 0:
            avg_diff = 0
            m = 0
            b = ym
        else:
            B = 0.5 * temp1 / temp2
            m1 = -B + np.sqrt(B ** 2 + 1)  # or -B - sqrt(B^2+1)
            m2 = -B - np.sqrt(B ** 2 + 1)
            b1 = ym - m1 * xm
            b2 = ym - m2 * xm
            sum1 = sum(abs(yd[i] - (b1 + m1 * xd[i])) / np.sqrt(1 + m1 ** 2) for i in range(len(xd)))  # eq.(3)
            sum2 = sum(abs(yd[i] - (b2 + m2 * xd[i])) / np.sqrt(1 + m2 ** 2) for i in range(len(xd)))

            if sum1 < sum2:
                avg_diff = sum1 / len(xd)
                m = m1
                b = b1
            else:
                avg_diff = sum2 / len(xd)
                m = m2
                b = b2

    # Step 2: find predicted direction
    x = xd[d_length]  # 1 more coord than dcode
    y = yd[d_length]
    if m < float('inf'):
        if m != 0:
            x2 = (y - b) / m  # line intersects with last coord
            y2 = m * x + b
        else:
            y2 = y
            if dcode[d_length - 1] == 2:
                x2 = x + 1
            else:
                x2 = x - 1
    else:
        x2 = x
        if dcode[d_length - 1] == 1:
            y2 = y + 1
        else:
            y2 = y - 1

    if dcode[d_length - 1] == 1:  # decode: north
        if m >= 0:
            ang = [np.pi - np.arctan(m), np.pi / 2 - np.arctan(m), np.arctan(m)]  # left, straight, right
        else:
            ang = [-np.arctan(m), np.pi / 2 + np.arctan(m), np.pi + np.arctan(m)]
        min_ang = min(ang)

        if y2 > y:
            ang[1] = 0
        elif x2 < x:
            ang[0] = 0
        else:
            ang[2] = 0

    elif dcode[d_length - 1] == 2:  # decode: east
        ang = [np.pi / 2 - np.arctan(m), abs(np.arctan(m)), np.pi / 2 + np.arctan(m)]  # left, straight, right
        min_ang = min(ang)

        if x2 > x:
            ang[1] = 0
        elif y2 > y:
            ang[0] = 0
        else:
            ang[2] = 0
    elif dcode[d_length - 1] == 3:  # decode: south ******
        if m >= 0:
            ang = [np.pi - np.arctan(m), np.pi / 2 - np.arctan(m), np.arctan(m)]
        else:
            ang = [-np.arctan(m), np.pi / 2 + np.arctan(m), np.pi + np.arctan(m)]
        min_ang = min(ang)

        if y2 < y:
            ang[1] = 0  # ang(2)/2
        elif x2 > x:
            ang[0] = 0  # ang(1)/2
        else:
            ang[2] = 0  # ang(3)/2

    elif dcode[d_length - 1] == 4:  # decode: west ******
        ang = [np.pi / 2 - np.arctan(m), abs(np.arctan(m)), np.pi / 2 + np.arctan(m)]
        min_ang = min(ang)

        if x2 < x:
            ang[1] = 0  # ang(2)/2
        elif y2 < y:
            ang[0] = 0  # ang(1)/2
        else:
            ang[2] = 0  # ang(3)/2

    # step 3: compute prob using von Mises distr. ********
    # 3.1 construct history points ******
    ind = 0
    xd2 = [0]
    yd2 = [0]
    hisLen = 2
    for i in range(length - 4, max(-1, length - 3 - hisLen - 2), -1):  # history length
        if full_dcode[i] == 1:  # North
            xd2.append(xd2[ind])
            yd2.append(yd2[ind] - 1)
        elif full_dcode[i] == 2:  # East
            xd2.append(xd2[ind] - 1)
            yd2.append(yd2[ind])
        elif full_dcode[i] == 3:  # South
            xd2.append(xd2[ind])
            yd2.append(yd2[ind] + 1)
        else:  # West
            xd2.append(xd2[ind] + 1)
            yd2.append(yd2[ind])
        ind += 1

    sum1 = 0
    for i in range(1, ind):
        if m < float('inf'):
            sum1 += abs(yd2[i] - (b + m * xd2[i])) / np.sqrt(1 + m ** 2)
        else:
            sum1 += abs(xd2[i] - x)

    if ind > 1:
        hisDiff = sum1 / (ind - 1)
    else:
        hisDiff = 0

    # 3.2 compute kappa and von Mises ditr. ******
    rho = 2.0
    tau1 = 0.5
    tau2 = 1.0
    kappa = rho * np.exp(-1 * tau1 * abs(hisDiff - avg_diff)) * np.exp(-1 * tau2 * abs(avg_diff))

    prob = np.zeros(3)
    for i in range(3):
        prob[i] = np.exp(kappa * np.cos(ang[i]))
    prob = prob / np.sum(prob)
    return prob


def propagate_carry(t, d):
    n = t - 1

    # Complement all the outstanding bits until the first 0-bit is complemented
    while d[n] == 1:
        d[n] = 0
        n -= 1
        if n == 0:
            break
    if n >= 1:
        d[n] = 1
    else:
        # Extend the length of the array and shift bits
        d = np.insert(d, 0, 1)  # Insert 1 at the beginning of the array
    return d


def aec(dcode):
    """
    Arithmetic Edge Encoder.

    Args:
        dcode (list): Direction code (1-4 means North, East, South, West).

    Returns:
        d (list): AEC codeword (binary).
        b_count (int): Number of used bits.
    """
    # Step 0: Initialize variables
    fit_length = 3  # Length of input vector for bestFitLine
    b = 0  # Interval base
    l = 1  # Interval length
    gamma = 2  # Interval rescaling factor
    b_count = 0  # Bit counter
    d = []  # The AEC codeword

    for ind in range(len(dcode)):
        # Step 1: Find best estimate of probabilities of 3 possible directions
        if ind == 0:
            prob = np.ones(4) / 4.0  # First chain
        else:
            if ind > fit_length - 1:
                prob = best_fit_line(dcode[:ind])
            else:
                prob = np.ones(3) / 3.0

            # Add end-code probability (here unused, code the symbol number instead)
            end_prob = 0
            prob = np.append(prob * (1.0 - end_prob), end_prob)

        # Step 2: Arithmetic coding - given prob. vector, index, compute new interval
        if ind == 0:
            ccode = dcode[0]  # First chain is in absolute terms
        elif ind < len(dcode):
            if dcode[ind] == dcode[ind - 1]:
                ccode = 2  # Straight
            elif (dcode[ind] - dcode[ind - 1] == 1) or (dcode[ind] == 1 and dcode[ind - 1] == 4):
                ccode = 3  # Right
            else:
                ccode = 1  # Left
        else:
            ccode = 4  # End-code

        # Update interval
        b += np.sum(prob[:ccode - 1]) * l
        l *= prob[ccode - 1]

        # Carry propagation
        if b >= 1:  # There is a carry
            b -= 1
            d = propagate_carry(b_count, d)

        # Interval renormalization
        while l <= 0.5:
            b_count += 1
            l *= 2
            if b >= 0.5:
                d.append(1)  # Output bit 1
                delta = 0.5
                b = gamma * (b - delta)
            else:
                d.append(0)  # Output bit 0
                delta = 0
                b = gamma * (b - delta)

    # Step 3: Choose final code value
    b_count += 1
    if b <= 0.5:
        d.append(1)
    else:
        d.append(0)
        d = propagate_carry(b_count - 1, d)

    # Remove the 0's at the end
    while d and d[-1] == 0:
        d.pop()
        b_count -= 1

    return d, b_count


def aed(d, sym_count):
    fit_length = 3
    dcode = []
    b = 0
    l = 1
    P = min(32, len(d))
    t = P
    v = 0

    for n in range(P):
        v += d[n] * 2 ** (-n-1)

    for ind in range(sym_count):
        if ind == 0:
            prob = np.ones(4) / 4.0  # First chain
        else:
            if ind > fit_length - 1:
                prob = best_fit_line(dcode[:ind])
            else:
                prob = np.ones(3) / 3.0

            # Add end-code probability (here unused, code the symbol number instead)
            end_prob = 0
            prob = np.append(prob * (1.0 - end_prob), end_prob)

        i = 1
        while b + prob[:i].sum() * l < v:
            i += 1
        if not dcode:
            dcode = [i]
        else:
            if dcode[-1] == 1:
                if i == 1:
                    dcode.append(4)
                elif i == 2:
                    dcode.append(1)
                else:
                    dcode.append(2)
            elif dcode[-1] == 2:
                if i == 1:
                    dcode.append(1)
                elif i == 2:
                    dcode.append(2)
                else:
                    dcode.append(3)
            elif dcode[-1] == 3:
                if i == 1:
                    dcode.append(2)
                elif i == 2:
                    dcode.append(3)
                else:
                    dcode.append(4)
            else:
                if i == 1:
                    dcode.append(3)
                elif i == 2:
                    dcode.append(4)
                else:
                    dcode.append(1)
        b += prob[: i - 1].sum() * l
        l *= prob[i - 1]

        if b >= 1:
            b -= 1
            v -= 1

        while l <= 0.5:
            if b >= 0.5:
                b = 2 * (b - 0.5)
                v = 2 * (v - 0.5)
            else:
                b *= 2
                v *= 2
            if len(d) > P:
                t += 1
                if t - 1 < len(d):
                    v += 2**(-P) * d[t-1]
            l *= 2

    return dcode
