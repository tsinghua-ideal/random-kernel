import numpy as np
from tvm import te, topi


def fast_sigmoid(X):
    nex = topi.fast_exp(-X)
    return te.compute(
        X.shape,
        lambda *i: 1 / (1 + nex[i])
    )


def padding_1d(X, d, p, v=0, name='padding_1d'):
    assert len(X.shape) >= 1
    if p == 0:
        return X
    assert p > 0
    s = X.shape[d]
    return te.compute(
        (*X.shape[:d], s + p * 2, *X.shape[d:][1:]),
        lambda *i: te.if_then_else(
            te.any(i[d] < p, i[d] >= s + p),
            v, X[i[:d] + (i[d], ) + i[d:][1:]]),
        name=f'{name}_out'
    )


def padding_2d(X, ph, pw, v=0, name='padding_2d'):
    assert len(X.shape) >= 2
    if ph == pw == 0:
        return X
    assert ph > 0 or pw > 0
    h, w = X.shape[-2], X.shape[-1]
    return te.compute(
        (*X.shape[0:-2], h + ph * 2, w + pw * 2),
        lambda *i: te.if_then_else(
            te.any(i[-2] < ph, i[-2] >= h + ph, i[-1] < pw, i[-1] >= w + pw),
            v, X[i[:-2] + (i[-2] - ph, i[-1] - pw)]),
        name=f'{name}_out'
    )


def view_as_4d(X):
    assert len(X.shape) > 4
    n, h, w = X.shape[0], X.shape[-2], X.shape[-1]
    c = np.prod(X.shape[1:-2])
    return topi.reshape(X, [n, c, h, w])


def repeat(X, d, k, name='repeat'):
    assert d < len(X.shape), k > 1
    return te.compute(
        (*X.shape[0:d], X.shape[d] * k, *X.shape[d:][1:]),
        lambda *i: X[i[:d] + (i[d] % X.shape[d], ) + i[d:][1:]],
        name=f'{name}_out'
    )


def repeat_to_group(X, d, k, name='repeat_to_group'):
    assert d < len(X.shape) and k >= 1
    if d < 0:
        d = (d + len(X.shape)) % len(X.shape)
    return te.compute(
        (*X.shape[0:d], k, *X.shape[d:]),
        lambda *i: X[i[:d] + i[d+1:]],
        name=f'{name}_out'
    )


def avg_pool(X, s, name='avg_pool'):
    assert len(X.shape) >= 2
    if s == 1:
        return X
    assert s == 2, 'Temporarily only support s = 2'
    assert X.shape[-2] % 2 == 0 and X.shape[-1] % 2 == 0
    return te.compute(
        (*X.shape[:-2], X.shape[-2] // 2, X.shape[-1] // 2),
        lambda *i: (X[i[:-2] + (i[-2] * 2, i[-1] * 2)] +
                    X[i[:-2] + (i[-2] * 2 + 1, i[-1] * 2)] +
                    X[i[:-2] + (i[-2] * 2, i[-1] * 2 + 1)] +
                    X[i[:-2] + (i[-2] * 2 + 1, i[-1] * 2 + 1)]) / 4.0,
        name=f'{name}_out'
    )


def reshape(X, new_shape):
    if X.shape == new_shape:
        return X
    return topi.reshape(X, new_shape)


def fc(X, W, d, has_nd: bool = True, name: str = 'fc'):
    # X: [n, d * (reduce, ), *], W: [nd, d * (reduce, )]
    nd_shift = int(has_nd)
    assert len(X.shape) >= d + 1
    assert len(W.shape) == d + nd_shift
    n = X.shape[0]
    nd = W.shape[0] if has_nd else 1

    reduce_axes = []
    for i in range(0, d):
        assert X.shape[i + 1] == W.shape[i + nd_shift]
        reduce_axes.append(te.reduce_axis((0, X.shape[i + 1]), name=f'{name}_r{i}'))
    reduce_axes = tuple(reduce_axes)

    return te.compute(
        (n, *((nd, ) if has_nd else ()), *X.shape[d + 1:]),
        lambda *i: te.sum(
            X[(i[0], ) + reduce_axes + i[1 + nd_shift:]] * W[((i[1], ) if has_nd else ()) + reduce_axes],
            axis=reduce_axes
        ),
        # attrs={'layout_free_placeholders': [W]},
        name=f'{name}_out',
    )


def merge_axis(start: int, iterators, fold_axes_map, max_length):
    new_axes = []
    for i, iterator in enumerate(iterators, start=start):
        assert i < max_length
        new_axes.append(iterator + fold_axes_map[i] if i in fold_axes_map else iterator)
    return tuple(new_axes)


def fused_unfold_fc(X, W, d, ud: tuple, has_nd: bool = True, name: str = 'fc'):
    # X: [n, d * (reduce, ), *], W: [nd, (d + unfold_dims) * (reduce, )]
    nd_shift = int(has_nd)
    assert len(X.shape) >= d + 1 + len(ud)
    assert len(W.shape) == d + len(ud) + nd_shift

    # Convert unfolds to non-negatives
    for s, _ in ud:
        assert -len(X.shape) <= s < len(X.shape)
    ud = tuple(((s + len(X.shape)) % len(X.shape), k) for s, k in ud)
    ud_map = {}
    for s, k in ud:
        assert s not in ud_map
        assert d + 1 <= s
        assert k >= 1 and k % 2 == 1
        ud_map[s] = k

    # Shapes
    n = X.shape[0]
    nd = W.shape[0] if has_nd else 1

    # Reduce axes
    reduce_axes = []
    for i in range(0, d):
        assert X.shape[i + 1] == W.shape[i + nd_shift]
        reduce_axes.append(te.reduce_axis((0, X.shape[i + 1]), name=f'{name}_r{i}'))
    reduce_axes = tuple(reduce_axes)

    # Padding
    pads = []
    for i in range(len(X.shape)):
        pads.append((ud_map[i] - 1) // 2 if i in ud_map else 0)
    pads = tuple(pads)
    padded = topi.nn.pad(X, pads)

    # Unfold reduce axes
    unfold_reduce_axes = []
    unfold_reduce_axes_map = {}
    for i, (s, k) in enumerate(ud):
        axis = te.reduce_axis((0, k), name=f'{name}_ru{i}')
        unfold_reduce_axes.append(axis)
        unfold_reduce_axes_map[s] = axis
    unfold_reduce_axes = tuple(unfold_reduce_axes)

    # Compute
    return te.compute(
        (n, *((nd, ) if has_nd else ()), *X.shape[d + 1:]),
        lambda *i: te.sum(
            padded[(i[0], ) + reduce_axes + merge_axis(d + 1, i[1 + nd_shift:], unfold_reduce_axes_map, len(X.shape))] *
            W[((i[1], ) if has_nd else ()) + reduce_axes + unfold_reduce_axes],
            axis=reduce_axes + unfold_reduce_axes
        ),
        # attrs={'layout_free_placeholders': [W]},
        name=f'{name}_out',
    )


def grouped_fc(X, W, d, name: str = 'grouped_fc', keep_dim: bool = False):
    # `d` could be zero for grouping all the dimensions
    # X: [n, g, d * (reduce, ), *], W: [nd, d * (reduce, )]
    assert len(X.shape) >= d + 2
    assert len(W.shape) == d + 1
    n, g, nd = X.shape[0], X.shape[1], W.shape[0]
    assert g >= 1, f'Illegal group number {g}'
    assert nd % g == 0, f'Illegal output channel {nd} and groups {g}'
    nd_per_g = nd // g

    reduce_axes = []
    for i in range(0, d):
        assert X.shape[i + 2] == W.shape[i + 1]
        reduce_axes.append(te.reduce_axis((0, X.shape[i + 2]), name=f'{name}_r{i}'))
    reduce_axes = tuple(reduce_axes)

    if keep_dim:
        return te.compute(
            (n, g, nd_per_g, *X.shape[d + 2:]),
            lambda *i: te.sum(
                X[i[:2] + reduce_axes + i[3:]] * W[(i[1] * nd_per_g + i[2],) + reduce_axes],
                axis=reduce_axes
            ),
            name=f'{name}_out'
        )

    return te.compute(
        (n, nd, *X.shape[d + 2:]),
        lambda *i: te.sum(
            X[(i[0], i[1] // nd_per_g) + reduce_axes + i[2:]] * W[(i[1], ) + reduce_axes],
            axis=reduce_axes
        ),
        name=f'{name}_out'
    )


def fused_unfold_grouped_fc(X, W, d, ud: tuple, name: str = 'grouped_fc',
                            keep_dim: bool = False):
    # `d` could be zero for grouping all the dimensions
    # X: [n, g, d * (reduce, ), *], W: [nd, (d + unfold_dims) * (reduce, )]
    assert len(X.shape) >= d + 2 + len(ud)
    assert len(W.shape) == d + 1 + len(ud)

    # Convert unfolds to non-negatives
    for s, _ in ud:
        assert -len(X.shape) <= s < len(X.shape)
    ud = tuple(((s + len(X.shape)) % len(X.shape), k) for s, k in ud)
    ud_map = {}
    for s, k in ud:
        assert s not in ud_map
        assert d + 1 <= s
        assert k >= 1 and k % 2 == 1
        ud_map[s] = k

    # Shapes
    n, g, nd = X.shape[0], X.shape[1], W.shape[0]
    assert g >= 1, f'Illegal group number {g}'
    assert nd % g == 0
    nd_per_g = nd // g

    # Reduce axes
    reduce_axes = []
    for i in range(0, d):
        assert X.shape[i + 2] == W.shape[i + 1]
        reduce_axes.append(te.reduce_axis((0, X.shape[i + 2]), name=f'{name}_r{i}'))
    reduce_axes = tuple(reduce_axes)

    # Padding
    pads = []
    for i in range(len(X.shape)):
        pads.append((ud_map[i] - 1) // 2 if i in ud_map else 0)
    pads = tuple(pads)
    padded = topi.nn.pad(X, pads)

    # Unfold reduce axes
    unfold_reduce_axes = []
    unfold_reduce_axes_map = {}
    for i, (s, k) in enumerate(ud):
        axis = te.reduce_axis((0, k), name=f'{name}_ru{i}')
        unfold_reduce_axes.append(axis)
        unfold_reduce_axes_map[s] = axis
    unfold_reduce_axes = tuple(unfold_reduce_axes)

    if keep_dim:
        return te.compute(
            (n, g, nd_per_g, *X.shape[d + 2:]),
            lambda *i: te.sum(
                padded[i[:2] + reduce_axes +
                       merge_axis(d + 2, i[3:], unfold_reduce_axes_map, len(X.shape))]
                * W[(i[1] * nd_per_g + i[2],) + reduce_axes + unfold_reduce_axes],
                axis=reduce_axes + unfold_reduce_axes
            ),
            name=f'{name}_out'
        )

    return te.compute(
        (n, nd, *X.shape[d + 2:]),
        lambda *i: te.sum(
            padded[(i[0], i[1] // nd_per_g) + reduce_axes +
                   merge_axis(d + 2, i[2:], unfold_reduce_axes_map, len(X.shape))]
            * W[(i[1], ) + reduce_axes + unfold_reduce_axes],
            axis=reduce_axes + unfold_reduce_axes
        ),
        name=f'{name}_out'
    )


def fold_avg(X, d, keep_dims: bool = False):
    if isinstance(d, int):
        return topi.sum(X, d) / X.shape[d]
    pi = 1
    for i in d:
        pi *= X.shape[i]
    return topi.divide(topi.sum(X, d, keepdims=keep_dims), pi)


def fold_sum(X, d, keep_dims: bool = False):
    return topi.sum(X, d, keepdims=keep_dims)


def fold_max(X, d, keep_dims: bool = False):
    return topi.max(X, d, keepdims=keep_dims)


def unfold_1d_reindex(i, f, t):
    cut = i[:t] + i[t:][1:]
    return cut[:f] + (cut[f] + i[t], ) + cut[f:][1:]  # if `f == -1`, `cut[f+1:]` will not work


def unfold_1d(X, k, f, t, name='unfold_1d'):
    assert len(X.shape) >= 1
    assert k >= 1 and k % 2 == 1
    if k == 1:
        return reshape(X, (*X.shape[:t], k, *X.shape[t:]))

    p = (k - 1) // 2
    padded = padding_1d(X, f, p, name=f'{name}_padding_1d')
    return te.compute(
        (*X.shape[:t], k, *X.shape[t:]),
        lambda *i: padded[unfold_1d_reindex(i, f, t)],
        name=f'{name}_out'
    )


def unfold_2d(X, k, name='unfold_2d'):
    assert len(X.shape) >= 2
    assert k >= 1 and k % 2 == 1
    if k == 1:
        return reshape(X, (*X.shape[0:-2], k, k, *X.shape[-2:]))

    p = (k - 1) // 2
    h, w = X.shape[-2], X.shape[-1]
    padded = padding_2d(X, p, p, name=f'{name}_padding_2d')
    return te.compute(
        (*X.shape[0:-2], k, k, h, w),
        lambda *i: padded[i[:-4] + (i[-2] + i[-4], i[-1] + i[-3])],
        name=f'{name}_out'
    )


def shift_1d(X, d, name='shift_1d'):
    assert d < len(X.shape)
    return te.compute(
        X.shape,
        lambda *i: X[i[:d] + ((i[d] + 1) % X.shape[d],) + i[d:][1:]],
        name=f'{name}_out'
    )


def shift_2d(X, name='shift_2d'):
    assert len(X.shape) >= 2
    return te.compute(
        X.shape,
        lambda *i: X[i[:-2] + ((i[-2] + 1) % X.shape[-2], (i[-1] + 1) % X.shape[-1])],
        name=f'{name}_out'
    )


def softmax_1d(X, d, name='softmax_1d'):
    return topi.nn.softmax(X, d)


def softmax_2d(X, name='softmax_2d'):
    assert len(X.shape) >= 2
    shape = X.shape
    max_elem = topi.max(X, [-2, -1], keepdims=False)
    exp = te.compute(
        shape,
        lambda *i: te.exp(X[i] - max_elem[i[:-2]]),
        name=f'{name}_exp'
    )
    exp_sum = topi.sum(exp, [-2, -1], keepdims=False)
    return te.compute(
        shape,
        lambda *i: exp[i] / exp_sum[i[:-2]],
        name=f'{name}_out'
    )
