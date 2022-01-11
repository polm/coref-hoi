"""Subnetwork that takes a ragged sequence like:

    [
        [
          seq1_vec1,
          seq1_vec2
        ],
        [
          seq2_vec1,
          seq2_vec2,
          seq2_vec3,
          seq2_vec4
        ],
        [
          seq3_vec1,
          seq3_vec2,
          seq3_vec3
        ]
    ]

    And creates a sequence of square matrices with the dot probabilities
    of the vectors within each sequence, so:

    [
      [
        s2v1 * s2v1, s1v1 * s1v2,
        s1v2 * s1v1, s2v2 * s2v2
      ],
      [
        s2v1 * s2v1, s2v1 * s2v2, s2v1 * s2v3, s2v2 * s2v4,
        ...
      ],
      [...]
    ]

    Two solutions are suggested:

        1) A dirty, filthy for loop (gasp).
        2) Convert to padded, use tensordot.
"""

def self_dot(algorithm: str="loop") -> Model[Ragged, Ragged]:
    if algorithm == "loop":
        return with_list(Model("self-dot", self_dot_loop))
    else:
        return with_padded(Model("self-dot", self_dot_padded))


def self_dot_loop(model: Model[List[Floats2d], List[Floats1d]], Xs: List[Floats2d], is_train: bool):
    Ys = [model.ops.gemm(X, X, trans2=True).ravel() for X in Xs]

    def self_dot_loop_backprop(dYs: List[Floats1d]) -> List[Floats2d]:
        dXs = []
        for X, dY_flat in zip(Xs, dYs:
            length = int(math.sqrt(dY_flat.size))
            dY = model.ops.reshape2f(dY_flat, length, length)
            dX = model.ops.gemm(X, dY, trans1=True)
            dXs.append(dX)
        return dXs

    return Ys, self_dot_loop_backprop


def self_dot_padded(model: Model[Padded, Padded], Xp: Padded, is_train: bool):
    X = cast(Floats3d, Xp.dataXd)
    # Ugh, can't get the tensordot right here? The einsum is super slow. There
    # must be a way to express this.
    Y = numpy.einsum('abc,def->abe', X, X)
    Yp = Padded(Y, Xp.size_at_t, Xp.lengths, Xp.indices)

    def self_dot_padded_backprop(dYp: Padded) -> Padded:
        dY = dYp.dataXd
        # If we had (2, 3, 4) and (2, 3, 3) we'd be doing:
        # dX[0] = gemm(dY[0], X[0].T)
        # dX[1] = gemm(dY[1], X[1].T)
        dX = numpy.einsum('abc,def->abf', dY, X)
        return Padded(dX, Xp.size_at_t, Xp.lengths, Xp.indices)

    return Yp, self_dot_padded_backprop
