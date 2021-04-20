from thinc.api import Model, Linear, Relu, chain, Softmax
from thinc.types import Floats1d, Floats2d

from typing import Callable, Tuple, cast

from coref_thinc import batch_select

from icecream import ic

InT = Floats2d
OutT = Floats2d

def build_topk(k: int = 5):
    return Model("TopK", forward=topk_forward, attrs={"k": k})

def topk_forward(model: Model, X, is_train: bool) -> Tuple[Floats2d, Callable]:
    xp = model.ops.xp
    k = model.attrs["k"]


    topk_idx = xp.argpartition(X, -k)[:,-k:]
    topk_vals = batch_select(xp, X, topk_idx)

    length = X.shape[0]

    def backward(dY: OutT) -> InT:
        dX = model.ops.alloc1f(length)
        dX[topk_idx] = dY
        dX = xp.expand_dims(dX, 1)
        return dX

    return topk_vals, backward
    

if __name__ == "__main__":
    import ml_datasets

    (train_X, train_Y), (dev_X, dev_Y) = ml_datasets.mnist()

    n_hidden = 64
    dropout = 0.2
    model = chain(
            build_topk(16), 
            Relu(nO=n_hidden, dropout=dropout), 
            Relu(nO=n_hidden, dropout=dropout), 
            Softmax()
            )

    train_X = model.ops.asarray(train_X)
    train_Y = model.ops.asarray(train_Y)
    dev_X = model.ops.asarray(dev_X)
    dev_Y = model.ops.asarray(dev_Y)

    model.initialize(X=train_X[:5], Y=train_Y[:5])
    nI = model.get_dim("nI")
    nO = model.get_dim("nO")
    print(f"Initialized model with input dimension nI={nI} and output dimension nO={nO}")

    from thinc.api import Adam, fix_random_seed
    from tqdm import tqdm

    fix_random_seed(23)
    optimizer = Adam(0.001)
    batch_size = 128

    for i in range(100):
        batches = model.ops.multibatch(batch_size, train_X, train_Y, shuffle=True)
        for X, Y in tqdm(batches):
            Yh, backprop = model.begin_update(X)
            dY = Yh - Y

            backprop(Yh - Y)
            model.finish_update(optimizer)
        correct = 0
        total = 0
        for X, Y in model.ops.multibatch(batch_size, dev_X, dev_Y):
            Yh = model.predict(X)
            correct += (Yh.argmax(axis=1) == Y.argmax(axis=1)).sum()
            total += Yh.shape[0]

        score = correct / total
        print(f" {i} accuracy: {float(score):.3f}")
