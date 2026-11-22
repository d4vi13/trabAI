import nn


class PerceptronModel:
    def __init__(self, dimensions):
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        return self.w

    def run(self, x):
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        return 1 if nn.as_scalar(self.run(x)) >= 0 else -1

    def train(self, dataset):
        while True:
            updated = False
            for x, y in dataset.iterate_once(1):
                lbl = nn.as_scalar(y)
                if self.get_prediction(x) != lbl:
                    self.w.update(x, lbl)
                    updated = True
            if not updated:
                break


class RegressionModel:
    def __init__(self):
        hiddenDim = 512
        self.batchSize = 200
        self.lr = 0.05

        self.w1 = nn.Parameter(1, hiddenDim)
        self.b1 = nn.Parameter(1, hiddenDim)
        self.w2 = nn.Parameter(hiddenDim, 1)
        self.b2 = nn.Parameter(1, 1)
        self.weights = [self.w1, self.b1, self.w2, self.b2]

    def run(self, x):
        hid = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        return nn.AddBias(nn.Linear(hid, self.w2), self.b2)

    def get_loss(self, x, y):
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        while True:
            lossSum = 0.0
            batches = 0
            for x, y in dataset.iterate_once(self.batchSize):
                loss = self.get_loss(x, y)
                lossSum += nn.as_scalar(loss)
                batches += 1
                for w, g in zip(self.weights, nn.gradients(loss, self.weights)):
                    w.update(g, -self.lr)
            if lossSum / batches < 0.015:
                break


class DigitClassificationModel:
    def __init__(self):
        hiddenDim = 200
        self.batchSize = 100
        self.lr = 0.5

        self.w1 = nn.Parameter(784, hiddenDim)
        self.b1 = nn.Parameter(1, hiddenDim)
        self.w2 = nn.Parameter(hiddenDim, 10)
        self.b2 = nn.Parameter(1, 10)
        self.weights = [self.w1, self.b1, self.w2, self.b2]

    def run(self, x):
        hid = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        return nn.AddBias(nn.Linear(hid, self.w2), self.b2)

    def get_loss(self, x, y):
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        for _ in range(20):
            for x, y in dataset.iterate_once(self.batchSize):
                loss = self.get_loss(x, y)
                for w, g in zip(self.weights, nn.gradients(loss, self.weights)):
                    w.update(g, -self.lr)
            # margin above 97% test cutoff
            if dataset.get_validation_accuracy() >= 0.98:
                break


class LanguageIDModel:
    def __init__(self):
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        hiddenDim = 256
        nLangs = len(self.languages)
        self.batchSize = 100
        self.lr = 0.1

        self.wIn = nn.Parameter(self.num_chars, hiddenDim)
        self.wRec = nn.Parameter(hiddenDim, hiddenDim)
        self.bRec = nn.Parameter(1, hiddenDim)
        self.wOut1 = nn.Parameter(hiddenDim, hiddenDim)
        self.bOut1 = nn.Parameter(1, hiddenDim)
        self.wOut2 = nn.Parameter(hiddenDim, nLangs)
        self.bOut2 = nn.Parameter(1, nLangs)
        self.weights = [
            self.wIn, self.wRec, self.bRec,
            self.wOut1, self.bOut1, self.wOut2, self.bOut2,
        ]

    def run(self, xs):
        state = nn.ReLU(nn.AddBias(nn.Linear(xs[0], self.wIn), self.bRec))
        for ch in xs[1:]:
            step = nn.Add(nn.Linear(ch, self.wIn), nn.Linear(state, self.wRec))
            state = nn.ReLU(nn.AddBias(step, self.bRec))
        proj = nn.ReLU(nn.AddBias(nn.Linear(state, self.wOut1), self.bOut1))
        return nn.AddBias(nn.Linear(proj, self.wOut2), self.bOut2)

    def get_loss(self, xs, y):
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        for _ in range(30):
            for xs, y in dataset.iterate_once(self.batchSize):
                loss = self.get_loss(xs, y)
                for w, g in zip(self.weights, nn.gradients(loss, self.weights)):
                    w.update(g, -self.lr)
            # margin above 81% test cutoff
            if dataset.get_validation_accuracy() >= 0.89:
                break
