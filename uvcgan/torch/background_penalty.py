from torch import nn

class BackgroundPenaltyReduction(nn.Module):

    def __init__(self, epochs_warmup, epochs_anneal, **kwargs):
        super().__init__(**kwargs)

        self._epochs_warmup = epochs_warmup
        self._epochs_anneal = epochs_anneal

        self._alpha = 0

    def end_epoch(self, epoch):
        if epoch is None:
            self._alpha = 1
            return

        if epoch < self._epochs_warmup:
            self._alpha = 0
            return

        progression = (epoch - self._epochs_warmup)

        if progression < self._epochs_anneal:
            self._alpha = progression / self._epochs_anneal
        else:
            self._alpha = 1

    def forward(self, fake, real):
        if self._alpha == 1:
            return fake

        result = fake - (1 - self._alpha) * fake * (real == 0)
        return result

