from torch import nn
from torch.nn import functional as F
import torch
import numpy as np
import random

from scipy.stats import norm, binomtest
from statsmodels.stats.proportion import proportion_confint
from datetime import datetime

class Smooth(object):
    """A smoothed classifier g """
    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, rate: float, input_func):
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.rate = rate
        self.input_func = input_func

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int, shuffle_type) -> (int, float):
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(x, n0, batch_size, shuffle_type)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, batch_size, shuffle_type)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0
        else:
            return cAHat, pABar
            # radius = self.sigma * norm.ppf(pABar)
            # return cAHat, radius

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int, shuffle_type) -> int:
        counts = self._sample_noise(x, n, batch_size, shuffle_type)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binomtest(count1, count1 + count2, p=0.5) > alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]

    def _sample_noise(self, sample, num: int, batch_size: int, shuffle_type) -> np.ndarray:
        samples_noised = []
        time1 = datetime.now()
        for _ in range(num):
            samples_noised.append(self.sample_with_noise(sample, self.rate, shuffle_type))
        time2 = datetime.now()
        predictions = np.array([], dtype=np.int_)
        for i in range(0, len(samples_noised), batch_size):
            y_pred_tmp = self.base_classifier.predict(
                self.input_func(samples_noised[i:i + batch_size]))
            predictions = np.concatenate([predictions, y_pred_tmp])
        time3 = datetime.now()
        counts = self._count_arr(predictions, self.num_classes)
        # print((time2-time1).total_seconds(),(time3-time2).total_seconds())
        return counts

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]

    def sample_with_noise(self, ps, rate, shuffle_type):
        if rate <= 0:
            return ps
        if shuffle_type == 'loss':
            return self.ps_loss(ps, rate)
        elif shuffle_type == 'retrans':
            return self.ps_retrans(ps, rate)
        elif shuffle_type == 'disorder':
            return self.ps_disorder(ps, rate)

    def ps_loss(self, ps_org, rate):
        while True:
            ps_noise = []
            for token in ps_org.split():
                if random.random() < rate:
                    continue
                ps_noise.append(token)
            if len(ps_noise)>1:
                return ' '.join(ps_noise)
    def ps_retrans(self, ps_org, rate):
        ps_noise = []
        for token in ps_org.split():
            ps_noise.append(token)
            if random.random() < rate:
                ps_noise.append(token)
        return ' '.join(ps_noise)
    def ps_disorder(self, ps_org, rate):
        ps_noise = ps_org.split()
        for i in range(len(ps_noise)-1):
            if random.random() < rate:
                ps_noise[i], ps_noise[i+1] = ps_noise[i+1], ps_noise[i]
        return ' '.join(ps_noise)
