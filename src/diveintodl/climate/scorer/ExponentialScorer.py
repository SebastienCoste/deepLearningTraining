from src.diveintodl.climate.scorer.Scorer import Scorer
from math import log


class ExponentialScorer(Scorer):

    def __init__(self, scorer: Scorer):
        self.internalScorer = scorer
        self.scores = []
        self.concluded = False
        self.minn, self.maxx, self.summ = 0

    def init(self):
        self.internalScorer.init()

    def score(self, point):
        score = self.internalScorer.score(point)
        self.scores.append(score)

    def conclude_scoring(self):
        self.internalScorer.conclude_scoring()

        self.minn, self.maxx, self.summ = self.scores[0]
        for n, v in enumerate(self.scores):
            self.scores[n] = self.internalScorer.correct_score(v)

        score_sorted = self.scores.copy()
        score_sorted.sort()

        idxmin = int(len(score_sorted) * 0.1)
        for i in range(int(len(score_sorted) * 0.8)):
            self.minn = min(self.minn, score_sorted[i + idxmin])
            self.maxx = max(self.maxx, score_sorted[i + idxmin])
            self.summ += score_sorted[i + idxmin]

        self.concluded = True
        pass

    def correct_score(self, score):
        if self.concluded:
            return 100 * log(99 * max(0, (score - self.minn) / (self.maxx - self.minn)) + 1) / log(100)
        else:
            self.conclude_scoring()
            return self.correct_score(score)


