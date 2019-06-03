from abc import ABCMeta, abstractmethod

from src.diveintodl.climate.Point import Point


class Scorer(metaclass=ABCMeta):

    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def score(self, point: Point):
        pass

    @abstractmethod
    def conclude_scoring(self):
        pass

    @abstractmethod
    def correct_score(self, score):
        return score

