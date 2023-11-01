from abc import ABC, abstractmethod


class KTDatasetBase(ABC):
    @abstractmethod
    def get_sequences(self):
        pass

    @abstractmethod
    def get_users(self):
        pass

    @abstractmethod
    def get_skill_name(self,sid):
        pass


    @abstractmethod
    def get_features(self):
        pass

    @abstractmethod
    def get_q_matrix(self):
        pass







