from abc import ABC, abstractmethod


class KTDatasetBase(ABC):
    @abstractmethod
    def get_sequences(self):
        pass

    @abstractmethod
    def get_users(self):
        pass

    @abstractmethod
    def get_skill_question_mappings(self):
        pass

    @abstractmethod
    def get_features(self):
        pass






