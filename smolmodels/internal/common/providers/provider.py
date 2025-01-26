import abc


class Provider(abc.ABC):
    @abc.abstractmethod
    def query(self, system_message: str, user_message: str, response_format=None) -> str:
        pass
