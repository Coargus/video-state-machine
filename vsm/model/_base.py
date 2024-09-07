import abc


class VideoModel(abc.ABC):
    @abc.abstractmethod
    def set_up(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self) -> None:
        raise NotImplementedError
