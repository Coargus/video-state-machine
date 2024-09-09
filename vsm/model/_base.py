import abc

from cog_cv_abstraction.schema.video_frame import VideoFrame


class VideoModel(abc.ABC):
    @abc.abstractmethod
    def set_up(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def add_frame(self, frame: VideoFrame) -> None:
        raise NotImplementedError
