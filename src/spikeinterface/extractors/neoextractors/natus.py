from pathlib import Path

from spikeinterface.core.core_tools import define_function_from_class

from .neobaseextractor import NeoBaseRecordingExtractor


class NatusRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading a Natus/XLTEK (NeuroWorks) EEG session folder.

    Based on :py:class:`neo.rawio.NatusRawIO`

    The folder holds the ``.erd`` (samples), ``.etc``, ``.stc``, ``.snc`` and ``.ent`` files
    of one session. Neo gives one stream per sample rate (ids such as ``"512Hz"``) and one
    segment per continuous piece of recording. A segment is recorded at one rate only, so
    this extractor keeps the segments in which the selected stream has samples; their
    indices in the Neo reader are in the ``natus_neo_segment_indices`` annotation.

    Segment times (``get_start_time``) are in seconds from the start of the first piece of
    the session, whatever the stream; the absolute (UTC) start of the session is the
    ``session_start_time`` annotation (ISO 8601 string).

    Parameters
    ----------
    folder_path : str | Path
        The session folder.
    stream_id : str, default: None
        If the session has several sample rates, the stream id to load, e.g. "512Hz".
        See :py:func:`~spikeinterface.extractors.get_neo_streams` ("natus", folder_path).
    stream_name : str, default: None
        If the session has several sample rates, the stream name to load, e.g. "512 Hz (46 ch)".
    all_annotations : bool, default: False
        Load exhaustively all annotations from neo.
    use_names_as_ids : bool, default: False
        Determines the format of the channel IDs used by the extractor. If set to True, the channel IDs will be the
        names from NeoRawIO (the montage names, e.g. "LA1"). If set to False, the channel IDs will be the ids
        provided by NeoRawIO (the physical channel numbers as strings).
    """

    NeoRawIOClass = "NatusRawIO"
    mode = "folder"

    def __init__(
        self,
        folder_path: str | Path,
        stream_id: str | None = None,
        stream_name: str | None = None,
        all_annotations: bool = False,
        use_names_as_ids: bool = False,
    ):
        neo_kwargs = self.map_to_neo_kwargs(folder_path)
        NeoBaseRecordingExtractor.__init__(
            self,
            stream_id=stream_id,
            stream_name=stream_name,
            all_annotations=all_annotations,
            use_names_as_ids=use_names_as_ids,
            **neo_kwargs,
        )

        # keep only the segments recorded at this stream's rate
        neo_segment_indices = [seg.segment_index for seg in self._segments if seg.get_num_samples() > 0]
        self._segments[:] = [seg for seg in self._segments if seg.get_num_samples() > 0]

        block_annotations = self.neo_reader.raw_annotations["blocks"][self.block_index]
        rec_datetime = block_annotations.get("rec_datetime")
        if rec_datetime is not None:
            self.annotate(session_start_time=rec_datetime.isoformat())
        self.annotate(natus_neo_segment_indices=neo_segment_indices)

        self._kwargs.update(dict(folder_path=str(Path(folder_path).absolute())))
        self.extra_requirements.append("neo[natus]")

    @classmethod
    def map_to_neo_kwargs(cls, folder_path):
        neo_kwargs = {"dirname": str(folder_path)}
        return neo_kwargs


read_natus = define_function_from_class(source_class=NatusRecordingExtractor, name="read_natus")
