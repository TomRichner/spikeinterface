"""
Tests of NatusRecordingExtractor on local Natus/XLTEK sample sessions.

The sample data is not public, so the tests are skipped unless these environment
variables are set (the same as for neo's test_natusrawio.py):

* ``NATUS_TEST_ROOT``: folder holding ``Natus_deident1`` and ``Natus_deident2``.
* ``NATUS_DEV_ROOT`` (optional): folder holding the de-identified development sessions;
  used for the session whose sample rate changes between files.
"""

import os
from pathlib import Path

import numpy as np
import pytest

SAMPLE_ROOT = os.environ.get("NATUS_TEST_ROOT")
DEV_ROOT = os.environ.get("NATUS_DEV_ROOT")

SAMPLES = {
    "426dd159": "Natus_deident1/Xxxxx~ Xxxxxx_426dd159-4f07-4447-918c-564f55d82d13",
    "85ba4df1": "Natus_deident1/Xxxx~ Xxxx_85ba4df1-98ef-49fc-9e67-ae3d1f5ad0b3",
    "662aafc9": "Natus_deident2/Xxxxxx~ Xxxx_662aafc9-f1ff-4a91-bc55-0102a0e2db27",
    "42a53fa4": "Natus_deident2/Xxxxxxxx~ Xxxx_42a53fa4-f077-4d5b-bcb3-6446aecae1c1",
}
# (sample rate, channel count, segment count)
EXPECTED = {
    "426dd159": (512.0, 276, 3),
    "662aafc9": (4096.0, None, 1),
    "42a53fa4": (256.0, 46, 24),
}
RATE_CHANGE_SESSION = "LastNameA~ FirstNameA_c0fe2d81-a1d2-4da9-818d-481ecbe0a4b1"

pytestmark = [
    pytest.mark.extractors,
    pytest.mark.skipif(not SAMPLE_ROOT, reason="NATUS_TEST_ROOT not set"),
]


def session_dir(key):
    if key == "c0fe2d81":
        if not DEV_ROOT:
            pytest.skip("NATUS_DEV_ROOT not set")
        d = Path(DEV_ROOT) / RATE_CHANGE_SESSION
    else:
        d = Path(SAMPLE_ROOT) / SAMPLES[key]
    if not d.is_dir():
        pytest.skip(f"session folder missing: {d}")
    return d


def neo_reader(d):
    from neo.rawio import NatusRawIO

    r = NatusRawIO(dirname=d)
    r.parse_header()
    return r


@pytest.mark.parametrize("key", list(SAMPLES))
def test_matches_neo(key):
    from spikeinterface.extractors.neoextractors import NatusRecordingExtractor

    d = session_dir(key)
    r = neo_reader(d)
    rec = NatusRecordingExtractor(d)

    stream_index = 0
    mask = r.header["signal_channels"]["stream_id"] == r.header["signal_streams"]["id"][stream_index]
    channels = r.header["signal_channels"][mask]
    assert list(rec.channel_ids) == list(channels["id"])
    assert list(rec.get_property("channel_name")) == list(channels["name"])
    assert rec.get_sampling_frequency() == r.get_signal_sampling_rate(stream_index)
    assert rec.get_num_segments() == r.segment_count(0)
    assert rec.get_dtype() == np.dtype("int32")
    np.testing.assert_allclose(rec.get_channel_gains(), channels["gain"])
    assert rec.get_annotation("natus_neo_segment_indices") == list(range(r.segment_count(0)))
    assert rec.get_annotation("session_start_time") == r.raw_annotations["blocks"][0]["rec_datetime"].isoformat()

    for seg in range(rec.get_num_segments()):
        assert rec.get_num_samples(seg) == r.get_signal_size(0, seg, stream_index)
        assert rec.get_start_time(seg) == pytest.approx(r.get_signal_t_start(0, seg, stream_index))

    # a slice in the middle of the last segment, a few channels
    seg = rec.get_num_segments() - 1
    n = rec.get_num_samples(seg)
    a, b = n // 3, min(n // 3 + 3000, n)
    chan_idx = [0, 2, 5]
    got = rec.get_traces(segment_index=seg, start_frame=a, end_frame=b, channel_ids=rec.channel_ids[chan_idx])
    expected = r.get_analogsignal_chunk(0, seg, a, b, stream_index=stream_index, channel_indexes=chan_idx)
    np.testing.assert_array_equal(got, expected)

    scaled = rec.get_traces(segment_index=seg, start_frame=a, end_frame=b, return_in_uV=True)
    raw = r.get_analogsignal_chunk(0, seg, a, b, stream_index=stream_index)
    uv = r.rescale_signal_raw_to_float(raw, dtype="float32", stream_index=stream_index)
    np.testing.assert_allclose(scaled, uv, rtol=1e-6)


@pytest.mark.parametrize("key", list(EXPECTED))
def test_known_values(key):
    from spikeinterface.extractors import read_natus

    rate, n_chan, n_seg = EXPECTED[key]
    rec = read_natus(session_dir(key))
    assert rec.get_sampling_frequency() == rate
    if n_chan is not None:
        assert rec.get_num_channels() == n_chan
    assert rec.get_num_segments() == n_seg


def test_names_as_ids():
    from spikeinterface.extractors.neoextractors import NatusRecordingExtractor

    rec = NatusRecordingExtractor(session_dir("85ba4df1"), use_names_as_ids=True)
    assert rec.channel_ids[0] == "LA1"


def test_registered():
    from spikeinterface.extractors import get_neo_streams, recording_extractor_full_dict, read_natus

    assert recording_extractor_full_dict["natus"] is read_natus
    names, ids = get_neo_streams("natus", session_dir("42a53fa4"))
    assert ids == ["256Hz"]


def test_dump_load_roundtrip():
    from spikeinterface.core import load
    from spikeinterface.extractors.neoextractors import NatusRecordingExtractor

    rec = NatusRecordingExtractor(session_dir("42a53fa4"))
    rec2 = load(rec.to_dict())
    assert rec2.get_num_segments() == rec.get_num_segments()
    np.testing.assert_array_equal(
        rec2.get_traces(segment_index=3, start_frame=0, end_frame=100),
        rec.get_traces(segment_index=3, start_frame=0, end_frame=100),
    )


def test_rate_change_session_per_stream():
    """c0fe2d81 has 256 Hz and 512 Hz files: each stream is its own recording."""
    from spikeinterface.extractors import get_neo_streams
    from spikeinterface.extractors.neoextractors import NatusRecordingExtractor

    d = session_dir("c0fe2d81")
    r = neo_reader(d)
    names, ids = get_neo_streams("natus", d)
    assert sorted(ids) == ["256Hz", "512Hz"]

    with pytest.raises(ValueError):
        NatusRecordingExtractor(d)  # several streams: one must be chosen

    total = 0
    for stream_index, (stream_name, stream_id) in enumerate(zip(names, ids)):
        rec = NatusRecordingExtractor(d, stream_id=stream_id)
        rec_by_name = NatusRecordingExtractor(d, stream_name=stream_name)
        assert rec_by_name.get_num_segments() == rec.get_num_segments()
        rate = float(stream_id[: -len("Hz")])
        assert rec.get_sampling_frequency() == rate
        neo_segs = [s for s in range(r.segment_count(0)) if r.get_signal_size(0, s, stream_index) > 0]
        assert rec.get_annotation("natus_neo_segment_indices") == neo_segs
        assert rec.get_num_segments() == len(neo_segs) > 0
        total += rec.get_num_segments()
        for seg, neo_seg in enumerate(neo_segs):
            assert rec.get_num_samples(seg) == r.get_signal_size(0, neo_seg, stream_index)
            assert rec.get_start_time(seg) == pytest.approx(r.get_signal_t_start(0, neo_seg, stream_index))
        n = min(2000, rec.get_num_samples(0))
        got = rec.get_traces(segment_index=0, start_frame=0, end_frame=n)
        expected = r.get_analogsignal_chunk(0, neo_segs[0], 0, n, stream_index=stream_index)
        np.testing.assert_array_equal(got, expected)
    assert total == r.segment_count(0)
