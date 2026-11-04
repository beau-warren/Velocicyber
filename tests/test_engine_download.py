import httpx
import pytest

from velocicyber.engine.downloads import download


def test_ranges_resume_saved_prefix_and_assemble_exactly(tmp_path):
    data = b"abcdefghijklmnopqrstuvwxy"
    target = tmp_path / "download.partial"
    target.write_bytes(data[:11])
    requests = []

    def serve(request):
        assert request.headers["accept-encoding"] == "identity"
        start, end = map(int, request.headers["range"].removeprefix("bytes=").split("-"))
        requests.append((start, end))
        return httpx.Response(206, content=data[start:end + 1], headers={
            "content-range": f"bytes {start}-{end}/{len(data)}",
        })

    with httpx.Client(transport=httpx.MockTransport(serve)) as client:
        download(client, "https://example.test/bundle", target, len(data), lambda _: None, chunk_size=8)
    assert target.read_bytes() == data
    assert sorted(requests) == [(11, 15), (16, 23), (24, 24)]


@pytest.mark.parametrize("status,header", [(200, ""), (206, "bytes 1-8/8")])
def test_rejects_ignored_or_wrong_range_before_assembly(tmp_path, status, header):
    target = tmp_path / "download.partial"
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(
        status, content=b"abcdefgh", headers={"content-range": header},
    ))) as client:
        with pytest.raises(RuntimeError, match="Unexpected HTTP range"):
            download(client, "https://example.test/bundle", target, 8, lambda _: None, chunk_size=8)
    assert not target.exists()


def test_oversized_response_is_not_adopted(tmp_path):
    target = tmp_path / "download.partial"
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(
        206, content=b"too many bytes", headers={"content-range": "bytes 0-3/4"},
    ))) as client:
        with pytest.raises(RuntimeError, match="exceeded"):
            download(client, "https://example.test/bundle", target, 4, lambda _: None, chunk_size=4)
    assert not target.exists()
