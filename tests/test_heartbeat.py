from assay.heartbeat import Heartbeat


def test_emit_appends_lines(tmp_path):
    p = tmp_path / "hb.log"
    hb = Heartbeat(str(p))
    hb.emit("quantize", "started")
    hb.emit("evaluate", "baseline done")
    lines = p.read_text().splitlines()
    assert len(lines) == 2
    assert "quantize" in lines[0] and "started" in lines[0]
    assert "evaluate" in lines[1]


def test_emit_redacts_secrets(tmp_path):
    p = tmp_path / "hb.log"
    hb = Heartbeat(str(p), secrets=["topsecret"])
    hb.emit("publish", "pushing with token topsecret now")
    text = p.read_text()
    assert "topsecret" not in text
    assert "***" in text


def test_emit_redacts_secret_at_every_occurrence(tmp_path):
    # Guards against a regression to .replace(secret, "***", 1), which would
    # leave later occurrences of the secret in the log.
    p = tmp_path / "hb.log"
    hb = Heartbeat(str(p), secrets=["topsecret"])
    hb.emit("publish", "topsecret then topsecret again and topsecret once more")
    text = p.read_text()
    assert "topsecret" not in text
    assert text.count("***") == 3


def test_emit_ignores_empty_secret(tmp_path):
    # Guards against removal of the `if s` filter in __init__: an empty
    # string in the secrets list must not match (and redact) everything.
    p = tmp_path / "hb.log"
    hb = Heartbeat(str(p), secrets=["", "realsecret"])
    hb.emit("publish", "pushing with token realsecret now")
    text = p.read_text()
    assert "realsecret" not in text
    assert "pushing with token *** now" in text
    assert "***" * 2 not in text


def test_emit_redacts_secret_in_stage(tmp_path):
    p = tmp_path / "hb.log"
    hb = Heartbeat(str(p), secrets=["topsecret"])
    hb.emit("publish-topsecret", "ok")
    text = p.read_text()
    assert "topsecret" not in text
    assert "***" in text


def test_emit_preserves_trailing_pipe_in_message(tmp_path):
    # Guards against the old `.rstrip(" |")` footgun, which silently
    # truncated any message ending in spaces or "|" characters.
    p = tmp_path / "hb.log"
    hb = Heartbeat(str(p))
    hb.emit("retry", "retry|")
    line = p.read_text().splitlines()[0]
    assert line.endswith("retry|")


def test_emit_is_ascii(tmp_path):
    p = tmp_path / "hb.log"
    hb = Heartbeat(str(p))
    hb.emit("done", "ok")
    assert p.read_text().isascii()


def test_emit_is_thread_safe(tmp_path):
    import threading
    from assay.heartbeat import Heartbeat
    hb = Heartbeat(str(tmp_path / "hb.log"))
    def worker():
        for _ in range(50):
            hb.emit("t", "x")
    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()
    lines = [ln for ln in (tmp_path / "hb.log").read_text().splitlines() if ln]
    assert len(lines) == 200  # no interleaved/dropped lines
