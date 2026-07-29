"""reader_loop.sh is thin bash exercised live, so what is test-worthy is its
CONTRACT (same family as the Dockerfile two-sided contract test): the single
guarded DELETE call site, the no-volume-writes discipline, the pinned pip
install, and valid syntax."""
import pathlib
import re
import subprocess

LOOP = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "reader_loop.sh"
TEXT = LOOP.read_text()


def test_bash_syntax():
    subprocess.run(["bash", "-n", str(LOOP)], check=True)


def test_exactly_one_delete_call_site_targeting_own_pod():
    # python:3.12-slim has no curl (whole-branch review finding); the self-delete
    # is a python3 stdlib urllib heredoc instead. Tie the assertions to the
    # heredoc body, not to substrings elsewhere in the file.
    deletes = re.findall(r'method="DELETE"', TEXT)
    assert len(deletes) == 1, deletes
    heredoc = re.search(r"<<'PY'\n(.*?)\nPY\n", TEXT, re.S)
    assert heredoc, "expected the self-delete python3 heredoc"
    body = heredoc.group(1)
    assert "RUNPOD_POD_ID" in body        # own id only
    assert "MAIN_POD_ID" not in body      # never the burn


def test_no_writes_to_volume():
    # Crude by design (F-032 family: pin the property in text): no redirection
    # into /runpod-volume anywhere in the loop.
    assert not re.search(r">>?\s*/runpod-volume", TEXT)
    for verb in ("cp ", "mv ", "rm ", "mkdir", "touch", "truncate"):
        for m in re.finditer(re.escape(verb) + r"[^\n]*", TEXT):
            assert "/runpod-volume" not in m.group(0), m.group(0)


def test_pinned_hub_install():
    assert "huggingface_hub==1.23.0" in TEXT   # R-13
    assert not re.search(r"pip install[^\n]*huggingface_hub(?!==)", TEXT)


def test_delete_has_graphql_fallback_transport():
    # F-042 (07-28 drill): rest.runpod.io 403'd origin-dependently from in-pod for
    # the entire drill window while api.runpod.io/graphql worked (diag pod proof,
    # dataset runs/_diag/rest403/) - a REST-only self-delete retried forever with
    # no TTL rescue (F-044 made the TTL inert too). The heredoc must carry the
    # burn-proven GraphQL podTerminate as fallback, REST DELETE staying primary,
    # own pod id only on both transports.
    heredoc = re.search(r"<<'PY'\n(.*?)\nPY\n", TEXT, re.S)
    assert heredoc, "expected the self-delete python3 heredoc"
    body = heredoc.group(1)
    assert "rest.runpod.io" in body
    assert "api.runpod.io/graphql" in body
    assert "podTerminate" in body
    assert body.index("rest.runpod.io") < body.index("api.runpod.io/graphql")
    assert "not found" in body    # already-terminated tolerated as success
    assert "MAIN_POD_ID" not in body


def test_delete_retries_and_never_exits_while_failing():
    # R-4: the DELETE must sit directly inside a retry loop - an exit on a
    # failed DELETE triggers RunPod's restart-billing loop. Anchor on the
    # `until python3` line through its matching `done`; the heredoc body sits
    # between `<<'PY'` and `PY`, and the retry-loop body between `do` and
    # `done`. A regression that drops the retry structure or retargets the
    # pod id must fail this test, not just the sibling test above.
    m = re.search(r"^\s*until python3[^\n]*$", TEXT, re.M)
    assert m, "self-delete python3 call must be the condition of an until retry loop"
    body = TEXT[m.end():TEXT.index("done", m.end())]
    assert "RUNPOD_POD_ID" in body, "retry structure must still target own pod id"
    assert "exit" not in body, "the DELETE retry body must never exit"
