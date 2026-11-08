"""A string argument that quotes an unclosed call opener, `<function=tools NAME>` or `<toolcall>` stays exact, and the call is read.

Found in review of the 0.2.12 parser picks: an unclosed quoted `<function=X> <parameter=Y>` in a value made param_end skip the
call's own close (the call came back as text), and the #5409 / #5276 rewrites changed a value's text.

    python -m unittest serve.test_quoted_drift_values -v
"""
import json
import unittest

from serve.frontend import CALL_END, CALL_START, FUNC_END, FUNC_START, PARAM_END, PARAM_START, OutputParser

WRITE = {"name": "write", "parameters": {"properties": {"path": {"type": "string"}, "content": {"type": "string"}}}}
BASH = {"name": "bash", "parameters": {"properties": {"command": {"type": "string"}}}}


def call(name, params):
    s = f"{CALL_START}\n{FUNC_START}{name}>\n"
    for k, v in params.items():
        s += f"{PARAM_START}{k}>\n{v}\n{PARAM_END}\n"
    return s + f"{FUNC_END}\n{CALL_END}"


CASES = [
    ("unclosed quoted opener, one line", WRITE, "write",
     {"path": "a.txt", "content": f"example: {FUNC_START}demo> {PARAM_START}x>1"}),
    ("unclosed quoted opener, lines", WRITE, "write",
     {"path": "a.txt", "content": f"example:\n{FUNC_START}demo>\n{PARAM_START}x>\n1"}),
    ("spaced prefix in a value", BASH, "bash", {"command": "echo '<function=tools bash>' done"}),
    ("misspelt opener in a value", BASH, "bash", {"command": f"echo '<toolcall>' && echo '{FUNC_START}bash>'"}),
]


class QuotedDriftValues(unittest.TestCase):
    def test_values_are_exact_whole_streamed_and_chunked(self):
        for label, tool, name, args in CASES:
            text = call(name, args)
            for stream in (False, True):
                for size in (1, 7, len(text)):
                    with self.subTest(label=label, stream=stream, size=size):
                        p = OutputParser(thinking=False, tools=[tool], stream_tools=stream)
                        events = []
                        for i in range(0, len(text), size):
                            events += p.feed(text[i:i + size])
                        events += p.finish()
                        self.assertEqual([e.call.arguments for e in events if e.kind == "tool_call"], [args])
                        if stream:
                            self.assertEqual(json.loads("".join(e.text for e in events if e.kind == "tool_args")), args)


if __name__ == "__main__":
    unittest.main()
