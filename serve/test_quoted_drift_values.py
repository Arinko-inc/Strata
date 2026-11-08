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
    ("spaced prefix after a closed quoted call", WRITE, "write",
     {"path": "a.txt", "content": f"{FUNC_START}demo>{PARAM_START}x>1{PARAM_END}{FUNC_END}\n{FUNC_START}tools write>"}),
    ("misspelt opener after a closed quoted call", BASH, "bash",
     {"command": f"cat <<E\n{FUNC_START}demo>\n{PARAM_START}x>\n1\n{PARAM_END}\n{FUNC_END}\n<toolcall>\n{FUNC_START}bash>\nE"}),
]


def stray_call():
    # text between a close and the next parameter: parse_tool_call reads on to that parameter, and so does _in_value
    return (f"{CALL_START}\n{FUNC_START}write>\n{PARAM_START}path>\na.txt\n{PARAM_END}\n{FUNC_END} stray note\n"
            f"{PARAM_START}content>\nplain text\n{FUNC_START}tools write>\nmore\n{PARAM_END}\n{FUNC_END}\n{CALL_END}")


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

    def parse(self, text, tools, stream, size):
        p = OutputParser(thinking=False, tools=tools, stream_tools=stream, recover=True)
        events = []
        for i in range(0, len(text), size):
            events += p.feed(text[i:i + size])
        return events + p.finish()

    def test_value_after_stray_text_stays_exact(self):
        text = stray_call()
        for stream in (False, True):
            for size in (1, 7, len(text)):
                with self.subTest(stream=stream, size=size):
                    for e in self.parse(text, [WRITE], stream, size):
                        if e.kind == "tool_call":
                            self.assertIn(f"{FUNC_START}tools write>", e.call.arguments.get("content", ""))

    def test_second_spaced_call_of_a_batch_is_the_declared_tool(self):
        actio = {"name": "actio", "parameters": {"properties": {"command": {"type": "string"}}}}
        read = {"name": "read", "parameters": {"properties": {"path": {"type": "string"}}}}
        text = (f"{CALL_START}\n{FUNC_START}actio>\n{PARAM_START}command>\nls\n{PARAM_END}\n{FUNC_END}\n"
                f"{FUNC_START}tools read>\n{PARAM_START}path>\n/x\n{PARAM_END}\n{FUNC_END}\n{CALL_END}")
        for stream in (False, True):
            for size in (1, 7, len(text)):
                with self.subTest(stream=stream, size=size):
                    events = self.parse(text, [actio, read], stream, size)
                    self.assertEqual([e.call.name for e in events if e.kind == "tool_call"], ["actio", "read"])

    def test_bare_spaced_call_at_a_line_start(self):
        actio = {"name": "actio", "parameters": {"properties": {"command": {"type": "string"}}}}
        text = f"Let me look.\n{FUNC_START}tools actio>\n{PARAM_START}command>\nls\n{PARAM_END}\n{FUNC_END}"
        for size in (1, 7, len(text)):
            with self.subTest(size=size):
                events = self.parse(text, [actio], False, size)
                self.assertEqual([(e.call.name, e.call.arguments) for e in events if e.kind == "tool_call"],
                                 [("actio", {"command": "ls"})])

    def test_unclosed_quote_across_parameters_never_makes_wrong_arguments(self):
        cases = [
            call("write", {"path": "a.txt", "content": f"docs {FUNC_START}demo>\n{PARAM_START}x>\n1\n{PARAM_END}\n{PARAM_START}y>\n2"}),
            call("write", {"path": f"a {FUNC_START}demo>\n{PARAM_START}x>\n1", "content": f"doc {FUNC_START}foo>{FUNC_END}"}),
        ]
        for n, text in enumerate(cases):
            for stream in (False, True):
                for size in (1, 7, len(text)):
                    with self.subTest(case=n, stream=stream, size=size):
                        calls = [e.call.arguments for e in self.parse(text, [WRITE], stream, size) if e.kind == "tool_call"]
                        # either the call as written, or (as before these picks) no call: never a call with other arguments
                        for args in calls:
                            self.assertEqual(set(args), {"path", "content"})
                            self.assertTrue(args["content"].endswith(("2", f"{FUNC_START}foo>{FUNC_END}")), args)


if __name__ == "__main__":
    unittest.main()
