"""A string argument that quotes an unclosed call opener stays exact, and the call is read (not turned back into text).

Found in review of the 0.2.12 parser picks: an unclosed quoted `<function=X> <parameter=Y>` in a value made param_end skip the
call's own close, so the whole call came back as text.  The fallback must never cut a value short: not when the quoted call is
closed and the call's own `</function>` is left out, and not across another parameter.

    python -m unittest serve.test_quoted_drift_values -v
"""
import json
import unittest

from serve.frontend import CALL_END, CALL_START, FUNC_END, FUNC_START, PARAM_END, PARAM_START, OutputParser

WRITE = {"name": "write", "parameters": {"properties": {"path": {"type": "string"}, "content": {"type": "string"}}}}


def call(name, params, func_end=True):
    s = f"{CALL_START}\n{FUNC_START}{name}>\n"
    for k, v in params.items():
        s += f"{PARAM_START}{k}>\n{v}\n{PARAM_END}\n"
    return s + (f"{FUNC_END}\n" if func_end else "") + CALL_END


CASES = [
    ("unclosed quoted opener, one line", {"path": "a.txt", "content": f"example: {FUNC_START}demo> {PARAM_START}x>1"}),
    ("unclosed quoted opener, lines", {"path": "a.txt", "content": f"example:\n{FUNC_START}demo>\n{PARAM_START}x>\n1"}),
]


class QuotedDriftValues(unittest.TestCase):
    def parse(self, text, tools, stream, size, recover=True):
        p = OutputParser(thinking=False, tools=tools, stream_tools=stream, recover=recover)
        events = []
        for i in range(0, len(text), size):
            events += p.feed(text[i:i + size])
        return events + p.finish()

    def test_unclosed_quoted_opener_keeps_the_call(self):
        for label, args in CASES:
            text = call("write", args)
            for recover in (False, True):
                for stream in (False, True):
                    for size in (1, 7, len(text)):
                        with self.subTest(label=label, recover=recover, stream=stream, size=size):
                            events = self.parse(text, [WRITE], stream, size, recover)
                            self.assertEqual([e.call.arguments for e in events if e.kind == "tool_call"], [args])
                            if stream:
                                self.assertEqual(json.loads("".join(e.text for e in events if e.kind == "tool_args")), args)

    def test_closed_quoted_call_with_the_own_function_end_left_out_is_not_cut(self):
        args = {"path": "a.txt", "content": f"{FUNC_START}demo>{PARAM_START}x>1{PARAM_END}{FUNC_END}\nTAIL"}
        text = call("write", args, func_end=False)
        for stream in (False, True):
            for size in (1, 7, len(text)):
                with self.subTest(stream=stream, size=size):
                    calls = [e.call.arguments for e in self.parse(text, [WRITE], stream, size) if e.kind == "tool_call"]
                    self.assertEqual(calls, [args])

    def test_a_turn_cut_by_max_tokens_never_guesses_the_end(self):
        # cut inside a value that quotes a whole call: the quoted close must not become the call's own (Codex review)
        text = (f"{CALL_START}\n{FUNC_START}write>\n{PARAM_START}path>\na.txt\n{PARAM_END}\n{PARAM_START}content>\n"
                f"{CALL_START}{FUNC_START}demo>{PARAM_START}x>1{PARAM_END}{FUNC_END}{CALL_END}")
        for reason in ("length", "cancel"):
            for stream in (False, True):
                for size in (1, 7, len(text)):
                    with self.subTest(reason=reason, stream=stream, size=size):
                        p = OutputParser(thinking=False, tools=[WRITE], stream_tools=stream, recover=True)
                        events = []
                        for i in range(0, len(text), size):
                            events += p.feed(text[i:i + size])
                        events += p.finish(reason)
                        self.assertEqual([e for e in events if e.kind == "tool_call"], [])
        # the same unclosed-quote call that ends naturally is read
        args = CASES[0][1]
        events = self.parse(call("write", args), [WRITE], False, 7)
        self.assertEqual([e.call.arguments for e in events if e.kind == "tool_call"], [args])

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
