import random
import unittest
from array import array

from serve import auto_prefix
from serve.auto_prefix import AutoPrefix, lcp

T = 248045  # <|im_start|>


def doc(n, seed):
    r = random.Random(seed)
    return [r.randrange(1000, 200000) for _ in range(n)]


def user_prompt(head, question, system=None):
    """[<|im_start|>system ...] <|im_start|>user HEAD QUESTION <|im_start|>assistant"""
    out = []
    if system:
        out += [T, 9125] + system
    return out + [T, 882] + head + question + [T, 77091]


class Lcp(unittest.TestCase):
    def test_against_a_loop(self):
        r = random.Random(1)
        for _ in range(300):
            n = r.randrange(0, 3000)
            a = [r.randrange(3) for _ in range(n)]
            b = a[:r.randrange(0, n + 1)] + [r.randrange(3) for _ in range(r.randrange(0, 50))]
            want = 0
            while want < min(len(a), len(b)) and a[want] == b[want]:
                want += 1
            self.assertEqual(lcp(array("q", a), array("q", b)), want)


class Pick(unittest.TestCase):
    def setUp(self):
        self.ap = AutoPrefix(T, env={"STRATA_AUTO_PREFIX": "1"})
        self.head = doc(40000, 7)

    def test_one_message_head_is_pinned_from_the_second_request(self):
        self.assertEqual(self.ap.pick(user_prompt(self.head, doc(400, 1))), 0)   # nothing to compare with yet
        p2 = self.ap.pick(user_prompt(self.head, doc(400, 2)))
        p3 = self.ap.pick(user_prompt(self.head, doc(400, 3)))
        self.assertGreater(p2, 40000 - auto_prefix.QUANTUM - auto_prefix.MARGIN)
        self.assertLessEqual(p2, 2 + 40000 - auto_prefix.MARGIN)
        self.assertEqual(p2 % auto_prefix.QUANTUM, 0)
        self.assertEqual(p3, p2)          # sent again: the engine drops it when an unrelated prompt comes between

    def test_head_in_a_system_message_is_left_to_the_root(self):
        sysmsg = doc(40000, 8)
        self.ap.pick(user_prompt(doc(300, 1), doc(50, 1), system=sysmsg))
        self.assertEqual(self.ap.pick(user_prompt(doc(300, 2), doc(50, 2), system=sysmsg)), 0)

    def test_next_turn_of_a_chat_is_left_to_the_turn_checkpoint(self):
        history = user_prompt(self.head, doc(50, 1))
        self.ap.pick(history)
        nxt = history + doc(300, 9) + [T, 882] + doc(200, 10) + [T, 77091]
        self.assertEqual(self.ap.pick(nxt), 0)

    def test_short_shared_start_is_ignored(self):
        h = doc(3000, 5)
        self.ap.pick(user_prompt(h, doc(3000, 1)))
        self.assertEqual(self.ap.pick(user_prompt(h, doc(3000, 2))), 0)

    def test_identical_prompt_is_left_to_the_turn_checkpoint(self):
        p = user_prompt(self.head, doc(10, 1))
        self.ap.pick(p)
        self.assertEqual(self.ap.pick(list(p)), 0)

    def test_pin_leaves_the_tail(self):
        # the shared start runs to the end of the user text: the pin still leaves tokens to read
        h = self.head + doc(10, 1)
        self.ap.pick([T, 882] + h + [5])
        pin = self.ap.pick([T, 882] + h + [6])
        self.assertGreater(pin, 0)
        self.assertLess(pin, len(h) + 3 - 1)

    def test_off_by_default_and_with_0(self):
        for env in ({}, {"STRATA_AUTO_PREFIX": "0"}, {"STRATA_AUTO_PREFIX": ""}, {"STRATA_AUTO_PREFIX": "false"},
                    {"STRATA_AUTO_PREFIX": "no"}, {"STRATA_AUTO_PREFIX": "true"}):
            ap = AutoPrefix(T, env=env)
            ap.pick(user_prompt(self.head, doc(400, 1)))
            self.assertEqual(ap.pick(user_prompt(self.head, doc(400, 2))), 0)

    def test_an_explicit_pin_is_sent_again_not_moved(self):
        first = user_prompt(self.head, doc(400, 1))
        self.ap.pick(first)
        self.ap.note_explicit(first, 40002)                                       # the client marked its document
        self.assertEqual(self.ap.pick(user_prompt(self.head, doc(400, 2))), 40002)

    def test_an_explicit_pin_lost_to_another_document_comes_back(self):
        first = user_prompt(self.head, doc(400, 1))
        self.ap.pick(first)
        self.ap.note_explicit(first, 40002)
        other = doc(40000, 14)
        self.ap.pick(user_prompt(other, doc(400, 2)))
        self.assertGreater(self.ap.pick(user_prompt(other, doc(400, 3))), 0)     # the engine now holds B's pin
        self.assertEqual(self.ap.pick(user_prompt(self.head, doc(400, 4))), 40002)

    def test_a_held_pin_is_not_moved_by_the_rounding(self):
        self.ap.pick(user_prompt(self.head, doc(400, 1)))
        p = self.ap.pick(user_prompt(self.head, doc(400, 2)))
        # questions that diverge right at p (user_prompt's 2 header tokens + head[:p - 2]): the rounding alone gives
        # p - QUANTUM, which would unpin the checkpoint at p
        self.ap.pick(user_prompt(self.head[:p - 2], doc(5000, 5)))
        self.assertEqual(self.ap.pick(user_prompt(self.head[:p - 2], doc(5000, 6))), p)

    def test_a_much_deeper_start_moves_the_pin(self):
        self.ap.pick(user_prompt(self.head, doc(400, 1)))
        p = self.ap.pick(user_prompt(self.head, doc(400, 2)))
        longer = self.head + doc(9000, 4)
        self.assertEqual(self.ap.pick(user_prompt(longer, doc(400, 3))), p)
        self.assertGreater(self.ap.pick(user_prompt(longer, doc(400, 5))), p + 4096)

    def test_an_explicit_pin_survives_a_later_one_that_never_ran(self):
        first = user_prompt(self.head, doc(400, 1))
        self.ap.pick(first)
        self.ap.note_explicit(first, 40002)
        other = user_prompt(doc(40000, 12), doc(400, 2))
        self.ap.note_explicit(other, 40002)                                       # (cancelled in the queue)
        self.assertEqual(self.ap.pick(user_prompt(self.head, doc(400, 3))), 40002)

    def test_the_pin_comes_back_after_an_unrelated_prompt(self):
        self.ap.pick(user_prompt(self.head, doc(400, 1)))
        p = self.ap.pick(user_prompt(self.head, doc(400, 2)))
        self.ap.pick(user_prompt(doc(150, 13), doc(10, 3)))                       # short and unrelated
        self.assertEqual(self.ap.pick(user_prompt(self.head, doc(400, 4))), p)

    def test_a_new_document_moves_the_pin(self):
        other = doc(40000, 11)
        self.ap.pick(user_prompt(self.head, doc(400, 1)))
        self.assertGreater(self.ap.pick(user_prompt(self.head, doc(400, 2))), 0)
        self.ap.pick(user_prompt(other, doc(400, 3)))
        self.assertGreater(self.ap.pick(user_prompt(other, doc(400, 4))), 0)

    def test_keep_and_min_from_env(self):
        ap = AutoPrefix(T, env={"STRATA_AUTO_PREFIX": "1", "STRATA_AUTO_PREFIX_KEEP": "1", "STRATA_AUTO_PREFIX_MIN": "bad"})
        self.assertEqual(ap.min_tokens, auto_prefix.DEFAULT_MIN)
        ap.pick(user_prompt(self.head, doc(400, 1)))
        ap.pick(user_prompt(doc(40000, 99), doc(400, 2)))                        # pushes the first one out
        self.assertEqual(ap.pick(user_prompt(self.head, doc(400, 3))), 0)


if __name__ == "__main__":
    unittest.main()
