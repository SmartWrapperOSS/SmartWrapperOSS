# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aditi Jain (SmartWrapperOSS)

"""
workflows/guess_game/game.py

The game ENGINE shared by both framework runners for the
`guess_the_person` eval: task loading, the game state machine, a
controlled-UI renderer, the card tools, the simulated player, and the
host's system prompt.

What this eval asks: can an agent reliably run a small, stateful
interaction while keeping the UI in sync with the game state? So the
engine records not just the final answer but the full sequence of
events and UI states that led to it.

Three layers, kept separate on purpose:

  1. TOOL CALLS   what the agent asked for       -> `tool_called` events
  2. GAME STATE   the state machine (below)      -> `guess_received`,
                                                    `feedback_rendered`,
                                                    `score_updated`, ...
  3. UI STATE     what each UI component actually -> `intro_rendered`,
                  shows right now                   `image_rendered`,
                                                    `results_rendered`

Tool-call correctness and UI correctness are scored independently: a
correct show_image call means nothing if the image card still shows the
previous image when the player guesses, and saying "you scored 3/4" in
chat doesn't count as rendering the results card.

THE FLOW
    INTRO --show_intro_card--> SHOW_IMAGE(r) --show_image(id_r)--> ASK(r)
    ASK(r) --host chat--> AWAIT_GUESS(r) --player guess--> JUDGE(r)
    JUDGE(r): reveal_answer(id_r) is allowed now (never before the guess),
              then a chat message starting with ✅ or ❌ is the feedback
              --> SHOW_IMAGE(r+1) | RESULTS
    RESULTS --show_results_card--> COMPLETE
    COMPLETE: the game is over. If the task has a post-game player
              message, the player sends it and the host must reply
              WITHOUT restarting the game; then DONE.

Any card or tool call that isn't legal in the current phase is REJECTED
(the agent gets an error and must recover) and recorded.

Answer matching follows the task's policy (case-insensitive, trimmed).
The expected identity is never in the image tool's result; the agent can
only get it from reveal_answer after the player has guessed.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import yaml

CORRECT_MARK = "✅"
WRONG_MARK = "❌"


# --- Task ---------------------------------------------------------------------

@dataclass
class Round:
    image_id: str
    expected_answer: str
    player_guess: str
    expected_correct: bool


@dataclass
class GuessGameTask:
    task_id: str
    instructions: str
    rounds: List[Round]
    opening_message: str
    post_game_message: Optional[str]
    expected_final_score: int

    @property
    def image_ids(self) -> List[str]:
        return [r.image_id for r in self.rounds]


def matches(guess: str, answer: str) -> bool:
    """The task's matching policy: case-insensitive after trimming
    (inner runs of whitespace collapsed)."""
    def norm(s):
        return " ".join(str(s).split()).lower()
    return norm(guess) == norm(answer)


def load_task(yaml_path: str) -> GuessGameTask:
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    rounds = []
    for i, r in enumerate(data["test_data"], start=1):
        rnd = Round(image_id=r["image_id"], expected_answer=r["expected_answer"],
                    player_guess=r["player_guess"],
                    expected_correct=bool(r["expected_correct"]))
        if matches(rnd.player_guess, rnd.expected_answer) != rnd.expected_correct:
            raise ValueError(f"{yaml_path}: round {i} expected_correct disagrees with the "
                             f"case-insensitive, trimmed matching policy")
        rounds.append(rnd)
    if not 1 <= len(rounds) <= 10:
        raise ValueError(f"{yaml_path}: a game needs between 1 and 10 rounds")
    expected = data.get("expected_outcome", {}).get("correct_guesses",
                                                    sum(r.expected_correct for r in rounds))
    if expected != sum(r.expected_correct for r in rounds):
        raise ValueError(f"{yaml_path}: expected_outcome.correct_guesses doesn't match test_data")
    return GuessGameTask(
        task_id=data["task_id"],
        instructions=data["instructions"].strip(),
        rounds=rounds,
        opening_message=data.get("opening_message", "Hi! I'd like to play."),
        post_game_message=data.get("post_game_message"),
        expected_final_score=expected,
    )


# --- State machine --------------------------------------------------------------

class Phase(str, Enum):
    INTRO = "intro"
    SHOW_IMAGE = "show_image"
    ASK = "ask"
    AWAIT_GUESS = "await_guess"
    JUDGE = "judge"
    RESULTS = "results"
    AWAIT_POST_GAME = "await_post_game"   # game complete; player about to send a message
    POST_GAME = "post_game"               # game complete; host must reply without restarting
    DONE = "done"


NEXT_STEP = {
    Phase.INTRO: "show the introduction card with show_intro_card",
    Phase.SHOW_IMAGE: "show the next image with show_image",
    Phase.ASK: "ask the player in chat to guess who is in the image",
    Phase.AWAIT_GUESS: "wait for the player's guess",
    Phase.JUDGE: (f"check the answer with reveal_answer, then give feedback in chat "
                  f"starting with {CORRECT_MARK} or {WRONG_MARK}"),
    Phase.RESULTS: "show the final score with show_results_card",
    Phase.AWAIT_POST_GAME: "nothing; the game is complete",
    Phase.POST_GAME: "reply in chat; the game is complete and must not restart",
    Phase.DONE: "nothing; the game is complete",
}

GAME_OVER_PHASES = (Phase.AWAIT_POST_GAME, Phase.POST_GAME, Phase.DONE)

# What the player says when no guess is due, based on what's on screen.
PLAYER_FILLER = {
    Phase.INTRO: "Sounds fun! How does it work?",
    Phase.SHOW_IMAGE: "I'm ready, but I don't see a picture on screen yet.",
    Phase.ASK: "I can see the picture. What should I do?",
    Phase.JUDGE: "So, was I right?",
    Phase.RESULTS: "So what's my final score?",
}


class GameMachine:
    """
    One game. Both runners send every tool call, host chat message, and
    player turn through this object. It enforces the flow, drives the UI
    renderer, and appends to an ordered event trace that the evaluator
    scores. The runners never score themselves.
    """

    def __init__(self, task: GuessGameTask):
        self.task = task
        self.phase = Phase.INTRO
        self.round = 1
        self.score = 0
        self.revealed: set = set()       # rounds whose answer has been revealed
        self.ui: Dict[str, Optional[dict]] = {"intro_card": None, "image_card": None,
                                              "results_card": None}
        self.events: List[Dict[str, Any]] = []
        self._emit("game_started", total_rounds=self.total)

    @property
    def total(self) -> int:
        return len(self.task.rounds)

    @property
    def done(self) -> bool:
        return self.phase == Phase.DONE

    def _emit(self, type_: str, **fields) -> None:
        self.events.append({"seq": len(self.events), "type": type_,
                            "phase": self.phase.value, **fields})

    def _ui_snapshot(self) -> dict:
        img = self.ui["image_card"]
        return {"image_card_image_id": img["image_id"] if img else None,
                "results_card_visible": self.ui["results_card"] is not None}

    # --- tool entry point (logs the call, then dispatches) ---------------------

    def call(self, tool: str, args: dict) -> str:
        handler = {"show_intro_card": self._show_intro_card, "show_image": self._show_image,
                   "reveal_answer": self._reveal_answer,
                   "show_results_card": self._show_results_card}.get(tool)
        if handler is None:
            self._emit("tool_called", tool=tool, args=args, accepted=False,
                       error=f"unknown tool '{tool}'")
            return f"ERROR: unknown tool '{tool}'."
        args = args if isinstance(args, dict) else {}
        try:
            error = handler(check_only=True, **args)
        except TypeError:
            error = f"invalid arguments for {tool}: {sorted(args)}"
        if error:
            self._emit("tool_called", tool=tool, args=args, accepted=False, error=error)
            if self.phase == Phase.JUDGE:
                return (f"ERROR: {error} The player has not received feedback for round "
                        f"{self.round} yet: no chat message starting with {CORRECT_MARK} or "
                        f"{WRONG_MARK} has reached them. Send that feedback in chat first.")
            if self.phase in GAME_OVER_PHASES:
                return f"ERROR: {error} The game is complete; do not start a new round."
            return f"ERROR: {error} Right now you should: {NEXT_STEP[self.phase]}."
        self._emit("tool_called", tool=tool, args=args, accepted=True)
        return handler(check_only=False, **args)

    # --- tools: each validates (check_only=True) or performs (False) -------------

    def _show_intro_card(self, check_only: bool, title: str = "", rules=None,
                         total_rounds=None) -> Optional[str]:
        if check_only:
            if self.phase != Phase.INTRO:
                return "The introduction card can only be shown at the start."
            if not str(title).strip() or not rules or not all(str(r).strip() for r in rules):
                return "The introduction card needs a title and at least one rule."
            return None
        props = {"title": title, "rules": list(rules), "total_rounds": total_rounds}
        self.ui["intro_card"] = props
        self._emit("intro_rendered", component="intro_card", props=props)
        self.phase = Phase.SHOW_IMAGE
        return "Introduction card displayed."

    def _show_image(self, check_only: bool, image_id: str = "") -> Optional[str]:
        expected = self.task.rounds[self.round - 1].image_id if self.round <= self.total else None
        if check_only:
            if self.phase != Phase.SHOW_IMAGE:
                return "An image can't be shown right now."
            if str(image_id).strip() != expected:
                return f"Wrong image: round {self.round} uses the next image in the list."
            return None
        props = {"image_id": image_id, "image_ref": f"fixtures/{image_id}.png",
                 "round_label": f"Round {self.round} of {self.total}"}
        self.ui["image_card"] = props
        self._emit("image_rendered", component="image_card", round=self.round, **props)
        self.phase = Phase.ASK
        return (f"Image {image_id} displayed in the image card "
                f"(round {self.round} of {self.total}).")

    def _reveal_answer(self, check_only: bool, image_id: str = "") -> Optional[str]:
        rnd = self.task.rounds[self.round - 1] if self.round <= self.total else None
        if check_only:
            if self.phase != Phase.JUDGE:
                return "The answer can only be revealed after the player has guessed."
            if str(image_id).strip() != rnd.image_id:
                return "That isn't the image for the current round."
            if self.round in self.revealed:
                return "The answer for this round was already revealed."
            return None
        self.revealed.add(self.round)
        return f"The person in {image_id} is {rnd.expected_answer}."

    def _show_results_card(self, check_only: bool, score=None, total=None) -> Optional[str]:
        if check_only:
            if self.phase != Phase.RESULTS:
                return "The results card can only be shown after every round is done."
            return None
        props = {"score": score, "total": total}
        self.ui["results_card"] = props
        self.ui["image_card"] = None    # no active round once results are up
        self._emit("results_rendered", component="results_card", props=props)
        self._emit("game_completed", final_score=self.score, active_round=None)
        self.phase = Phase.AWAIT_POST_GAME if self.task.post_game_message else Phase.DONE
        return "Results card displayed. The game is complete."

    # --- chat ------------------------------------------------------------------

    def host_said(self, text: str, with_tool_call: bool = False) -> None:
        """A chat message from the host. `with_tool_call` marks text the model
        wrote in the same response as a tool call; runners deliver it BEFORE
        the tool runs, the way a chat UI shows it."""
        text = text or ""
        self._emit("chat_message", sender="host", text=text, with_tool_call=with_tool_call)
        if self.phase == Phase.ASK and text.strip():
            self.phase = Phase.AWAIT_GUESS
        elif self.phase == Phase.JUDGE:
            s = text.lstrip()
            verdict = ("correct" if s.startswith(CORRECT_MARK)
                       else "incorrect" if s.startswith(WRONG_MARK) else None)
            if verdict:
                self._emit("feedback_rendered", component="chat", round=self.round,
                           verdict=verdict, text=text,
                           answer_was_revealed=self.round in self.revealed)
                before = self.score
                self.score += verdict == "correct"
                self._emit("score_updated", round=self.round, before=before, after=self.score)
                if self.round < self.total:
                    self.round += 1
                    self.phase = Phase.SHOW_IMAGE
                else:
                    self.round += 1
                    self.phase = Phase.RESULTS
        elif self.phase == Phase.POST_GAME and text.strip():
            self.phase = Phase.DONE

    def player_reply(self) -> str:
        """The simulated player's turn: their scripted guess or post-game
        message when one is due. Otherwise the player reacts to what is (or
        isn't) on screen, like a real person would, instead of a bare
        "Okay!" that a host could mistake for a missing guess."""
        if self.phase == Phase.AWAIT_GUESS:
            guess = self.task.rounds[self.round - 1].player_guess
            self._emit("guess_received", round=self.round, guess=guess, ui=self._ui_snapshot())
            self.phase = Phase.JUDGE
            return guess
        if self.phase == Phase.AWAIT_POST_GAME:
            msg = self.task.post_game_message
            self._emit("post_game_message", text=msg)
            self.phase = Phase.POST_GAME
            return msg
        filler = PLAYER_FILLER.get(self.phase, "Okay!")
        self._emit("player_filler", text=filler)
        return filler


# --- Card tools as plain typed functions ----------------------------------------
#
# Both frameworks build their native tool schemas from these SAME
# signatures and docstrings, so neither gets a better tool description.

def make_card_tools(machine: GameMachine) -> list:
    def show_intro_card(title: str, rules: List[str], total_rounds: int) -> str:
        """Show the INTRODUCTION card that explains the game. Call once, at the start.

        Args:
            title: Short title for the card.
            rules: The rules, one short sentence per item.
            total_rounds: How many rounds this game has.
        """
        return machine.call("show_intro_card",
                            {"title": title, "rules": rules, "total_rounds": total_rounds})

    def show_image(image_id: str) -> str:
        """Show an image in the IMAGE card for the current round.

        Args:
            image_id: The id of the image to show.
        """
        return machine.call("show_image", {"image_id": image_id})

    def reveal_answer(image_id: str) -> str:
        """Get the correct answer for the current image. Only works after the
        player has guessed.

        Args:
            image_id: The id of the current image.
        """
        return machine.call("reveal_answer", {"image_id": image_id})

    def show_results_card(score: int, total: int) -> str:
        """Show the final RESULTS card. Call once, after the last round's feedback.

        Args:
            score: How many guesses were correct.
            total: How many rounds were played.
        """
        return machine.call("show_results_card", {"score": score, "total": total})

    return [show_intro_card, show_image, reveal_answer, show_results_card]


# --- Host prompt (identical for both frameworks) --------------------------------

def host_system_prompt(task: GuessGameTask) -> str:
    return f"""You are the host of a game called Guess the Person, with {len(task.rounds)} rounds.

The player sees only the UI cards you show with your tools and the chat
messages you write. Use the introduction card for the rules, the image card
for each picture, and the results card for the final score. Talk to the
player, ask for guesses, and give feedback in chat.

INSTRUCTIONS:
{task.instructions}

Images, in order: {", ".join(task.image_ids)}

To check a guess, call reveal_answer for the current image after the player
has guessed. Start every feedback message with {CORRECT_MARK} if the guess was
right or {WRONG_MARK} if it was wrong."""
