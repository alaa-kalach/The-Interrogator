# The Interrogator

A persona-driven LLM chatbot for EECE 503P/798S C2. Every message becomes a **case**: the app extracts evidence, forms hypotheses, asks up to five targeted follow-up questions, and closes with a structured conclusion, narrated by a hardboiled detective persona.

Built on Groq's free API tier, with a Gradio interface, few-shot prompting, prompt chaining, structured case state, automatic context management, and a Reasoning Mode stretch goal.

---

## 1. Setup and Running the Notebook

1. **Get a free Groq API key** (Groq is the service that runs the language model this app talks to). Go to [console.groq.com](https://console.groq.com), sign in, open the "API Keys" page, and create a new key. Copy it somewhere safe because you'll paste it once, in step 4.
2. **Open the notebook in Google Colab.** Go to [colab.research.google.com](https://colab.research.google.com), choose Upload, and select the notebook file.
3. **Run the cells in order, top to bottom.** Use the menu **Runtime → Run all**, which runs every cell automatically in the correct order.
4. **When prompted, paste your Groq API key.** The first code cell will ask for it (input is hidden as you type/paste, like a password field). Or, you can simply add the key in the "Secrets" section of colab where you do not have to paste it every single time. This secret is called "GROQ_API_KEY".
5. Keep running cells top to bottom. Each section prints its own output as it runs. The intermediate sections are demonstrating individual techniques on their own, before everything comes together in the final app.
6. **When the last cell ("Dynamic UI") runs, it will *not* show an embedded preview inside Colab.** The cell prints a line like:

   ```
   * Running on public URL: https://xxxxxxxxxxxx.gradio.live
   ```

   **Click that link.** It opens the actual app in a new, full-size browser tab, and is what testing in this document are based on.
7. The share link is temporary (about a week). The cell keeps running so Colab can keep streaming live logs into the output as you use the app — to stop the server.

---

## 2. Model Selection

**Why Groq.** Groq offers a free API tier that serves open-source models (Qwen, Kimi, gpt-oss) through custom LPU hardware, giving fast inference without needing to host anything locally. It has a meaningful advantage for iterating quickly in Colab, where local GPU hosting of a 20B+ parameter model would consume a lot. Its API is OpenAI-compatible, so the same `groq` Python SDK patterns (`client.chat.completions.create(...)`) apply throughout.

Rather than hardcoding one model, `Environment Setup` queries `client.models.list()` and picks the best available option from a preference list, since Groq's free-tier lineup has changed several times while testing my app, models have been added, deprecated, and renamed mid-development:

```python
PREFERRED_MODELS = [
    "qwen/qwen3.8-27b",
    "qwen/qwen3.6-27b",
    "qwen/qwen3-32b",
    "moonshotai/kimi-k2-instruct-0905",
    "moonshotai/kimi-k2-instruct",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]

def pick_model(exclude=None):
    """Picks the best available model from PREFERRED_MODELS, skipping anything in
    `exclude` (used to skip a model that just hit its daily quota). Falls back to any
    other available model if none of the preferred ones are left."""
    exclude = exclude or set()
    candidates = [m for m in PREFERRED_MODELS if m in available_ids and m not in exclude]
    if not candidates:
        candidates = [m for m in available_ids if m not in exclude]
    return candidates[0] if candidates else None
```

Plain instruct models (Qwen, Kimi) are preferred over gpt-oss, because gpt-oss's "Harmony" response format can raise a spurious tool-call error once a system prompt is introduced (this only affects the single-prompt path in the chain never sends a system message, so gpt-oss is safe there). Different model families also need different `reasoning_effort` handling:

```python
def reasoning_effort_for(model_id):
    """Qwen models are hybrid 'thinking' models; reasoning_effort='none' suppresses
    visible <think> tags. gpt-oss models use a different low/medium/high scale.
    Other models get no reasoning_effort parameter at all."""
    if model_id and model_id.startswith("qwen/"):
        return "none"
    if model_id and "gpt-oss" in model_id:
        return "low"
    return None
```

**Context window.** The app's top-preference model, `qwen/qwen3.8-27b` on Groq, has a **131,042-token (~131K) context window**. This isn't hardcoded or assumed, though — `get_context_window(MODEL_ID)` reads it live from Groq's own API at runtime (every model object Groq returns reports its own `context_window` field), so the figure stays accurate even if a fallback model ends up selected instead. The Context Handling section prints the live value for whichever model is actually active in a given session.

**Automatic model fallback.** Free-tier daily quotas are easy to exhaust during heavy testing, where this happened repeatedly during development, including in the middle of live testing sessions. `create_with_retry` (the single choke-point every API call in the app routes through) distinguishes two kinds of failure by parsing Groq's own error message:

- **Short-lived rate limit** (a per-minute cap, such as *"Please try again in 8.16s"*) → waits out the stated delay and retries the same model.
- **Long-lived limit** (a daily quota, such as *"Rate limit reached ... on tokens per day (TPD) ... Please try again in 4m4s"*) → calls `switch_to_next_model()`, which picks the next candidate from `PREFERRED_MODELS` via the same `pick_model` function above, updates the global `MODEL_ID` and its matching `REASONING_EFFORT`, and retries immediately, instead of the app just breaking mid-conversation. If every available model is exhausted, only then does it raise.

This was tested against the exact real error message Groq returns for a daily-quota exhaustion, with a scripted fallback from `qwen/qwen3.8-27b` to `openai/gpt-oss-120b`, confirming the switch and retry complete correctly, including updating `REASONING_EFFORT` to match the new model family.

**Troubleshooting.** If the first cell rejects your API key, double-check it was copied without extra whitespace and that it hasn't been revoked on the Groq console. If the app itself starts returning the in-character *"the precinct's lines are jammed"* message repeatedly, that's the graceful-failure path in `stateful_investigate` catching a real API error after retries and fallback were exhausted, check the printed `[stage error]` line above it in the cell output for the underlying cause.

## 3. Architecture

The notebook builds the app incrementally, one section at a time, each layering on the last rather than being independent demos. The final Dynamic UI cell runs with the fully-evolved logic (question-history tracking, reasoning mode, evidence-list trimming, etc.) even though the underlying helper functions were first introduced several cells earlier, in a simpler form. Running the notebook top to bottom reproduces this evolution in order.

| Section | What it adds |

| Baseline Chat | Plain chat, no persona, the "before" comparison point that makes the persona's effect visible |
| Persona and Few-Shot Prompting | The detective persona via a system prompt, plus 3 worked examples embedded in it |
| Prompt Chaining | Splits one conversational turn into 5 separate, narrowly-scoped API calls instead of one big prompt |
| Structured Case State | Replaces resending the whole raw conversation with a compact, deduplicated evidence list that persists in code, not in the prompt |
| Context Handling | Recent-history retention + summarization for the raw-history path, a parallel, independent safety net for the evidence list |
| Reasoning Mode | A toggle that elicits and displays chain-of-thought before the final answer |
| Dynamic UI | The live Gradio app, combines every technique above into one interface |

### 3.1 The five-stage pipeline

Every turn in the chain / case-state / Dynamic UI path runs through five stages, each a **separate API call** with its own narrow instruction (rather than one prompt asking a model to do everything at once):

1. **Evidence extraction** (`extract_new_evidence`) — pulls only the new, concrete facts from *this* message, given the last question asked as context and the facts already known (so it doesn't duplicate or lose track)
2. **Hypothesis generation** (`generate_hypotheses`) — 2-4 plausible explanations given the evidence accumulated so far
3. **Evidence gap** (`find_missing`) — what's missing that would actually help decide between the hypotheses
4. **Decision** (`decide_close`) — a single word, `CLOSE` or `CONTINUE`, based on whether the evidence is sufficient yet
5. **Next question or conclusion** (`generate_question` / `generate_conclusion`) — depending on stage 4's decision

A full turn's console output looks like this (from a real session):

```
--- Stage 1: New-Evidence Extraction ---
['You have a math exam tomorrow.']

--- Stage 2: Hypothesis Generation ---
- Insufficient time to review all necessary material is causing the fear.
- A lack of confidence in specific math topics is driving the worry.

--- Stage 3: Evidence Gap ---
- How much time you have actually spent reviewing the material so far.
- Which specific math topics you feel least confident about.

--- Stage 4: Decision --- CONTINUE

--- Stage 5a: Next Question ---
Which specific math topics do you feel least confident about?
```

This decomposition is also what makes the app debuggable: because each stage's raw output prints to the console, a wrong final answer can be traced back to exactly which stage introduced the problem, rather than having to guess inside one opaque completion.

### 3.2 Case state

Rather than re-deriving everything from the chat transcript on every turn, the app keeps one persistent dictionary per case that accumulates as the investigation proceeds:

```python
{
  "case_id": int, "status": "OPEN"/"CLOSED", "stage_label": str,
  "evidence": [...],            
  "hypotheses": "", "missing": "", "recommendation": "", "reasoning_trace": "",
  "questions_asked": 0, "last_question": "",
  "asked_questions": [...],     # full history, so a later stage can't repeat an already-asked question
  "stall_count": 0,             # consecutive turns with no new evidence extracted
}
```

`evidence`, `asked_questions`, and `stall_count` exist specifically because real testing surfaced bugs that a stateless design couldn't have caught or fixed: a case's evidence list needs to persist exactly as extracted (not re-summarized each turn, which caused duplicate/drifting facts in earlier versions). question history exists because the app was once observed asking the identical question three turns in a row. stall count exists because that same bug wasted the entire question budget without ever recognizing the investigation had stalled. 

`MAX_QUESTIONS = 5` caps investigation length, but this is a **ceiling, not a target** — `decide_close` evaluates the evidence every single turn and closes as soon as it judges the case sufficiently resolved. Real test cases closed anywhere from 3 to 5 questions, and the cap exists to guarantee the conversation can't run away indefinitely if the model keeps finding reasons to continue.

### 3.3 Web UI and UX

Built with `gr.Chatbot`, which natively gives a scrollable message history and a clear visual separation between the user's messages (right-aligned bubbles) and the detective's replies (left-aligned). Beside the chat sits a separate "Case File" panel (Evidence, Hypotheses/Recommendation, Missing/Reasoning Trace), themed as an interrogator, dark violet gradient background, a typewriter font, an animated detective figure that changes pose between "investigating" and "case closed," and a rotated CONFIDENTIAL stamp. The panel's internal sections are laid out with `gr.Row`/`gr.Column` and a `min_width` on each, so on a narrow screen Gradio's own layout engine wraps them vertically instead of squeezing them unreadably thin. This responsive behavior was verified across different desktop browser-window widths during development.

## 4. Prompting Techniques

**Why these two.** Few-shot prompting was chosen because a persona's *voice* is easy to specify in words ("speak like a detective") but its *judgment quality*. What counts as a sharp hypothesis versus a vague one, when there's enough evidence to close a case, is much easier to demonstrate than to describe, which is exactly what worked examples are for. Prompt chaining was chosen because the task itself decomposes naturally into distinct judgment calls (what's the evidence, what follows from it, what's missing, is it time to decide, what's the decision). Then collapsing all of that into one prompt asking for a paragraph back leaves no way to verify or fix any one part of the reasoning without touching all of it. Both choices were validated by what actually went wrong during development: several real bugs were fixable *because* the pipeline was chained into inspectable stages, not because the model got smarter.

### 4.1 Few-shot prompting

Three worked examples are embedded in the persona's system prompt (vague opener → ask, don't guess; a report with a real signal → sharp hypotheses; enough evidence in one message → close immediately). Inside the chain, three of the five stages carry their own smaller worked examples, hypothesis generation, the next-question stage, and the conclusion stage, controlled by a single `USE_FEW_SHOT` flag (default `True`), which also makes a direct ablation comparison possible. The other two stages (extraction, decision) are narrow, mechanical judgment calls where an example wouldn't add much, so they stay zero-shot by design.

**Measured effect.** A controlled ablation (same inputs, examples on vs. off) found the written rules alone already produce correct structure, single-question compliance, and correct pronoun handling, both conditions scored identically there. The real, measurable difference showed up specifically in conclusion quality: *"goes beyond restating the evidence"* scored 6/6 with examples vs. 4/6 without, across two separate real-model runs. Example: without examples, a museum-theft conclusion simply restated the three given facts back as a sentence; with examples, it drew an actual inference ("removed by someone who bypassed the alarm") and gave a grounded, hedged assessment of what remained unconfirmed.

A second, more subtle finding from the same ablation: on genuinely thin evidence (a case opened with just "You have a mother"), the version *without* examples sometimes produced the single most honest possible response — a one-line hypothesis admitting there wasn't enough to go on — while the version *with* examples reliably produced 2-4 hypotheses even then, formatted correctly but slightly more speculative. Few-shot examples improved format reliability, but on very thin evidence the "unformatted" honesty is arguably a feature the automated checks weren't designed to reward.

**How the persona is enforced (system prompt excerpt):**

```
You are THE INTERROGATOR, a sharp, methodical detective persona.

You do not answer questions directly. Instead, you treat every user message as a CASE.
Your job is to investigate the case, not to immediately solve it.

Behavior rules:
1. Every single response, without exception, must use these exact section
   headers in this order: for a new or ongoing case use CASE OPENED (first
   turn only) or CASE UPDATE (later turns), then EVIDENCE:, HYPOTHESES:,
   MISSING:, and end with your one question. ...
```

**How a few-shot example is embedded (one of three worked examples in the same system prompt):**

```
EXAMPLE 1: A vague opener with almost no evidence yet — ask, don't guess ---
User: My cat has been acting weird lately.
Detective:
CASE OPENED

EVIDENCE:
- Your cat's behavior has changed, but no specifics are given yet.

HYPOTHESES:
- H1: A physical health issue (pain, illness).
- H2: A stress or environmental change (new pet, moved furniture, schedule change).
- H3: A normal age-related behavior shift.
...
```

**How a chain stage is instructed** (`generate_hypotheses`, one of the five pipeline stages):

```
"You are the HYPOTHESIS-GENERATION stage of a detective's case-analysis pipeline. "
"Given ONLY this evidence, list 2-4 plausible hypotheses about the practical "
"SITUATION or PROBLEM described, as specific as the evidence allows. Stay within
what the evidence supports: never invent circumstances, people, or events it does
not mention, and when the evidence is thin keep the hypotheses general rather than
inventing specifics. ..."
```

Each of the five pipeline stages has its own narrowly-scoped instruction like this — a single-purpose prompt per stage, rather than one prompt trying to do everything at once, which is the core idea behind prompt chaining below.

### 4.2 Prompt chaining

Compared directly against the single-prompt version on an identical 3-turn conversation, across three separate real-model runs:

| Run | Single-prompt tokens | Chain tokens | Chain API calls |
|---|---|---|---|
| 1 | 4,993 | 5,297 | 15 vs. 3 |
| 2 | 5,049 | 4,987 | 15 vs. 3 |
| 3 | 5,281 | 5,232 | 15 vs. 3 |

No consistent token winner across three runs, the single prompt's large system prompt gets resent every turn, roughly offsetting the chain's extra round trips. The real cost of chaining is the 5× call count (latency, rate limits), not tokens.

**What chaining actually bought during development** was the ability to isolate and fix problems a single opaque completion would have hidden. A representative sample of real bugs the staged structure made fixable:

- **Evidence bleeding across case boundaries.** A closed case's evidence was initially still influencing the *next* case's extraction, because the extraction stage's conversation-history window wasn't reset at a case boundary. Because extraction is its own stage with its own printed output, this was visible directly in the console log rather than buried inside a final answer that merely looked slightly off.
- **A repeated question wasting the entire question budget.** In one real session, the app asked the identical question three turns in a row because the question-generation stage had no memory of what it had already asked. Fixed by giving that stage explicit question history, a fix only possible because "generate the next question" is its own addressable stage, not a sub-clause inside a longer prompt.
- **An empty question reaching the user.** On a fallback model, the question stage occasionally returned nothing (likely hidden reasoning tokens consuming a too-tight budget). Because this stage's output is checked in code before being shown, a safety net could be added, falling back to the first "Missing" bullet as a phrased question, rather than the user seeing a broken blank line.

None of these would have been diagnosable, let alone fixable, if the whole turn were one prompt producing one block of text.

## 5. Context Handling

Two different, deliberately different policies, because the two architectures have genuinely different problems.

### 5.1 Raw-history path: recency retention + summarization

**`interrogator_chat`** (the single-prompt version) resends the full raw transcript every turn, so it can genuinely hit a context limit on a long conversation. It gets real recency-based retention:

```python
def estimate_tokens(text):
    """Rough character-based estimate (~4 characters per token), used only to decide
    when to trim, not for billing accuracy. Good enough for a retention policy: being
    off by 20% still trims at roughly the right point."""
    return max(1, len(text) // 4)


CONTEXT_BUDGET_TOKENS = min(6000, (CONTEXT_WINDOW or 8192) // 2)
RECENT_TURNS_TO_KEEP = 6  
```

The character-based estimate (rather than a real tokenizer) was a deliberate simplicity trade-off: it needs no extra dependency, runs instantly, and only has to be accurate enough to decide *when* to trim, not to bill correctly. The budget is deliberately conservative (half the model's real window, capped at 6,000) so trimming engages well before the model would actually reject the request.

Once the budget is crossed, the system prompt and the most recent 6 turns stay verbatim; everything older is collapsed into one summary turn, generated by the model itself via `summarize_old_turns`. Tested against a real model on a synthetic 140-turn conversation:

```
Before: 140 turns, ~7021 estimated tokens (budget: 6000)
After:  7 turns, ~1707 estimated tokens, trimmed=True

Summary: "You have experienced a headache localized to your forehead and behind
your eyes since this morning, which worsens with exposure to bright light.
Contributing factors include approximately five hours of continuous laptop use,
lack of water intake since breakfast, a skipped lunch, and only five hours of
sleep the previous night. You took a short walk around noon and report no fever,
noting that similar symptoms have occurred a couple of times during past exam weeks."
```

All nine original facts present, nothing invented — a ~76% token reduction with no information loss.

### 5.2 Chain / case-state / Dynamic UI path: structured state instead of a transcript

The chain never resends raw history at all. `extract_new_evidence` only ever sees the current message, the last question asked, and the running evidence list, already a stronger form of context management than truncation, since it's compact and deduplicated by construction rather than being a shrunk-down copy of a much larger transcript. It still gets a matching safety net, `manage_evidence`: the 10 most recent facts stay verbatim; once the list's estimated token count crosses 500, older facts collapse into one combined sentence via `summarize_evidence_facts`, using the same underlying idea a summarization but applied to a list of facts instead of a list of turns.

Under the default `MAX_QUESTIONS=5`, a real case rarely produces enough facts to actually cross that 500-token threshold, this was confirmed directly: a real live test with a full 5-question case reached only about 60-90 estimated tokens of evidence, nowhere near the limit. This is genuine, tested behavior (verified separately by seeding a 30-item evidence list and confirming it correctly collapses to 10 recent items plus one summary, preserving the newest facts), not something the default settings are expected to exercise in ordinary use, it exists as a safety net for longer cases (if `MAX_QUESTIONS` were raised later), not a mechanism the demo is built to showcase by default.

## 6. Reasoning Mode (Stretch Goal)

A toggle (on by default, next to Send / New Case, styled with a teal accent matching the trace panel it controls) enables chain-of-thought elicitation at the conclusion stage. When on, `generate_conclusion_with_reasoning` replaces the plain `generate_conclusion` for the final stage only, every other stage, and all of the grounding rules, stay identical between the two modes, so the only variable is whether the model shows its work first.

**The chain-of-thought instruction** (word-capped deliberately, after an early version let the model's reasoning run to an essay-length wall of text that got cut off mid-thought by the token budget):

```
"You are the CASE-CONCLUSION stage of a detective's case-analysis pipeline, "
"running in REASONING MODE. Think through this step by step BEFORE answering. "
"Be concise throughout — this is a brief investigation trace, not an essay. ..."
"First, under the header INVESTIGATION SUMMARY, write EXACTLY these four lines. "
"Each line has a HARD LIMIT of 20 words — count as you write and cut anything over. ..."
"Evidence considered: a single short clause naming only the most relevant facts.\n"
"Hypotheses considered: a single short clause naming only the most relevant hypotheses.\n"
"Key missing or decisive evidence: a single short clause on the one fact that settles it.\n"
"Decision basis: AT MOST 4 numbered steps, each under 15 words, one plain clause each ..."
"Then, under the header FINAL CASE CONCLUSION, write these EXACT section headers: "
"PRIMARY CONCLUSION, EVIDENCE, HYPOTHESES RULED OUT, RECOMMENDATION. ..."
```

**Visual separation** (the assignment's specific requirement) is implemented on two surfaces at once: in the chat, the investigation summary renders as a markdown blockquote immediately before a `**FINAL CASE CONCLUSION**` heading; in the Case File panel, it gets its own teal-accented "Reasoning Trace" column laid out beside Evidence and Recommendation, visually distinct in color from the rest of the panel's violet theme. The `Decision basis`, usually the longest part, is wrapped in a native HTML `<details>`/`<summary>` disclosure toggle, collapsed by default, so the panel stays readable at a glance without an expandable wall of text pushing everything else off-screen.

**A real example**, from live testing (case: substituting baking powder with baking soda and vinegar):

```
INVESTIGATION SUMMARY
Evidence considered: You need to replace one tablespoon of baking powder using
available baking soda and acids but lack the specific substitution ratio.
Hypotheses considered: The core issue is determining the precise volumes of
baking soda and acid required to replicate the leavening of baking powder.
Key missing or decisive evidence: The specific volume measurements for the
baking soda and acid components constitute the decisive missing fact.
Decision basis:
1. You explicitly stated that providing the specific volume of vinegar is
   the detective's job.
2. You also requested the specific quantity of baking soda.
3. You confirmed you do not know the required ratio or quantities.
4. Therefore, the case requires the detective to provide these exact
   measurements.
```

**Demonstration prompts.** The notebook also includes two classic reasoning-sensitive problems with one objectively correct answer each, a cleaner test of correctness than a real, judgment-heavy detective case, and closer to what the assignment's own example ("a multi-step word problem or a logic puzzle") describes, run through both the plain and reasoning-mode conclusion functions for direct comparison:

1. **Bat-and-ball**: "A bat and a ball cost \$1.10 total. The bat costs \$1.00 more than the ball. How much does the ball cost?" - correct answer \$0.05; the common wrong intuitive answer is \$0.10.
2. **Three mislabeled boxes**: three boxes labeled Apples / Oranges / Apples+Oranges are all mislabeled; drawing an apple from the box labeled "Apples+Oranges" reveals the correct labeling of all three.


**A genuine trade-off, not just a feature.** On a real medical-style case (new severe headache with a preceding visual aura), Reasoning Mode's step-by-step deduction was medically sound (aura duration + no neurological deficits genuinely does favor migraine over stroke), but it also produced a more *confident* diagnostic-sounding conclusion ("not indicative of a stroke") than the non-reasoning version, which hedged toward "seek immediate medical attention." For a symptom-checker-style chatbot, the hedged answer is arguably the safer one to give, even though the reasoning behind the confident one was sound. Reasoning Mode can make the persona sound more authoritative, that's a real cost on judgment-heavy domains, not an unambiguous upgrade.

## 7. Example Conversations

All from real, unscripted test sessions (not cherry-picked for success for what didn't work, and the bugs for what was found and fixed along the way).

**Fractions — a genuinely non-obvious diagnosis.** "I have a math exam tomorrow" narrows through five questions (topics unsure about → fractions → the specific "keep change flip" method → which fraction gets inverted → mixed numbers) to land on a precise, correct mechanical diagnosis: the student was inverting the wrong fraction. The conclusion explicitly declined to rule out hypotheses the evidence didn't actually address, rather than overclaiming.

**A typo'd, ambiguous opener.** "You mentioned a 3-abel cake" (a typo for "3-layer") — instead of stalling, the detective hypothesized what was meant, then investigated its way (tiers → filling → vertical stacking → a mysterious "stick" inside it) to a sensible read: a decorative topper, not a structural support. Demonstrates the persona working on something low-stakes and slightly silly, not just serious topics.

**Exam anxiety, resolved by the evidence itself.** "I have a math exam tomorrow, I'm afraid I won't do well" → narrowed to a specific claimed weak spot (factorization) → revealed an 8/10 practice score on exactly that topic. Conclusion: *"Your anxiety is likely disproportionate to your actual performance"* — a genuinely useful insight the person hadn't stated themselves, derived from evidence they'd given.

**A gift, narrowed to what's actually useful.** "I want to buy my friend a gift" → friend likes Disney, specifically Ariel, already owns an Ariel doll and an Ariel purse → conclusion: *"Your friend's active display of Ariel items signals a deep, ongoing passion, making it essential to find a specific gift that complements rather than duplicates their existing collection."* A practical, non-obvious synthesis (avoid duplicating what they already have) rather than a generic "get them a Disney item."

**Honest uncertainty, not overclaiming.** A baking-powder-substitution case closed with: *"No hypotheses are ruled out because the evidence does not contain facts that directly contradict them. The uncertainty regarding the specific ratio remains unconfirmed rather than disproven."* — the grounding rules working as intended: not pretending to know more than the evidence supports.

**Few-shot ablation, side by side** (same inputs, only the worked examples differ):

> *Without examples* — "The painting disappeared from the museum overnight while the alarm remained silent and the night guard's log recorded no visitors." *(restates the evidence)*
>
> *With examples* — "The painting was likely removed by an individual with the ability to bypass security, as the alarm remained silent and the guard recorded no visitors." *(draws an inference)*

**A bug caught in the act, and what fixed it.** In one real session investigating a coding exam ("How many of the 20 practice problems have you completed for each of the two syllabus topics?"), the exact same question was asked three turns in a row, because the question-generation stage had no memory of what it had already asked. It just saw the same unfilled evidence gap each time and proposed the same question again. This burned the entire 5-question budget on one unproductive loop and forced a premature, hedged close. The fix: track every question asked in `case_state["asked_questions"]`, pass that history into the question-generation prompt with an explicit "do not repeat any of these" instruction, and track `stall_count` so the decision stage is nudged toward closing once the same gap has gone unfilled twice, rather than continuing to ask into a dead end.

## 8. Known Limitations

- **The persona can feel clinical on emotional topics.** Built for practical troubleshooting, the investigative format can read as overly probing when applied to something like "I'm sad" or "I miss my friend", pressing for specifics (sleep patterns, physical symptoms, exact timelines) where a person might want space rather than an interrogation. Not a bug, but a real fit issue worth being upfront about.
- **Reasoning Mode can sound more confident than it should**, particularly on medical-style judgment calls. The underlying deduction can be sound while the tone reads as more authoritative than a chatbot should be for anything health-related.
- **Evidence extraction is a single LLM judgment call per turn**, with one retry as a safety net (added after real testing caught two consecutive dropped facts, direct, substantive answers to specific questions that a model judged "nothing extractable"). The retry recovers the common case; it is not a hard guarantee, and a determined model could still fail both passes.
- **Development was constrained by Groq's free-tier daily quota**, which was exhausted and hit the automatic model-fallback path during testing. Some live-tested behavior described in this document, therefore ran on whichever model `PREFERRED_MODELS` selected at that moment, not necessarily `qwen/qwen3.8-27b` specifically in every case. The app is designed to behave consistently across the preference list, but exhaustive cross-model verification of every feature was not feasible within the free tier's limits.
