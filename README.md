# \# MarketLoop MCP Client

# 

# \## Install

# 

# pip install -r requirements.txt

# 

# \## Run

# 

# python client/cli.py

# 

# \## Memory \& Context Management

# 

# \### The problem

# 

# MarketLoop support calls that involve a return can run 35+ turns, most of

# them tool calls (order lookup, shipment tracking, inventory checks, fee

# lookups). The customer states the return reason once, near the start of the

# call (e.g. "item arrived damaged" vs. "changed my mind"). Per MarketLoop's

# return policy, that single early fact decides whether a 15% restocking fee

# applies at the end of the call. If a context-pruning strategy silently drops

# that detail while trimming tool noise, the agent risks charging a customer

# for damage that wasn't their fault — a real cost, not a hypothetical one.

# 

# This section covers the short-term memory layer built to survive that

# problem: a rolling buffer, four context-management strategies evaluated

# against a fixed long-context test suite, and a promote-or-drop router that

# decides what graduates from short-term to episodic memory. (Semantic memory

# consolidation and zone-based pruning are a teammate's piece and are not

# covered by this module yet.)

# 

# \### Context management strategies: comparison table

# 

# Evaluated with `context\_eval/comparison\_harness.py` against

# `context\_eval/scenario.py`'s fixed test suite: 12 cases, \~35 tool-noise turns

# each, across all 4 return-reason variants.

# 

# | Strategy | Accuracy | Avg tokens/run | Avg latency (ms) |

# |---|---|---|---|

# | Sliding window (last 10) | 0/12 | 45.0 | 0.006 |

# | Observation masking (keep last 3 tool outputs) | 12/12 | 32.8 | 0.011 |

# | Recursive summarization (every 10 turns) | 12/12 | 61.8 | 0.009 |

# | Zone-based pruning | pending (teammate's piece) | - | - |

# 

# \### Final choice: Observation Masking

# 

# \*\*We ship with Observation Masking as the default context strategy.\*\*

# 

# \- \*\*Sliding window fails outright (0/12).\*\* MarketLoop's transcripts are

# &#x20; tool-call-dominated: trimming by turn count throws away the return reason

# &#x20; the moment 10 tool calls pass, regardless of how important it was.

# \- \*\*Masking and summarization both hit 12/12\*\*, so accuracy alone doesn't

# &#x20; decide it. The tie-breaker is cost: masking uses roughly half the tokens

# &#x20; of summarization (32.8 vs 61.8 avg tokens/run). That's because masking

# &#x20; targets the actual bloat source directly — it keeps only the last 3 tool

# &#x20; outputs and leaves every dialogue turn untouched — while summarization

# &#x20; keeps a larger 10-turn recency window verbatim (including raw tool noise

# &#x20; inside that window) before folding anything older into a summary line.

# \- MarketLoop's failure mode is specifically \*\*tool-output bloat, not long

# &#x20; dialogue\*\* (each call has only 1-2 user turns that matter), so masking's

# &#x20; narrower, cheaper approach matches the real shape of the problem without

# &#x20; giving up any accuracy.

# 

# Sliding window is not used in production. Summarization remains implemented

# and benchmarked (it would be the better fit for a use case with many

# user-stated facts spread across a long dialogue), but is not the default.

# 

# \### Files

# 

# \- `mcp\_server/memory/rolling\_buffer.py` — short-term rolling message buffer

# \- `mcp\_server/memory/sliding\_window.py` — strategy 1

# \- `mcp\_server/memory/masking.py` — strategy 2 (shipped default)

# \- `mcp\_server/memory/summarization.py` — strategy 3

# \- `mcp\_server/memory/promote\_drop\_router.py` — forget-vs-episodic routing,

# &#x20; with reasoning logged per decision; never writes to semantic memory

# &#x20; directly

# \- `context\_eval/scenario.py` — fixed long-context test suite (do not edit

# &#x20; once evaluation has started)

# \- `context\_eval/comparison\_harness.py` — produces the table above

