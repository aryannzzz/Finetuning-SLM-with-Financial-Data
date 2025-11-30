# Bond Intent Router (SLM) 🧠📈

A production-style **bond query intent classifier** and **router** built on top of
`microsoft/deberta-v3-small`. It classifies user queries into **15 intents**
(13 bond-specific + 2 non-bond router intents) and also predicts:

- bond **intent**
- **sector** multi-labels
- **rating** bucket
- **duration** bucket
- **constraints** (5 binary flags)

The 2 extra non-bond intents (`non_bond_search`, `non_bond_llm`) allow you to
**route queries away from the bond engine** to either a search backend or an
LLM backend.

All training data is **synthetic + augmented**, generated using a mix of:
- hand-crafted templates,
- Gemini (Google GenAI) synthetic generation,
- augmentation and carefully designed **edge cases** for equities, cash flows,
  generic finance, etc. so the model doesn’t misclassify non-bond queries as bonds.


---

## 1. Project Overview

### Problem

You have a unified text interface where users can type queries like:

- “Reduce my duration but keep yield above 7%.”
- “Explain why you suggested switching into PSU Energy.”
- “Show me last quarter’s cash flow statement for TCS.”
- “Summarise the Fed’s latest meeting minutes.”

You need a **lightweight classifier** that:

1. **Understands bond-specific intents** (reduce_duration, increase_yield, etc.).
2. **Detects non-bond queries** and routes them to:
   - a **search stack** (`non_bond_search`), or
   - a **generic LLM** (`non_bond_llm`).
3. Extracts enough structure (sector, rating, constraints) to plug into downstream
   rule-based / optimization / retrieval logic.

### Solution

This repo contains:

- A **master dataset** (`*_final.jsonl`) with:
  - bond intents (13 classes),
  - non-bond router intents (2 classes),
  - synthetic + Gemini-generated **edge cases** (stocks, generic finance, etc.).
- A **DeBERTa v3 small** based classifier (`ProductionBondClassifier`) trained
  **from scratch** on the final dataset.
- A **router-ready inference wrapper** to map intents to:
  - `bond` (send to bond engine),
  - `search` (non_bond_search),
  - `llm` (non_bond_llm).

---

## 2. Intent Schema

### 2.1 Bond Intents (13)

| ID | Intent                | Description                                                  |
|----|-----------------------|--------------------------------------------------------------|
| 0  | buy_recommendation    | User wants bond ideas to buy, often with filters            |
| 1  | sell_recommendation   | User wants to sell / exit certain bonds                     |
| 2  | portfolio_analysis    | Analyze current bond portfolio, risks, diversification      |
| 3  | reduce_duration       | Reduce interest rate / duration risk                        |
| 4  | increase_yield        | Increase portfolio yield / returns                          |
| 5  | hedge_volatility      | Hedge against rate / price volatility                       |
| 6  | sector_rebalance      | Rebalance sector allocation / concentration                 |
| 7  | barbell_strategy      | Construct barbell strategies (short + long duration)        |
| 8  | switch_bonds          | Switch from one bond/issuer to another                      |
| 9  | explain_recommendation| Explain rationale behind a recommendation / trade idea      |
| 10 | market_outlook        | Ask about bond market outlook, rates, spreads, etc.         |
| 11 | credit_analysis       | Analyze credit quality, default risk, rating changes        |
| 12 | forecast_prices       | Forecast bond prices / yields / returns                     |

### 2.2 Non-Bond Router Intents (2)

| ID | Intent           | Route   | Description                                               |
|----|------------------|---------|-----------------------------------------------------------|
| 13 | non_bond_search  | search  | Non-bond queries best handled via search / retrieval     |
| 14 | non_bond_llm     | llm     | Non-bond queries best handled by a generic LLM response  |

Router logic is simple:

```python
def route_from_intent(intent: str) -> str:
    if intent == "non_bond_search":
        return "search"
    elif intent == "non_bond_llm":
        return "llm"
    else:
        return "bond"
# Finetuning-SLM-with-Financial-Data
