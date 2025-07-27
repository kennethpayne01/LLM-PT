# An Analysis of AI Decision Making under Risk: Prospect Theory in Large Language Models

## Overview

This repository contains the code, data, and analysis for a series of experiments testing how Large Language Models (LLMs) make decisions under risk. The research investigates whether LLMs exhibit the same cognitive biases as humans, as described by Daniel Kahneman and Amos Tversky's landmark Prospect Theory. We test state-of-the-art models, including chain-of-thought "reasoners," across a range of narrative and purely symbolic scenarios to explore the interplay between language, context, and mathematical reasoning.

Our findings suggest that LLMs have, by acquiring human language, also acquired our heuristics and biases. However, these biases are not monolithic; they are highly dependent on the semantic context of the problem. This work reframes the debate about reasoning and memorization in LLMs, suggesting that for both humans and machines, language is the heuristic that does the reasoning.

## Key Findings

1.  **Context is Determinative:** LLMs exhibit strong, human-like framing effects in narrative scenarios but behave as perfect rational agents in purely mathematical tests. The language of a scenario, not just a simple gain/loss frame, dictates the decision-making heuristic.
2.  **No Universal Framing Effect:** Classic prospect theory (risk-seeking in losses, risk-averse in gains) emerges strongly in competitive contexts (e.g., sports, military conflict) but is dampened or even reversed in others (e.g., business, careers).
3.  **"Semantic Dampening" in Geopolitics:** The language of statecraft and international relations suppresses extreme risk-taking. In military scenarios, models consistently avoid the highest-risk "escalate" options, favoring more moderate, stability-focused heuristics.
4.  **Emergent "Cognitive Personalities":** Different LLMs show distinct and consistent reasoning styles. For instance, GPT-4o acts as a "polarized chameleon," adopting the purest form of a heuristic, while Claude is a "hawkish bureaucrat" in military contexts, and Gemini is a "nuanced interpreter" uniquely sensitive to linguistic shifts.
5.  **Implicit Risk Aversion as a Tie-Breaker:** When forced to choose between mathematically identical options in symbolic tests, models consistently default to the lowest-variance (risk-free) option—a heuristic they introduce themselves.
6.  **Spontaneous Theoretical Recognition:** Advanced models (like OpenAI's `o3`) can spontaneously identify the underlying theoretical framework (Prospect Theory) from a problem's mathematical structure alone, even when no semantic cues are provided.

## Experiments

This repository contains the scripts and data for three main experimental setups:

1.  **Narrative Scenarios (`run1_one_shot_prospect_theory.py`):** Tests five leading LLMs across seven rich narrative scenarios in civilian (business, careers, sports) and geopolitical (military, diplomacy) domains.
2.  **Symbolic Scenarios - Free-Choice (`symbolic_prospect_theory_corrected.py`):** Models are presented with purely mathematical problems where all options have equal expected values. They are free to declare indifference.
3.  **Symbolic Scenarios - Forced-Choice (`symbolic_forced_choice.py`):** The same mathematical setup as above, but models are required to choose a single optimal action, forcing them to employ a tie-breaking heuristic.

## Results Data

The complete raw data, including model decisions and rationales, is available in this repository.

-   **Symbolic (Free-Choice):** `symbolic_corrected_results_20250725_120430.csv` / `.json`
-   **Symbolic (Forced-Choice):** `symbolic_forced_choice_results_20250725_090406.csv` / `.json`
-   **Narrative (Civilian):** `civilian_prospect_theory_20250711_223018.csv` / `.json`
-   **Narrative (Military):** `Results for Arxiv/military_prospect_theory_consolidated_v3.csv`
-   **Coder Analysis:** `Results for Arxiv/coded_rationales_all.csv`

## How to Reproduce the Experiments

To replicate these experiments, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/kennethpayne01/LLM-PT.git
    cd LLM-PT
    ```

2.  **Set up a Python virtual environment:**
    ```bash
    python3 -m venv prospect_theory_env
    source prospect_theory_env/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure API Keys:**
    Create a file named `.env` in the root of the project and add your API keys:
    ```
    OPENAI_API_KEY="your_openai_api_key"
    ANTHROPIC_API_KEY="your_anthropic_api_key"
    GOOGLE_API_KEY="your_google_api_key"
    ```

5.  **Run the experiments:**
    You can run any of the main experimental scripts directly:
    ```bash
    # Run the free-choice symbolic experiment
    python symbolic_prospect_theory_corrected.py

    # Run the forced-choice symbolic experiment
    python symbolic_forced_choice.py

    # Run the narrative scenarios experiment
    python run1_one_shot_prospect_theory.py
    ``` 