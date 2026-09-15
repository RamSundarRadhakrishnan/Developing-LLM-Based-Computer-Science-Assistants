# Developing LLM-Based Computer Science Assistants

An ongoing capstone project on building a Computer Science tutoring assistant that can provide useful, assignment-aware guidance without completing assessed work for the student.

The project investigates how to enforce this boundary reliably and with acceptable latency. It treats safety as a context-dependent classification problem: the assignment specification, the student's request, the assistant's response, and—in some cases—the preceding conversation must be considered together.

## Scope

The proposed system is a modular tutoring pipeline with layered guardrails around an open-source language model. It is designed to permit explanations of concepts, syntax, APIs, errors, complexity, and debugging strategies, while detecting content that materially gives away an assignment solution.

The work covers the following components:

- **Input guardrails** to identify irrelevant requests, prompt-injection attempts, policy-bypass attempts, and other queries that should not be handled as ordinary tutoring requests.
- **Output guardrails** to detect solution-proximal responses, including complete code, algorithms, pseudocode, flowcharts, or step-by-step procedures that would enable completion of the assigned task.
- **Session-aware checks** to identify cumulative leakage across multiple individually benign turns, using semantic similarity over the conversation history.
- **Complementary techniques** ranging from lightweight heuristics and regular-expression checks to encoder-based classifiers and LLM-as-a-judge evaluation.
- **Canary-token checks** to detect unintended leakage of protected context in generated responses.
- **Preference-based fine-tuning** of a tutor model, using policy-compliant preferred responses in place of unsafe solution-giving responses. The planned approach uses QLoRA and Direct Preference Optimisation (DPO), followed by comparison with the base model.

The project evaluates each technique on both effectiveness and latency, then uses the results to select a practical guardrail configuration rather than assuming one method is best in every case.

## Current Status

The project is in the implementation and evaluation stage. The repository currently contains data-preparation utilities, prototype guardrail modules, labelled data, and initial benchmark outputs; it does not yet contain a finished end-user tutoring application.

| Area | Status | Evidence in this repository |
| --- | --- | --- |
| Policy definitions and assignment-aware judging | Implemented | Granite Guardian prompts incorporate the assignment context, conversation, and explicit definitions of `irrelevancy` and `solution_proximity`. |
| Golden-dataset preparation | Implemented | Conversation anonymisation, assignment-context matching, Granite Guardian scoring, and cross-check outputs are included in `Golden_Dataset_Preparation/` and `golden_dataset/`. |
| Heuristic and session-level safeguards | Implemented as prototypes | Prompt-injection heuristics, canary-token checks, and a MiniLM-based semantic drift detector are included in `encoder_classifier/`. |
| Encoder classifier experiments | In progress | DistilBERT experiment notebooks and synthetic/organic training data are included in `encoder_classifier/`. |
| LLM-as-a-judge evaluation | Initial benchmark complete | `LASJ_Finetuning/granite_guardian_eval.py` evaluates Granite Guardian 3.1 2B, with predictions, metrics, and confusion matrices stored in `granite_guardian_results/`. |
| Tutor-model DPO fine-tuning | Planned / experimental | The training direction and exploratory work are present, but a final fine-tuned tutor checkpoint and comparative evaluation are not yet part of this repository. |
| Integrated deployment | Planned | Integration with the tutor model and deployment platform remains future work. |

### Initial LLM-as-a-Judge Results

The first benchmark evaluates **IBM Granite Guardian 3.1 2B** served locally through `llama.cpp` in FP16. These figures are interim results for the currently available evaluation split, not final project claims.

| Risk | Accuracy | Precision | Recall | F1 | Mean latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Irrelevancy | 83.26% | 64.91% | 94.28% | 76.88% | 94.2 ms |
| Solution proximity | 80.13% | 73.92% | 50.18% | 59.78% | 130.2 ms |

The results show a useful low-latency baseline, with particularly high recall for irrelevancy. Solution-proximity recall remains an important area for improvement, to be addressed through prompt/model comparisons, threshold analysis, encoder baselines, and subsequent fine-tuning experiments.

## Repository Structure

```text
.
├── Golden_Dataset_Preparation/   # Anonymisation, context mapping, and LLM-based scoring
├── LASJ_Finetuning/              # LLM-as-a-judge evaluation and fine-tuning exploration
├── encoder_classifier/           # Encoder experiments, heuristics, canaries, and drift detection
├── golden_dataset/               # Scored conversations and cross-checked outputs
└── granite_guardian_results/     # Predictions, metrics, and confusion matrices
```

## Evaluation Approach

Each guardrail is evaluated under the same policy definitions and dataset splits. The primary measures are precision, recall, F1 score, false-positive rate, and inference latency. The final comparison will assess whether the guardrail stack and DPO-trained tutor reduce code leakage while preserving the quality of explanations and avoiding unnecessary refusals.

## Project Status Note

This repository is an active research and implementation workspace. Scripts, notebooks, datasets, and results may evolve as the comparative experiments and tutor-model fine-tuning are completed.