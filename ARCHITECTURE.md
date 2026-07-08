# Room Matcher Architecture

This document explains the model architecture, the data flow, the training process, and the math behind the current room-matching system.

## 1. Problem Definition

We want a function

\[
f(q, C) \to M
\]

where:

- \(q\) is a single input room name
- \(C = \{c_1, c_2, \dots, c_n\}\) is a candidate list with \(5 \le n \le 20\)
- \(M \subseteq C\) is the subset of candidates that match \(q\)

The system solves this by reducing the list problem to repeated pair scoring:

\[
s(q, c_i) \in [0,1]
\]

Then the final matched set is:

\[
M = \{c_i \in C \mid s(q, c_i) \ge \tau\}
\]

where \(\tau\) is the threshold selected on validation data.

## 2. Data Source

The raw file is `room_matching.csv` with columns:

- `nuitee_room_name`
- `provider_room_name`

Each row is a positive match pair:

\[
(q, c, y=1)
\]

The raw data does not contain explicit negative examples, so negatives are generated during training.

## 3. Cleaning Pipeline

The cleaning logic lives in `room_matcher/cleaning.py`.

The text normalization function maps each raw string \(x\) to a normalized string \(N(x)\).

### 3.1 Normalization Function

For a room string \(x\), the pipeline does:

1. Unicode normalization
2. ASCII folding
3. lowercasing
4. abbreviation replacement
5. number-word replacement
6. punctuation removal
7. whitespace collapse

So:

\[
N : \text{raw text} \to \text{normalized text}
\]

Examples:

- `Non Smoking` \(\to\) `nonsmoking`
- `kg` \(\to\) `king`
- `qn` \(\to\) `queen`
- `two` \(\to\) `2`

### 3.2 Pair Aggregation

If the same positive pair appears multiple times, it is merged:

\[
((q, c), 1), ((q, c), 1), \dots \to ((q, c), \text{pair\_count}=k)
\]

This produces a deduplicated table with:

- `room_name`
- `candidate_room`
- `room_name_normalized`
- `candidate_room_normalized`
- `pair_count`
- `provider_room_ambiguity`

### 3.3 Ambiguity

For a provider room string \(c\), define:

\[
A(c) = |\{q : (q,c)\text{ exists}\}|
\]

If \(A(c) > 1\), the candidate text is ambiguous because the same provider room name maps to multiple target room names.

By default, ambiguous rows are dropped during training to reduce contradictory supervision.

## 4. Supervised Training Set Construction

After cleaning, the data is still positive-only:

\[
\mathcal{D}^+ = \{(q_i, c_i, 1)\}
\]

We need negative pairs:

\[
\mathcal{D}^- = \{(q_i, c_j, 0)\}
\]

where \(c_j\) is a candidate that does not belong to the positive set of \(q_i\).

### 4.1 Positive Lookup

For each query room \(q\), define the true candidate set:

\[
P(q) = \{c : (q,c)\in \mathcal{D}^+\}
\]

### 4.2 Negative Sampling

For each positive pair \((q,c)\), the code samples negatives from:

1. a hard-negative pool built from overlapping informative tokens
2. a random candidate fallback pool

So for each positive pair, training creates:

\[
(q, c, 1), (q, c^-_1, 0), \dots, (q, c^-_m, 0)
\]

where \(m\) is controlled by `--negatives-per-positive`.

This converts the dataset into binary classification examples.

## 5. Train / Validation / Test Split

The split is deterministic and hash-based.

For each normalized pair key:

\[
k = N(q) \, ||| \, N(c)
\]

we compute a stable bucket:

\[
b(k) \in \{0,\dots,99\}
\]

Then:

- train if \(b < 80\)
- validation if \(80 \le b < 90\)
- test if \(90 \le b < 100\)

This gives a stable split without needing a saved split file.

## 6. Baseline Model Architecture

The baseline model lives in `room_matcher/model.py`.

It is a pairwise binary classifier:

\[
s(q,c) = \Pr(y=1 \mid q,c)
\]

### 6.1 Feature Map

The final feature vector is:

\[
x(q,c) =
\begin{bmatrix}
h(q,c) \\
g(q,c)
\end{bmatrix}
\]

where:

- \(h(q,c)\) is a hashed sparse character n-gram vector
- \(g(q,c)\) is a dense numeric feature vector

### 6.2 Character N-gram Features

The text feature uses a hashed character bag with:

- analyzer: `char_wb`
- n-grams: \(3\) to \(5\)
- dimension: \(2^{18}\)

Conceptually:

\[
h(q,c) = \text{HashVectorizer}(\text{pair\_text}(q,c))
\]

where the pair text is built as:

\[
\text{pair\_text}(q,c)
=
\text{"query "}N(q)
+ \text{" candidate "}N(c)
+ \text{" cross "}N(q)N(c)
\]

This makes the model sensitive to spelling patterns, shared fragments, abbreviations, and local text structure.

### 6.3 Dense Numeric Features

The dense vector \(g(q,c)\) contains:

1. token Jaccard overlap
2. overlap relative to query tokens
3. overlap relative to candidate tokens
4. number overlap
5. bed-type overlap
6. attribute agreement score
7. query contained in candidate
8. candidate contained in query
9. normalized length difference

#### Token Jaccard

If \(T_q\) and \(T_c\) are token sets:

\[
\text{Jaccard}(q,c)=\frac{|T_q \cap T_c|}{|T_q \cup T_c|}
\]

#### Directional Overlap

\[
\text{OverlapLeft}(q,c)=\frac{|T_q \cap T_c|}{|T_q|}
\]

\[
\text{OverlapRight}(q,c)=\frac{|T_q \cap T_c|}{|T_c|}
\]

#### Number Overlap

Let \(D_q\) and \(D_c\) be numeric token sets, such as `1`, `2`.

\[
\text{NumOverlap}(q,c)=
\frac{|D_q \cap D_c|}{|D_q \cup D_c|}
\]

when the union is non-empty, otherwise \(0\).

#### Bed-Type Overlap

Let \(B_q\) and \(B_c\) be tokens from:

\[
\{\text{king}, \text{queen}, \text{double}, \text{twin}, \text{single}, \text{sofa}, \text{bunk}\}
\]

Then:

\[
\text{BedOverlap}(q,c)=
\frac{|B_q \cap B_c|}{|B_q \cup B_c|}
\]

#### Attribute Agreement

Let the attribute set be:

\[
\mathcal{A} =
\{\text{accessible}, \text{nonsmoking}, \text{smoking}, \text{suite}, \text{studio}, \text{deluxe}, \text{superior}, \text{premium}, \text{family}, \text{junior}, \text{executive}\}
\]

Then:

\[
\text{AttrAgree}(q,c)
=
\frac{1}{|\mathcal{A}|}
\sum_{a\in\mathcal{A}}
\mathbf{1}[(a \in T_q) = (a \in T_c)]
\]

This is a feature-alignment term.

#### Containment

\[
\text{ContainLeft}(q,c)=\mathbf{1}[N(q) \subseteq N(c)]
\]

\[
\text{ContainRight}(q,c)=\mathbf{1}[N(c) \subseteq N(q)]
\]

#### Length Difference

If \(L_q = |N(q)|\) and \(L_c = |N(c)|\), then:

\[
\text{LenDiff}(q,c)=\frac{|L_q-L_c|}{\max(L_q,L_c,1)}
\]

## 7. Classifier

The classifier is logistic regression trained with SGD:

\[
z(q,c) = w^\top x(q,c) + b
\]

\[
s(q,c)=\sigma(z)=\frac{1}{1+e^{-z}}
\]

So the predicted probability of a match is:

\[
\Pr(y=1\mid q,c)=s(q,c)
\]

## 8. Weighted Training Objective

Repeated pairs should matter more than one-off pairs.

If a pair appears \(k\) times in the raw data, its training weight is:

\[
\alpha(k)=1+\log(1+k)
\]

The weighted binary cross-entropy objective is:

\[
\mathcal{L}
=
\sum_i
\alpha_i
\left[
-y_i \log s_i
-(1-y_i)\log(1-s_i)
\right]
\]

where:

- \(y_i \in \{0,1\}\)
- \(s_i = s(q_i,c_i)\)
- \(\alpha_i\) is the pair weight

SGDClassifier optimizes this approximately with mini-batch updates through `partial_fit`.

## 9. Scenario-Based Evaluation

The real task is not pair classification in isolation.
The real task is:

- one query room
- a list of candidates
- return the matching subset

So evaluation uses candidate-list scenarios.

For a room \(q\), build:

\[
C_q = \{c_1,\dots,c_n\}, \quad 5 \le n \le 20
\]

with:

- \(1\) to \(3\) true positives
- the rest sampled negatives

Each candidate gets a score:

\[
s_i = s(q,c_i)
\]

The predicted match set is:

\[
\hat{M}_q(\tau) = \{c_i \in C_q : s_i \ge \tau\}
\]

The true set is:

\[
M_q = \{c_i \in C_q : y_i=1\}
\]

## 10. Threshold Selection

Threshold tuning is done on validation scenarios.

For thresholds:

\[
\tau \in \{0.20, 0.25, 0.30, \dots, 0.85\}
\]

the system computes mean scenario precision, recall, and F1.

The selected threshold is:

\[
\tau^* = \arg\max_\tau \text{F1}_{\text{val}}(\tau)
\]

with precision and recall used as tie-breakers.

## 11. Evaluation Metrics

For one scenario:

\[
\text{TP} = |\hat{M}_q \cap M_q|
\]

\[
\text{FP} = |\hat{M}_q \setminus M_q|
\]

\[
\text{FN} = |M_q \setminus \hat{M}_q|
\]

Then:

\[
\text{Precision} = \frac{\text{TP}}{|\hat{M}_q|}
\]

\[
\text{Recall} = \frac{\text{TP}}{|M_q|}
\]

\[
\text{F1}=
\frac{2PR}{P+R}
\]

if \(P+R > 0\), otherwise \(0\).

The reported metrics are means across scenarios:

\[
\overline{P},\;\overline{R},\;\overline{F1}
\]

There is also exact match rate:

\[
\text{ExactMatchRate}
=
\frac{1}{N}
\sum_{q=1}^{N}
\mathbf{1}[\hat{M}_q = M_q]
\]

## 12. “Before Training” Baseline

The command `room_matcher.evaluate --baseline-only` does not use an untrained logistic model.

Instead it uses a heuristic overlap scorer from `room_matcher/evaluate.py`.

That heuristic defines:

\[
s_{\text{heur}}(q,c)
=
0.45\cdot \text{Jaccard}
+ 0.20\cdot \text{NumOverlap}
+ 0.20\cdot \text{BedOverlap}
+ 0.10\cdot \text{Containment}
+ 0.05\cdot \text{AttributeOverlap}
\]

This gives a meaningful pre-training baseline for comparison.

## 13. Inference Pipeline

At API time:

1. receive `room_name`
2. receive `candidate_rooms`
3. compute \(s(q,c_i)\) for each candidate
4. apply threshold \(\tau\)
5. return:
   - `matched_rooms`
   - `scored_candidates`

So the inference rule is:

\[
\hat{M} = \{c_i : s(q,c_i) \ge \tau\}
\]

The API entry point is `room_matcher/api.py`.

## 14. Hugging Face Model

The optional second model lives in `room_matcher/model2.py`.

It uses a multilingual transformer as a cross-encoder:

\[
(q,c) \to \text{Transformer}([q;c]) \to \text{logits} \to \text{softmax}
\]

The default checkpoint is:

- `microsoft/Multilingual-MiniLM-L12-H384`

with tokenizer override:

- `xlm-roberta-base`

### 14.1 HF Pair Classification

For each pair \((q,c)\), the transformer outputs logits:

\[
\ell(q,c) = (\ell_0, \ell_1)
\]

Then:

\[
\Pr(y=1 \mid q,c)
=
\frac{e^{\ell_1}}{e^{\ell_0}+e^{\ell_1}}
\]

This score replaces the logistic baseline score, but the downstream thresholding and scenario evaluation are the same.

### 14.2 HF Training Objective

The transformer model is trained as a standard binary sequence-classification model with cross-entropy:

\[
\mathcal{L}_{\text{HF}}
=
-
\sum_i
\log \Pr(y_i \mid q_i,c_i)
\]

The pair construction, split logic, and candidate-list evaluation remain aligned with the baseline model so the comparison is fair.

## 15. Files Produced

### Baseline

- `artifacts/baseline/room_matching_clean.csv`
- `artifacts/baseline/room_matching_clean.sqlite3`
- `artifacts/baseline/room_matcher.joblib`
- `reports/baseline_cleaning_summary.json`
- `reports/baseline_training_summary.json`
- `reports/baseline_threshold_grid.json`
- `reports/baseline_sample_predictions.json`
- `reports/baseline_evaluation_before_training.json`
- `reports/baseline_evaluation_after_training.json`

### Hugging Face

- `artifacts/hf/room_matching_clean.csv`
- `artifacts/hf/room_matching_clean.sqlite3`
- `artifacts/hf/room_matcher/`
- `reports/hf_cleaning_summary.json`
- `reports/hf_training_summary.json`
- `reports/hf_threshold_grid.json`
- `reports/hf_sample_predictions.json`
- `reports/hf_evaluation_after_training.json`

## 16. End-to-End Command Flow

### Baseline

Clean:

```bash
uv run python -m room_matcher.clean --input-csv room_matching.csv --max-clean-rows 50000
```

Evaluate before training:

```bash
uv run python -m room_matcher.evaluate --input-csv room_matching.csv --max-clean-rows 50000 --max-positive-pairs 5000 --baseline-only --rebuild-cleaned
```

Train:

```bash
uv run python -m room_matcher.train --input-csv room_matching.csv --max-clean-rows 50000 --max-positive-pairs 5000 --rebuild-cleaned
```

Evaluate after training:

```bash
uv run python -m room_matcher.evaluate --model-type baseline --model-path artifacts/baseline/room_matcher.joblib --input-csv room_matching.csv --max-clean-rows 50000 --max-positive-pairs 5000 --rebuild-cleaned
```

Serve baseline model:

```bash
ROOM_MATCHER_MODEL_TYPE=baseline uv run uvicorn room_matcher.api:app --host 0.0.0.0 --port 8000 --reload
```

### Hugging Face

Install optional dependencies:

```bash
uv sync --group dev --extra hf
```

Train HF model:

```bash
uv run python -m room_matcher.model2 --input-csv room_matching.csv --max-clean-rows 50000 --max-positive-pairs 5000 --rebuild-cleaned
```

Evaluate HF model:

```bash
uv run python -m room_matcher.evaluate --model-type hf --model-path artifacts/hf/room_matcher --input-csv room_matching.csv --max-clean-rows 50000 --max-positive-pairs 5000 --rebuild-cleaned
```

Serve HF model:

```bash
ROOM_MATCHER_MODEL_TYPE=hf ROOM_MATCHER_MODEL_PATH=artifacts/hf/room_matcher uv run uvicorn room_matcher.api:app --host 0.0.0.0 --port 8000 --reload
```

## 17. Summary

The current system is a pairwise ranking/classification setup.

Mathematically:

1. normalize text
2. create positive and negative room pairs
3. learn a score function \(s(q,c)\)
4. tune threshold \(\tau\)
5. predict matches with \(s(q,c) \ge \tau\)

The baseline model learns:

\[
s(q,c)=\sigma(w^\top x(q,c)+b)
\]

and the transformer model learns:

\[
s(q,c)=\text{softmax}(\ell(q,c))_1
\]

Both are evaluated in the same list-of-candidates setting, which is the real product behavior.
