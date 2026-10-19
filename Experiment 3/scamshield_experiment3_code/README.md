# ScamShield Experiment 3: Adversarial Evasion Testing

## 1. Install dependencies

```bash
python -m pip install pandas numpy scikit-learn matplotlib
```

## 2. Prepare your dataset

Your CSV must contain one message per row, with a text column and a label column. For example:

```csv
text,label
"Your account is suspended, verify now",scam
"Are we meeting at 5 pm?",legitimate
```

Use your real dataset and its actual column names. Do not create fake rows for the research results.

## 3. Run

From the folder containing `experiment3.py`:

```bash
python experiment3.py --data "your_dataset.csv" --text-col text --label-col label --output-dir experiment3_results
```

If the scam class is encoded as `1`:

```bash
python experiment3.py --data "your_dataset.csv" --text-col message --label-col label --positive-label 1 --output-dir experiment3_results
```

## 4. Output files

- `experiment3_metrics.csv`: clean and attacked metrics.
- `experiment3_transformation_by_model.csv`: transformation-by-model summary.
- `experiment3_attack_examples.csv`: original and transformed held-out examples plus model predictions.
- `experiment3_clean_vs_attacked.png`: macro-F1 comparison chart.
- `experiment3_attack_success_rate.png`: attack success rate chart.
- `experiment3_classification_reports.txt`: clean held-out classification reports.
- `experiment3_protocol.txt`: configuration and metric definition.

## 5. Human validation is required

The script can generate candidate variants, but it cannot guarantee that every variant preserves meaning. In particular, paraphrases are heuristic substitutions, not validated semantic paraphrases. Review `experiment3_attack_examples.csv` before treating variants as valid. Exclude ambiguous variants from the main evaluation.

To apply a reviewed allow-list, create a CSV with columns:

```csv
message_id,transformation,keep,notes
0,homoglyphs,1,Meaning unchanged
1,paraphrases,0,Meaning altered
```

`message_id` is the zero-based index within the held-out split, as exported. `keep` must be `1` to include a variant in the main metric. Then run:

```bash
python experiment3.py --data "your_dataset.csv" --text-col text --label-col label --review-file reviewed_variants.csv --output-dir experiment3_reviewed_results
```

Use the same data, split seed, and model settings when comparing experiments. Report the number of generated, reviewed, and retained variants for every transformation.

## 6. Interpretation

- Macro-F1: overall balance across classes.
- Scam recall: proportion of actual scam messages detected.
- Attack success rate: among scam messages detected correctly in the clean condition, the proportion misclassified as non-scam after transformation.
- Macro-F1 change: attacked macro-F1 minus clean macro-F1; negative means degradation.

Do not claim the best defense until you have run the experiment. Report measured results, not expected results.
