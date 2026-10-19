# Scam and Spam Message Detection: Experimental Evaluation

This repository accompanies research on machine-learning methods for SMS spam and scam-message detection. It is intended to document experimental protocols, make results inspectable, and support reproducibility.

> **Repository status:** The repository was empty when this README was created. The experiment scripts, result tables, and figure files must be committed before the experiments can be reproduced or the figures displayed. The figure paths below are intended locations, not existing result images.

## Research objectives

The evaluation considers performance beyond a single aggregate score, including generalization across split strategies, class imbalance, deployment-oriented evaluation, and abstention or uncertainty-aware prediction where implemented. Multilingual claims require evaluation on appropriately sourced messages in each language; machine-translated English messages alone do not establish multilingual generalization.

## Suggested repository structure

```text
scam-spam-evaluation/
├── README.md
├── requirements.txt
├── data/README.md
├── experiments/
│   ├── experiment1_deployment.py
│   ├── experiment2_generalization.py
│   ├── experiment3_<description>.py
│   ├── experiment4_<description>.py
│   └── experiment5_abstention.py
└── results/
    ├── tables/
    └── figures/
        ├── experiment1_deployment.png
        ├── experiment2_generalization.png
        ├── experiment3_results.png
        ├── experiment4_results.png
        └── experiment5_abstention.png
```

This is a suggested layout, not a claim that all listed files exist. Rename the paths to match the actual files committed to the repository.

## Experiments and result figures

Each experiment should document its research question, dataset and split, preprocessing, model configuration, metrics, saved output tables, and limitations. Generate figures directly from saved experimental outputs.

### Experiment 1 — Deployment-oriented evaluation

Evaluate the detector under the deployment conditions implemented in the script. Report test-set composition, decision threshold, precision, recall, F1-score, confusion matrix, and any latency or resource measurements actually collected. Distinguish offline benchmark performance from real-world deployment performance.



### Experiment 2 — Generalization and split strategy

Compare the implemented data-splitting protocols and assess how repeated or closely related messages affect evaluation. Report each strategy separately and document deduplication and leakage controls. Do not claim statistical significance without an appropriate test.

![Experiment 2 generalization results](results/figures/experiment2_generalization.png)

### Experiment 3 — Additional evaluation

Replace this section with the actual research question and protocol for Experiment 3. Include only metrics computed by the experiment and add the corresponding figure file.

![Experiment 3 results](results/figures/experiment3_results.png)

### Experiment 4 — Additional evaluation

Replace this section with the actual research question and protocol for Experiment 4. If it evaluates language generalization, report each language separately and distinguish naturally occurring messages from translated or synthetic data.

![Experiment 4 results](results/figures/experiment4_results.png)

### Experiment 5 — Abstention and uncertain predictions

If implemented by the experiment script, evaluate how abstention affects coverage and errors among covered examples. Report the abstention rule or threshold, coverage, performance on covered examples, and the non-abstaining baseline on the same evaluation examples.

![Experiment 5 abstention results](results/figures/experiment5_abstention.png)

**Figure note:** These links render only after the corresponding PNG files are added at those paths. Do not publish placeholder plots as measured results. Figures should include clear axis labels, the evaluation protocol, and sample counts.

## Dataset and citation

If the experiments use the SMS Spam Collection, cite both the original publication and the UCI dataset record:

- Almeida, T. A., Hidalgo, J. M. G., & Yamakami, A. (2011). *Contributions to the study of SMS spam filtering: new collection and results*. Proceedings of the 2011 ACM Symposium on Document Engineering, 259–262. https://doi.org/10.1145/2034691.2034742
- UCI Machine Learning Repository. *SMS Spam Collection*. https://doi.org/10.24432/C5CC84
- Dataset page: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

Verify that each experiment actually uses this dataset before retaining this citation. Cite any additional datasets separately and document their sources, licenses, labels, preprocessing, and role. Do not imply an external dataset was created by this project.

## Evaluation metrics

Define the positive class explicitly and report class counts with all results.

- **Precision:** Proportion of predicted spam/scam messages that are labelled spam/scam.
- **Recall:** Proportion of labelled spam/scam messages detected by the model.
- **F1-score:** Harmonic mean of precision and recall.
- **Macro-F1:** Unweighted mean of class-level F1-scores.
- **PR-AUC:** Area under the precision–recall curve, useful with imbalanced classes.
- **Confusion matrix:** True positives, false positives, true negatives, and false negatives.

Accuracy alone is insufficient when class proportions are imbalanced.

## Reproducibility

After the scripts and dependency file have been committed, document the tested Python version and exact command for each experiment. Typical setup:

```bash
git clone https://github.com/983111/scam-spam-evaluation.git
cd scam-spam-evaluation
python -m venv .venv
```

Activate the environment, then install the actual dependencies and run the appropriate script. For example, once the named files exist:

```bash
pip install -r requirements.txt
python experiments/experiment2_generalization.py
```

The experiment command is illustrative until that script is present. Record random seeds, package versions, data-splitting rules, model parameters, and commands used to generate each table and figure. Do not commit credentials or sensitive data.

## Research integrity and limitations

- Report measured results without selectively hiding failures.
- Keep training, validation, and test data separate; fit preprocessing only on training data where applicable.
- Document duplicate removal and dataset-overlap checks.
- Identify whether results come from a benchmark, an external test set, or a real deployment.
- Do not claim multilingual capability, production readiness, or real-world effectiveness without suitable evaluation evidence.
- Preserve scripts and raw result tables so that figures and manuscript claims can be checked independently.

## License

Add a `LICENSE` file specifying the license for this repository. The repository code license does not automatically grant additional rights to third-party datasets; follow each dataset provider's terms.

## Contact

For questions about the code or research protocol, open a GitHub issue in this repository.
