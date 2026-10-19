# Evaluating Text-Based Scam and Spam Detection

This repository contains the code, datasets, processed outputs, experiment reports, and research manuscript for a study of text-based spam/scam detection. The experiments investigate dataset quality, generalization under different splits, adversarial robustness, multilingual transfer, abstention, operational efficiency, and explanation stability.

**Manuscript:** [Evaluating Text-Based Scam and Spam Detection: Generalization, Robustness, Multilingual Transfer, and Operational Efficiency](Evaluating%20Text-Based%20Scam%20and%20Spam%20Detection_%20Generalization,%20Robustness,%20Multilingual%20Transfer,%20and%20Operational%20Efficiency.pdf)

## Repository contents

| Experiment | Focus | Main artifacts |
|---|---|---|
| [1 — Dataset audit](<Experiment%201/>) | Dataset cleaning, label/class audit, duplicate checks, and stratified split construction | [Audit summary](<Experiment%201/reports/dataset_audit_summary.json>), [class distribution](<Experiment%201/reports/class_distribution.csv>), [split statistics](<Experiment%201/reports/split_statistics.csv>), [visualization](<Experiment%201/reports/experiment1_dataset_audit.svg>) |
| [2 — Random vs. grouped evaluation](<Experiment%202/>) | Compare evaluation results under different split strategies | [Notebook](<Experiment%202/Experiment_2_Random_vs_Grouped_Evaluation.ipynb>), [results](<Experiment%202/experiment2_results.csv>), [macro-F1 figure](<Experiment%202/experiment2_macro_f1.png>) |
| [3 — Adversarial text robustness](<Experiment%203/>) | Test how text transformations affect model predictions | [Code](<Experiment%203/scamshield_experiment3_code/experiment3.py>), [clean vs. attacked](<Experiment%203/scamshield_experiment3_code/experiment3_results/experiment3_clean_vs_attacked.png>), [attack success rate](<Experiment%203/scamshield_experiment3_code/experiment3_results/experiment3_attack_success_rate.png>), [metrics](<Experiment%203/scamshield_experiment3_code/experiment3_results/experiment3_metrics.csv>) |
| [4 — Multilingual transfer](<Experiment%204/>) | Evaluate performance across the languages and data conditions defined by the experiment | [Notebook](<Experiment%204/experiment4_colab_code/Experiment4_Multilingual_Scam_Detection.ipynb>), [macro-F1 by language](<Experiment%204/experiment4_all_results/macro_f1_by_language.png>), [cross-language loss](<Experiment%204/experiment4_all_results/cross_language_loss.png>), [metrics](<Experiment%204/experiment4_all_results/experiment4_metrics.csv>) |
| [5 — Abstention and calibration](<Experiment%205/>) | Measure the relationship between prediction coverage, errors, calibration, and review workload | [Script](<Experiment%205/experiment5_abstention.py>), [coverage vs. error](<Experiment%205/experiment5_outputs/coverage_vs_error.png>), [calibration](<Experiment%205/experiment5_outputs/calibration_comparison.png>), [review workload](<Experiment%205/experiment5_outputs/review_workload.png>) |
| [6 — Deployment cost and efficiency](<Experiment%206/>) | Compare model performance and operational efficiency, including latency | [Script](<Experiment%206/experiment6_deployment_cost.py>), [performance vs. latency](<Experiment%206/experiment6_results/experiment6_performance_vs_latency.png>), [clean vs. adversarial](<Experiment%206/experiment6_results/experiment6_clean_vs_adversarial.png>), [paper table](<Experiment%206/experiment6_results/experiment6_paper_table.csv>) |
| [7 — Explanation stability](<Experiment%207/>) | Examine prediction consistency and the stability of feature explanations | [Script](<Experiment%207/experiment7_explanation_stability.py>), [explanation stability](<Experiment%207/experiment7_results/experiment7_explanation_stability.png>), [prediction consistency](<Experiment%207/experiment7_results/experiment7_prediction_consistency.png>), [scam feature importance](<Experiment%207/experiment7_results/experiment7_scam_feature_importance.png>), [legitimate feature importance](<Experiment%207/experiment7_results/experiment7_legitimate_feature_importance.png>) |

## Experiment results

### Experiment 1 — Dataset audit and split composition

Experiment 1 uses the repository's audit reports to document dataset preparation. The committed audit summary reports 5,574 initial rows, 415 exact duplicate rows removed, 5,159 final rows, no empty-text rows, no invalid-label rows, and no exact message texts crossing the train, validation, and test splits. The split uses stratified random sampling with seed 42.

![Experiment 1 dataset audit and split composition](<Experiment%201/reports/experiment1_dataset_audit.svg>)

Class distribution after exact-duplicate removal:

| Class | Count | Share |
|---|---:|---:|
| Legitimate (ham) | 4,517 | 87.6% |
| Spam | 642 | 12.4% |
| **Total** | **5,159** | **100%** |

Split counts in the committed report:

| Split | Ham (label 0) | Spam (label 1) | Total |
|---|---:|---:|---:|
| Train | 3,162 | 449 | 3,611 |
| Validation | 677 | 97 | 774 |
| Test | 678 | 96 | 774 |

The source dataset uses the label **spam**, which is not equivalent to verified fraud, phishing, or scam activity. Near-duplicate candidates are reported separately and require appropriate review. A random split may still place related message templates across different splits even when exact duplicate texts do not cross splits.

### Experiment 2 — Random versus grouped evaluation

This experiment compares performance across the split protocols implemented in the notebook. The repository includes results tables and a macro-F1 visualization.

![Experiment 2 macro-F1 results](<Experiment%202/experiment2_macro_f1.png>)

Interpret the comparison using the notebook's actual split definitions and results. Do not infer that grouped splitting eliminates all leakage unless the grouping strategy and data checks establish that.

### Experiment 3 — Adversarial text robustness

This experiment examines changes in model performance under the text transformations implemented in its script.

![Experiment 3 clean versus attacked results](<Experiment%203/scamshield_experiment3_code/experiment3_results/experiment3_clean_vs_attacked.png>)

![Experiment 3 attack success rate](<Experiment%203/scamshield_experiment3_code/experiment3_results/experiment3_attack_success_rate.png>)

Use the accompanying CSV and classification report for exact metrics and model-specific comparisons.

### Experiment 4 — Multilingual transfer

This experiment reports the language-level results and cross-language performance changes recorded in its outputs.

![Experiment 4 macro-F1 by language](<Experiment%204/experiment4_all_results/macro_f1_by_language.png>)

![Experiment 4 cross-language performance loss](<Experiment%204/experiment4_all_results/cross_language_loss.png>)

Claims must reflect the actual data construction. Translated English examples alone are not evidence of generalization to naturally occurring messages in the target language; report data provenance and evaluation conditions clearly.

### Experiment 5 — Abstention, calibration, and review workload

This experiment evaluates prediction abstention and related operating trade-offs using its saved outputs.

![Experiment 5 coverage versus error](<Experiment%205/experiment5_outputs/coverage_vs_error.png>)

![Experiment 5 calibration comparison](<Experiment%205/experiment5_outputs/calibration_comparison.png>)

![Experiment 5 review workload](<Experiment%205/experiment5_outputs/review_workload.png>)

Read these figures alongside the experiment's metrics and metadata. Coverage, the abstention threshold, and the population on which errors are measured should be stated when reporting results.

### Experiment 6 — Deployment efficiency

This experiment compares the operational measurements and model results recorded in its output files.

![Experiment 6 performance versus latency](<Experiment%206/experiment6_results/experiment6_performance_vs_latency.png>)

![Experiment 6 clean versus adversarial results](<Experiment%206/experiment6_results/experiment6_clean_vs_adversarial.png>)

Latency and efficiency results depend on the recorded environment and execution protocol. They should not be generalized to other hardware or production traffic without further measurement.

### Experiment 7 — Explanation stability

This experiment investigates the stability of explanations and prediction consistency in the implemented evaluation.

![Experiment 7 explanation stability](<Experiment%207/experiment7_results/experiment7_explanation_stability.png>)

![Experiment 7 prediction consistency](<Experiment%207/experiment7_results/experiment7_prediction_consistency.png>)

![Experiment 7 scam feature importance](<Experiment%207/experiment7_results/experiment7_scam_feature_importance.png>)

![Experiment 7 legitimate feature importance](<Experiment%207/experiment7_results/experiment7_legitimate_feature_importance.png>)

Feature-importance visualizations describe the behavior of the evaluated model and explanation method; they should not automatically be interpreted as causal evidence.

## Dataset and citation

The repository's Experiment 1 audit identifies the **UCI SMS Spam Collection** as its source dataset. The UCI repository describes it as a labelled SMS collection for mobile-phone spam research and lists the dataset under a CC BY 4.0 license. See the [official dataset page](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).

Please cite the dataset and its associated publication:

> Almeida, T. & Hidalgo, J. (2011). *SMS Spam Collection* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84

> Almeida, T. A., Gómez Hidalgo, J. M., & Yamakami, A. (2011). Contributions to the study of SMS spam filtering: new collection and results. *Proceedings of the 2011 ACM Symposium on Document Engineering*, 259–262. https://doi.org/10.1145/2034691.2034742

The repository also contains experiment-specific data and generated datasets. For publication, identify which source dataset and transformation pipeline each experiment uses, and cite any additional sources as appropriate. Follow the original dataset's attribution and license requirements.

## Reproducibility

The repository contains a mixture of Python scripts, notebooks, CSV result tables, processed data, and a PDF manuscript. Open each experiment folder for its specific entry point and outputs. Some experiments are provided as notebooks intended for an environment such as Google Colab; others are Python scripts.

For reproducible reporting:

1. Record the Python and package versions used for each run.
2. Preserve the exact split strategy, random seed, preprocessing, and model configuration.
3. Report metrics from committed outputs and distinguish measured results from interpretation.
4. Include dataset provenance, class counts, and evaluation sample sizes.
5. Avoid using test data to tune thresholds or select models.
6. Treat benchmark results as offline evaluations, not proof of real-world effectiveness.

No single root-level installation command is specified here because the repository currently contains experiment-specific scripts and notebooks rather than one verified, shared dependency file. Follow the imports and setup instructions in the relevant experiment code.

## Limitations

- The UCI SMS Spam Collection is an SMS spam benchmark; its labels do not establish that every positive example is a confirmed scam or fraud attempt.
- Exact-duplicate removal does not guarantee that semantically similar messages or campaign templates are separated across splits.
- Multilingual performance depends on the source, quality, and naturalness of the language-specific data.
- Adversarial robustness, abstention, explanation stability, and latency depend on the transformations, models, thresholds, and environment used in each experiment.
- Conclusions should be restricted to the reported data and protocols.

## License

No repository-wide code license is specified in this README. Check the licenses of the code and each included dataset before reuse. The source dataset's license does not automatically determine the license of this repository's code.
