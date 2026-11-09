# 04 Evaluation · TabArena

**Research activity · STATS 201 Week 3**

[Read Erickson et al. (2025), TabArena: A Living Benchmark for Machine Learning on Tabular Data, NeurIPS Datasets and Benchmarks](https://papers.neurips.cc/paper_files/paper/2025/file/1697e3fb412da11dc9488249f9e7bbc9-Paper-Datasets_and_Benchmarks_Track.pdf) · [Authors' benchmark code and result artifacts](https://github.com/autogluon/tabarena).

[Open the standalone Evaluation Colab](https://colab.research.google.com/github/sunshineluyao/ml4ss/blob/main/modules/04-evaluation/week03_04_tabarena_evaluation.ipynb). Fit two classifiers and a class-prior baseline on the same five folds. Compare log loss and balanced accuracy using bundled scikit-learn data. The example's numerical results come from the classroom experiment.

## The prediction task and the evaluation question

In a prediction task, a fitted method outputs a class or numerical value for an unseen case. In an evaluation, we decide what evidence to collect and how to compare predictions. TabArena provides a protocol and reference results for comparing supervised methods on tabular classification and regression tasks. The paper's initial suite uses 51 datasets and 16 curated models, with AutoGluon as a reference pipeline.

## Read §2.2–2.3 and Figure 4

| Evaluation choice | Find it in the paper | Ask what would change |
|---|---|---|
| Dataset eligibility | Figure 4 separates 33 TabPFNv2-compatible datasets from 36 TabICL-compatible classification datasets. | Which methods can be compared on the same eligible tasks? |
| Holdout and tuning | §2.3 repeats outer test folds. §2.1 and §2.3 describe inner validation for selecting configurations. | What happens if we choose a model using the data that score it? |
| Metrics and aggregation | §2.3 uses ROC AUC for binary classification, log loss for multiclass classification and RMSE for regression in its main results. Elo summarizes pairwise rankings. | What information about the magnitude and consequences of errors does a ranking omit? |
| Baseline and budget | Figure 4 includes the AutoGluon 1.3 four-hour reference. §3 describes the hardware and methods. | Would the comparison support the same decision under your time or hardware limits? |

## Apply one evaluation rule to the robot lab

In the [Week 3 Prediction Colab](https://colab.research.google.com/github/sunshineluyao/ml4ss/blob/main/modules/03-prediction/week03_03_robot_prediction.ipynb), the logistic classifier and the always-No baseline each match one of the two held-out labels at threshold 0.50. Their accuracy is 1/2 and their Yes recall is 0/1. The two frames come from one trajectory and illustrate a false negative.

Design the next evaluation. Identify the independent test unit, which cases to hold out, the common baseline and metric, and whose experience is affected by the error. For example, hold out complete trajectories, choose a threshold using validation trajectories, and report Yes recall with the number of actual Yes cases. Treat this as a classroom transfer of the evaluation idea to images. TabArena's reported results concern tabular data.

**Ed reply.** Cite the paper section or figure, report one score or paired difference from the Colab, and explain how you would apply one evaluation rule to your own prediction task.
