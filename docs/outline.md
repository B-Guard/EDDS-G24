# EDDS Ex 2 outline

## The tasks

Reproduce the experimental setup, experiments, and results of the paper
Confirm the numbers, and findings reported, alternatively uncover inconsistencias and challenge the conclusions
- Is the information given in the paper sufficient to reproduce the results?
- Statistically significance differences can be made out between the different settings?
- Could values reported in the paper stem from the distribution sampled by the reproduced experiments?

Efforts must be documented in a report, accompanied by code and workflow used to reproduce the experiments.
- Outline necessary steps to reproduce results
- Detail strategies, difficulties, key findings and a conclusion about the experimental design.
Report must be the ACM Conference Proceedings single-column format, max 6. pages

Provide experiment descriptions in a machine readable format (details tba in Jan 25)

For each step:
- Document which information is given in the paper and which measures you took to implement such step
- Identify deviations from the numbers reported in the original paper, their origin and estimate whether they will have a significant impact on subsequent steps.
- Perform adequate tests to test for statistical significance.

Upload report + code/workflow to Zenodo
Prepare a very short slide deck (4-5 slides, max 3min) to present in January (precise date tba). Strategies, encountered difficulties and key findings so far must be presented. 

## The paper

Novel metrics for recency, popularity and diversity - metrics linked to perceived explainability by users.
New suite of re-ranking approaches that optimize the top k list of recommended products + their associated explanation according to the newly proposed metrics
Evaluation of this re-ranking on recommendation utility, considering the entire user base and specific demographical groups, using two real world public data sets.

Paper proposes a new objective function to maximize the sum of the expected recommendation utility AND the sum of explanation properties. However, the cardinality of this possible "explanation property" space is so large, it is assumed that the model will only be trained based on recommendation utility. Thus the new approach simply consists on re-ranking the list of recommended products to maximize for explanation properties and recommendation utility.

Selected explanation properties:
- Recency of the linking interaction
- Popularity of the shared entity
- Diversity of the explanation type

#### Design of the user study
5min questionnaire (https://tinyurl.com/exp-quality-survey) sent to a pool of users extracted from the general public covering several demographic groups (gender, age, country).

#### Operationalization
Better go see paper for formulas. Pages 5 and 6

#### Explanation property optimization
Soft and weighted. Soft seems lame (personal opinion).

### Experimental setup

Datasets: MovieLens 1M and LastFM-1B. For ML1M, a KG generated in [6] was used, discarding movies with less than 200 interactions. Likewise for LASTFM, the KG of [29] was used.

Metrics: NDCG with binary relevance scores and log2 decay + explanation quality by the 3 aforementioned new metrics.

Benchmarked models: matrix factorization (FM [23], NFM [22], BPR [22]), explainable recommendation based on: regularization (CFKG [2], CKE [33], KGAT [29]) and explanation paths (PGPR [32]). Focus is to improve the quality of the text explanation

### Results

Paper reports NDCG, LIR, SEP and ETD for three models (optimizing for R, P, or D), for different alphas (Figure 2).
Table 3 also shows comparison of NDCG from baselines compared to their models, but no explicit explanation of which alpha was used for their models is given.

Paper mentions having optimized two explanation properties jointly, but only reports "similar patterns to those observed for a single-property optimization".

Which alpha is used for Table 4 too?
Table 5 reports statistical difference under "non-parametric Kruskal Wallists test with p=0.05".

## Questions and comments

- Datasets are not truly cited, may be because they are relatively famous...? Anyways, let's hope they are uniquely identifiable via just their names.
- They report results of their user study, haven't checked if they also provide the answers to the study themselves. In case they do, do we need to also reproduce this results?
- Results reported for some RQs do not have any reference to statistical testing (only Table 5 reports such a test). Should we perform statistical test? If so, on which other data split?
- My understanding is, they trained the PGPR on train set and used val set to find ideal alpha (?). Then, trained baseline models with same train set and compared their performance (NDCG and other properties) on test set. What do you think? If such were the case, should we try to verify results by slightly modifying the train/val/test splits?


## Further comments:

Table 3 cannot be reproduced by the code in this repo. Need to look at other submodule. 