# Measured checkpoint results under memo 07 November

                        val_ndcg_10                 test_ndcg_10                
                               mean       std count         mean       std count
dataset family                                                                  
beauty  early_B            0.065755  0.000451     3     0.052198  0.001608     3
        early_BT           0.069011  0.000425     3     0.054759  0.000947     3
        early_C            0.060206  0.000608     3     0.046443  0.001569     3
        early_CB           0.069773  0.000360     3     0.055247  0.000775     3
        early_CBT          0.070060  0.000802     5     0.055165  0.000188     3
        early_CT           0.061920  0.000376     3     0.048601  0.000386     3
        early_T            0.064413  0.000247     3     0.050682  0.000277     3
        early_plus_meta    0.077549  0.000564     5          NaN       NaN     0
        early_sd           0.069319  0.000964     5     0.055772  0.000171     3
        id                 0.060314  0.000647     5     0.047851  0.000756     3
        id_plus_meta       0.069161  0.000312     5          NaN       NaN     0
        late_B             0.056557  0.001121     3     0.044808  0.001243     3
        late_BT            0.056797  0.000775     3     0.045095  0.001252     3
        late_C             0.057817  0.000804     3     0.046004  0.001147     3
        late_CB            0.057029  0.000916     3     0.045157  0.000933     3
        late_CBT           0.057420  0.000730     3     0.045170  0.000660     3
        late_CT            0.057756  0.000639     3     0.045582  0.001145     3
        late_T             0.057726  0.000742     3     0.045889  0.001296     3
        title_shuffled     0.059546  0.000262     3          NaN       NaN     0
sports  early_B            0.032821  0.000545     3     0.025476  0.000476     3
        early_BT           0.035036  0.000253     3     0.027100  0.000142     3
        early_C            0.034461  0.000589     3     0.026773  0.000353     3
        early_CB           0.036943  0.000548     3     0.028586  0.000336     3
        early_CBT          0.038070  0.000332     3     0.029247  0.000090     3
        early_CT           0.036118  0.000496     3     0.027918  0.000258     3
        early_T            0.036853  0.000406     3     0.029185  0.000724     3
        early_sd           0.036852  0.000420     3          NaN       NaN     0
        id                 0.034008  0.000253     3     0.026199  0.000198     3
        late_B             0.031975  0.000172     3     0.024210  0.000437     3
        late_BT            0.031495  0.000183     3     0.023656  0.000501     3
        late_C             0.032301  0.000221     3     0.024430  0.000556     3
        late_CB            0.031858  0.000512     3     0.023906  0.000472     3
        late_CBT           0.031894  0.000354     3     0.024047  0.000724     3
        late_CT            0.032297  0.000172     3     0.024265  0.000759     3
        late_T             0.032018  0.000227     3     0.024319  0.000537     3
toys    early_B            0.061682  0.000645     3     0.049971  0.000443     3
        early_BT           0.065916  0.000717     3     0.052713  0.001066     3
        early_C            0.060096  0.001144     3     0.046533  0.000531     3
        early_CB           0.067297  0.000721     3     0.054824  0.000864     3
        early_CBT          0.069009  0.000169     3     0.056341  0.000498     3
        early_CT           0.064062  0.000561     3     0.051494  0.000615     3
        early_T            0.067615  0.000387     3     0.055858  0.000236     3
        early_sd           0.068367  0.000310     3          NaN       NaN     0
        id                 0.061224  0.000627     3     0.049488  0.000679     3
        late_B             0.056983  0.000870     3     0.045384  0.000639     3
        late_BT            0.055994  0.000733     3     0.044643  0.000633     3
        late_C             0.056538  0.000435     3     0.045455  0.000607     3
        late_CB            0.055613  0.000449     3     0.044626  0.001002     3
        late_CBT           0.055607  0.000628     3     0.044708  0.000845     3
        late_CT            0.056168  0.000282     3     0.044976  0.000630     3
        late_T             0.057764  0.001041     3     0.046616  0.001400     3

Blank Test is NOT_OPENED, not zero. Gates are validation-only before freeze. Historical diagnostic Test values remain separate.

beauty/framing_decision.json

{
  "framing": "F2",
  "framing_family": [
    "all_fixed",
    "global_subset",
    "group_subset",
    "global_gate",
    "unsupervised",
    "softmax_null",
    "independent_ID",
    "early_no_sd",
    "id_plus_meta",
    "early_plus_meta"
  ],
  "best_baseline": "early_plus_meta",
  "comparison": {
    "method_a": "UASF",
    "method_b": "early_plus_meta",
    "metric_name": "NDCG@10",
    "mean_a": 0.065832,
    "mean_b": 0.077549,
    "mean_diff": -0.011717546678624234,
    "ci_lower": -0.013047396428077913,
    "ci_upper": -0.01039058478538526,
    "p_value": 0.0,
    "is_significant": true
  },
  "secondary_six_selectors": {
    "framing": "F2",
    "framing_family": [
      "all_fixed",
      "global_subset",
      "group_subset",
      "global_gate",
      "unsupervised",
      "softmax_null"
    ],
    "best_baseline": "group_subset",
    "comparison": {
      "method_a": "UASF",
      "method_b": "group_subset",
      "metric_name": "NDCG@10",
      "mean_a": 0.065832,
      "mean_b": 0.069446,
      "mean_diff": -0.0036146603510241286,
      "ci_lower": -0.0044298245669249494,
      "ci_upper": -0.0028202890574118558,
      "p_value": 0.0,
      "is_significant": true
    },
    "sensitivity_only": true,
    "used_for_primary_decision": false
  },
  "framing_policy": "memo0711_v1_locked_ten_v3",
  "validation_means": {
    "all_fixed": 0.06931918011715878,
    "global_subset": 0.06931918011715878,
    "group_subset": 0.06944641458503857,
    "global_gate": 0.0680158001730264,
    "unsupervised": 0.06839234525682909,
    "softmax_null": 0.06456980147159444,
    "independent_ID": 0.060314400530442284,
    "early_no_sd": 0.07005965230575771,
    "id_plus_meta": 0.06916138603515982,
    "early_plus_meta": 0.07754930091263867,
    "uasf": 0.06583175423401445
  },
  "seeds": [
    42,
    43,
    44,
    45,
    46
  ],
  "test_used": false,
  "rule": "mean UASF exceeds best predeclared locked baseline and paired-user CI lower > 0"
}