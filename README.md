# KSR

This is our Theano implementation for the paper:

> Jin Huang, Wayne Xin Zhao, Hongjian Dou, Ji-Rong Wen and Edward Y.Chang(2018).
Improving Sequential Recommendation with Knowledge-Enhanced Memory Networks.
[paper](https://dl.acm.org/doi/10.1145/3209978.3210017). In SIGIR'2018.

This code is based on [GRU4Rec](https://github.com/hidasib/GRU4Rec).

A PyTorch re-implementation of KSR is integrated into Recbole [here](https://recbole.io/docs/user_guide/model/sequential/ksr.html).

## Requirements

- Python 2.7
- Theano 0.9.0

For convenient, you can use virtualenv together with the
[requirement](https://github.com/mquad/hgru4rec/blob/master/requirements.txt)
to set up a virtual environment before running the code.

## Acknowledgement
Any scientific publications that use our codes and datasets should cite the
following paper as the reference:

```
@inproceedings{KSR-SIGIR-2018,
  author    = {Jin Huang and
               Wayne Xin Zhao and
               Hongjian Dou and
               Ji{-}Rong Wen and
               Edward Y. Chang},
  title     = {Improving Sequential Recommendation with Knowledge-Enhanced Memory
               Networks},
  booktitle = {The 41st International {ACM} {SIGIR} Conference on Research {\&}
               Development in Information Retrieval, {SIGIR} 2018, Ann Arbor, MI,
               USA, July 08-12, 2018},
  pages     = {505--514},
  year      = {2018},
}
```

## Contact us
If you have any question, please contact Jin Huang
(betsyj.huang@gmail.com & note [question for KSR] in mail title).
