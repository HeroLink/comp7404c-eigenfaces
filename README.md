# COMP7404C - Final Project

Group 21

Members: CHEN Lu, TAN Tonghao, WANG Yanbo, YU Rentao

Email: {link, tanth, wyb00239, rtyu}@connect.hku.hk

Selected Paper (IEEE Style):

1. `M. A. Turk and A. P. Pentland, “Face recognition using eigenfaces,” in Proceedings. 1991 IEEE computer society
conference on computer vision and pattern recognition. IEEE Computer Society, 1991, pp. 586–587.`
2. `D. Xu, S. Yan, L. Zhang, S. Lin, H.-J. Zhang, and T. S. Huang, “Reconstruction and recognition of tensor-based
objects with concurrent subspaces analysis,” IEEE Transactions on Circuits and Systems for Video Technology,
vol. 18, no. 1, pp. 36–47, 2008.`

Dataset: [Labeled Faces in the Wild (LFW)](http://vis-www.cs.umass.edu/lfw/)

Report link: [`report.tex` - Overleaf](https://www.overleaf.com/1772296625kqfbqmhkhmrz)

Presentation slide link: [`slide.tex` - Overleaf](https://www.overleaf.com/1772296625kqfbqmhkhmrz)

# File Structure

```plaintext
├── Eigenfaces.pdf
├── README.md
├── data                            # Code will automatically download LFW dataset to this directory
├── jupyter
│   ├── dataset-analysis.ipynb      # Dataset analysis notebook
│   └── eigenfaces.ipynb            # Experiments Notebook
└── src
    ├── __pycache__
    ├── csa.py                      # Code of Concurrent Subspaces Analysis
    ├── lda.py                      # Code of Linear Discriminant Analysis
    ├── pca.py                      # Code of Principal Component Analysis
    ├── real_time.py                # Attempt for real-time recognition
    └── utils.py
```

<!-- ## Assignment

1. WANG Yanbo: Report, Presentation slide

2. CHEN Lu: Code (1-4), Report

3. YU Rentao: Code (5-7) -->
