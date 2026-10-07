# Smock, Pesala, Abraham - 2022 - GriTS Grid table similarity metric for table structure recognition

- Source HTML: `raw/html/Smock, Pesala, Abraham - 2022 - GriTS Grid table similarity metric for table structure recognition.html`
- Source SHA256: `3afe8c484075f6c17c63bc8f088928dd05f2792da7243ca222c1f52c563a26ce`
- Source URL: https://ar5iv.labs.arxiv.org/html/2203.12555
- Generated from: `scripts/fetch_web_text.py`
- Extraction: `structured-html-v2` (headings, links, MathML/TeX and tables; figures require visual review)

## Extracted Text

<a id="source-section-0"></a>

11institutetext: Microsoft, Redmond WA, USA

11email: {brsmock,ropesala,robin.abraham}@microsoft.com


<a id="source-section-1"></a>

# GriTS: Grid table similarity metric for table structure recognition


Brandon Smock
11
[0009-0002-7002-0800](https://orcid.org/0009-0002-7002-0800)


Rohith Pesala
11
[0009-0004-7373-853X](https://orcid.org/0009-0004-7373-853X)


Robin Abraham
11
[0000-0003-1915-8118](https://orcid.org/0000-0003-1915-8118)


<a id="source-section-2"></a>

###### Abstract


In this paper, we propose a new class of metric for table structure recognition (TSR) evaluation, called grid table similarity (GriTS).
Unlike prior metrics, GriTS evaluates the correctness of a predicted table directly in its natural form as a matrix.
To create a similarity measure between matrices, we generalize the two-dimensional largest common substructure (2D-LCS) problem, which is NP-hard, to the 2D most similar substructures (2D-MSS) problem and propose a polynomial-time heuristic for solving it.
This algorithm produces both an upper and a lower bound on the true similarity between matrices.
We show using evaluation on a large real-world dataset that in practice there is almost no difference between these bounds.
We compare GriTS to other metrics and empirically validate that matrix similarity exhibits more desirable behavior than alternatives for TSR performance evaluation.
Finally, GriTS unifies all three subtasks of cell topology recognition, cell location recognition, and cell content recognition within the same framework, which simplifies the evaluation and enables more meaningful comparisons across different types of TSR approaches.
Code will be released at [https://github.com/microsoft/table-transformer](https://github.com/microsoft/table-transformer).


<a id="source-section-3"></a>

## 1 Introduction


Table extraction (TE) [[14](https://ar5iv.labs.arxiv.org/html/2203.12555#bib.bib14), [17](https://ar5iv.labs.arxiv.org/html/2203.12555#bib.bib17), [3](https://ar5iv.labs.arxiv.org/html/2203.12555#bib.bib3)] is the problem of inferring the presence, structure, and—to some extent—meaning of tables in documents or other unstructured presentations.
In its presented form, a table is typically expressed as a collection of cells organized over a two-dimensional grid [[16](https://ar5iv.labs.arxiv.org/html/2203.12555#bib.bib16), [5](https://ar5iv.labs.arxiv.org/html/2203.12555#bib.bib5), [13](https://ar5iv.labs.arxiv.org/html/2203.12555#bib.bib13)].
Table structure recognition (TSR) [[6](https://ar5iv.labs.arxiv.org/html/2203.12555#bib.bib6), [7](https://ar5iv.labs.arxiv.org/html/2203.12555#bib.bib7)] is the subtask of TE concerned with inferring this two-dimensional cellular structure from a table’s unstructured presentation.


While straightforward to describe, formalizing the TSR task in a way that enables effective performance evaluation has proven challenging [[9](https://ar5iv.labs.arxiv.org/html/2203.12555#bib.bib9)].
Perhaps the most straightforward way to measure performance is to compare the sets of predicted and ground truth cells for each table and measure the percentage of tables for which these sets match exactly—meaning, for each predicted cell there is a matching ground truth cell, and vice versa, that has the same rows, columns, and text content.
However, historically this metric has been eschewed in favor of measures of partial correctness that score each table’s correctness on a range of $[0,1]$ rather than as binary correct or incorrect.
Measures of partial correctness are useful not only because they are more granular, but also because they are less impacted by errors and ambiguities in the ground truth.
This is important, as creating unambiguous ground truth for TSR is a challenging problem, which can introduce noise not only into the learning task but also performance evaluation [[15](https://ar5iv.labs.arxiv.org/html/2203.12555#bib.bib15), [10](https://ar5iv.labs.arxiv.org/html/2203.12555#bib.bib10)].


Figure 1: Grid table similarity (GriTS) is based on computing the two-dimensional most similar substructures (2D-MSS) between two matrices (see [Sec. 3](https://ar5iv.labs.arxiv.org/html/2203.12555#S3) for details). In this example, the 2D-MSS of matrices $\mathbf{A}$ and $\mathbf{B}$ are substructures $\mathbf{\hat{A}}$ and $\mathbf{\hat{B}}$. $\mathbf{\hat{A}}$ and $\mathbf{\hat{B}}$ create a direct correspondence (or alignment) between entries in the original matrices, depicted here as darker shading for one pair of corresponding entries.


Designing a metric for partial correctness of tables has also proven challenging.
The naive approach of comparing predicted cells with ground truth cells by their absolute positions suffers from the problem that a single mistake in cell segmentation can offset all subsequent cells by one position, which may result in a disproportionate penalty.
Several metrics have been proposed that instead consider the relative positions of predicted cells [[6](https://ar5iv.labs.arxiv.org/html/2203.12555#bib.bib6), [4](https://ar5iv.labs.arxiv.org/html/2203.12555#bib.bib4), [18](https://ar5iv.labs.arxiv.org/html/2203.12555#bib.bib18), [12](https://ar5iv.labs.arxiv.org/html/2203.12555#bib.bib12)].
However, these metrics capture relative position in different ways that do not fully account for a table’s global two-dimensional (2D) structure.
Metrics also offer differing perspectives on what constitutes the task to be measured, what property of a predicted cell is evaluated, and whether predicted cells can be partially correct.


To address these issues, in this paper we develop a new class of metric for TSR called grid table similarity (GriTS).
GriTS attempts to unify the different perspectives on TSR evaluation and address these in a modular way, both to simplify the evaluation and to make comparisons between approaches more meaningful.
Among our contributions:


- •


GriTS is the first metric to evaluate tables directly in their matrix form, maintaining the global 2D relationships between cells when comparing predictions to ground truth.


- •


To create a similarity between matrices, we extend the 2D largest common substructure (2D-LCS) problem, which is NP-hard, to 2D most similar substructures (2D-MSS) and propose a polynomial-time heuristic to solve it. This algorithm produces both an upper and lower bound on its approximation, which we show have little difference in practice.


- •


We outline the properties of an ideal TSR metric and validate empirically on a large real-world dataset that GriTS exhibits more ideal behavior than alternatives for TSR evaluation.


- •


GriTS is the first metric that addresses cell topology, cell content, and cell location recognition in a unified manner. This makes it easier to interpret and compare results across different modeling approaches and datasets.


<a id="source-section-4"></a>

## 2 Related Work


Table 1: Comparison of metrics proposed for table structure recognition.


| 列 1 | 列 2 | 列 3 | 列 4 | 列 5 |
| --- | --- | --- | --- | --- |
| Name | Task/Cell Property | Data Structure | Cell Partial Correctness | Form |
| $\textrm{DAR}_{\textrm{Con}}$ [[6](https://ar5iv.labs.arxiv.org/html/2203.12555#bib.bib6)] | Content | Set of adjacency relations | Exact match | F-score |
| $\textrm{DAR}_{\textrm{Loc}}$ [[4](https://ar5iv.labs.arxiv.org/html/2203.12555#bib.bib4)] | Location | Set of adjacency relations | Avg. at multiple IoU thresholds | F-score |
| BLEU-4 [[12](https://ar5iv.labs.arxiv.org/html/2203.12555#bib.bib12)] | Topology & function | Sequence of HTML tokens | Exact match | BLEU-4 |
| TEDS [[18](https://ar5iv.labs.arxiv.org/html/2203.12555#bib.bib18)] | Content & function | Tree of HTML tags | Normalized Levenshtein similarity | TEDS |
| $\textbf{GriTS}_{\textbf{Top}}$ | Topology | Matrix of cells | IoU | F-score |
| $\textbf{GriTS}_{\textbf{Con}}$ | Content | Matrix of cells | Normalized LCS | F-score |
| $\textbf{GriTS}_{\textbf{Loc}}$ | Location | Matrix of cells | IoU | F-score |


A number of metrics exist for evaluating table structure recognition methods.
These include the cell adjacency-based metric used in the ICDAR 2013 Table Competition, which we refer to as directed adjacency relations (DAR) [[6](https://ar5iv.labs.arxiv.org/html/2203.12555#bib.bib6), [4](https://ar5iv.labs.arxiv.org/html/2203.12555#bib.bib4)], 4-gram BLEU score (BLEU-4) [[12](https://ar5iv.labs.arxiv.org/html/2203.12555#bib.bib12)], and tree edit distance similarity (TEDS) [[18](https://ar5iv.labs.arxiv.org/html/2203.12555#bib.bib18)].
In [Tab. 1](https://ar5iv.labs.arxiv.org/html/2203.12555#S2.T1) we categorize these metrics across four dimensions: subtask (cell property), data structure, cell partial correctness, and overall score formulation.


<a id="source-section-5"></a>

### 2.1 Subtask/property


Each metric typically poses the table structure recognition task more specifically as one of the following:


- 1.


Cell topology recognition considers the layout of the cells, specifically the rows and columns each cell occupies over a two-dimensional grid.


- 2.


Cell content recognition considers the layout of cells and the text content of each cell.


- 3.


Cell location recognition considers the layout of cells and the absolute coordinates of each cell within a document.


One way to characterize these subtasks is in terms of the property of the cell that is considered most central to the recognition task.
For cell topology, this can be considered the colspan and rowspan of each cell.
Each perspective is useful.
Cell content recognition is most aligned with the end goal of TE, but for table image input it can depend on the quality of optical character recognition (OCR).
Cell location recognition does not depend on OCR, but not every TSR method reports cell locations.
Cell topology recognition is independent of OCR and is applicable to all TSR methods, but is not anchored to the actual content of the cells by either text or location.
Thus, accurate cell topology recognition is necessary but not sufficient for successful TSR.


Functional analysis is the subtask of table extraction concerned with determining whether each cell is a key (header) or value.
While it is usually considered separate from structure recognition, metrics sometimes evaluate aspects of TSR and functional analysis jointly, such as TEDS and BLEU-4.


<a id="source-section-6"></a>

### 2.2 Data structure


A table is presented in two dimensions, giving it a natural representation as a grid or matrix of cells.
The objective of TSR is usually considered to be inferring this grid structure.
However, for comparing predictions with ground truth, prior metrics have used alternative abstract representations for tables.
These possibly decompose a table into sub-units other than cells and represent them in an alternate structure with relationships between elements that differ from those of a matrix.
The metrics proposed by Göbel et al. [[6](https://ar5iv.labs.arxiv.org/html/2203.12555#bib.bib6)] and Gao et al. [[4](https://ar5iv.labs.arxiv.org/html/2203.12555#bib.bib4)] deconstruct the grid structure of a table into a set of directed adjacency relations, corresponding to pairs of neighboring cells.
We refer to the first metric, which evaluates cell text content, as $\text{DAR}_{\text{Con}}$, and the second, which evaluates cell location, as $\text{DAR}_{\text{Loc}}$.
Li et al. [[12](https://ar5iv.labs.arxiv.org/html/2203.12555#bib.bib12)] represent a table as a token sequence, using a simplified HTML encoding.
Zhong et al. [[18](https://ar5iv.labs.arxiv.org/html/2203.12555#bib.bib18)] also represent a table using HTML tokens but use the nesting of the tags to consider a table’s cells to be tree-structured, which more closely represents a table’s two-dimensional structure than does a sequence.


Metrics based on these different representations each have their own sensitivities and invariances to changes to a table’s structure.
Zhong et al. [[18](https://ar5iv.labs.arxiv.org/html/2203.12555#bib.bib18)] investigate a few of these sensitivities and demonstrate that DAR mostly ignores the insertion of contiguous blank cells into a table, while TEDS does not.
However, largely the sensitivities of these metrics that result from their different representations have not been studied.
This makes it more challenging to interpret and compare them.


<a id="source-section-7"></a>

### 2.3 Cell partial correctness


Each metric produces a score between 0 and 1 for each table.
For some metrics this takes into account simply the fraction of matching sub-units between a prediction and ground truth.
Some metrics also define a measure of correctness for each sub-unit, which is between 0 and 1.
For instance, TEDS incorporates the normalized Levenshtein similarity to allow the text content of an HTML tag to partially match the ground truth.
Defining partial correctness at the cell level is useful because it is less sensitive to minor discrepancies between a prediction and ground truth that may have little or no impact on table extraction quality.


<a id="source-section-8"></a>

### 2.4 Form


The form of a metric is the way in which the match between prediction and ground truth is aggregated over matching and non-matching sub-units.
DAR uses both precision and recall, which can be taken together to produce the standard F-score.
BLEU-4 treats the output of a table structure recognition model as a sequence and uses the 4-gram BLEU score to measure the degree of match between a predicted and ground truth sequence.
TEDS computes a modified tree edit distance, which is the cost of transforming partially-matching and non-matching sub-units between the tree representations of a predicted and ground truth table.


<a id="source-section-9"></a>

### 2.5 Spreadsheet diffing


A related problem to the one explored in this paper is the identification of differences or changes between two versions of a spreadsheet [[2](https://ar5iv.labs.arxiv.org/html/2203.12555#bib.bib2), [8](https://ar5iv.labs.arxiv.org/html/2203.12555#bib.bib8)].
In this case the goal is to classify cells between the two versions as either modified or unmodified, and possibly to generate a sequence of edit transformations that would convert one version of a spreadsheet into another.


<a id="source-section-10"></a>

## 3 Grid Table Similarity


To motivate the metrics proposed in this paper, we first introduce the following attributes that we believe an ideal metric for table structure recognition should exhibit:


- 1.


Task isolation: the table structure recognition task is measured in isolation from other table extraction tasks (detection and functional analysis).


- 2.


Cell isolation: a true positive according to the metric corresponds to exactly one predicted cell and one ground truth cell.


- 3.


Two-dimensional order preservation [[1](https://ar5iv.labs.arxiv.org/html/2203.12555#bib.bib1)]: For any two true positive cells, $\textrm{tp}_{1}$ and $\textrm{tp}_{2}$, the relative order in which they appear is the same in both dimensions in the predicted and ground truth tables. More specifically:


- (a)


The maximum true row of $\textrm{tp}_{1}$ < minimum true row of $\textrm{tp}_{2}$ $\iff$ the maximum predicted row of $\textrm{tp}_{1}$ < minimum predicted row of $\textrm{tp}_{2}$.


- (b)


The maximum true column of $\textrm{tp}_{1}$ < minimum true column of $\textrm{tp}_{2}$ $\iff$ the maximum predicted column of $\textrm{tp}_{1}$ < minimum predicted column of $\textrm{tp}_{2}$.


- (c)


The maximum true row of $\textrm{tp}_{1}$ = minimum true row of $\textrm{tp}_{2}$ $\iff$ the maximum predicted row of $\textrm{tp}_{1}$ = minimum predicted row of $\textrm{tp}_{2}$.


- (d)


The maximum true column of $\textrm{tp}_{1}$ = minimum true column of $\textrm{tp}_{2}$ $\iff$ the maximum predicted column of $\textrm{tp}_{1}$ = minimum predicted column of $\textrm{tp}_{2}$.


- 4.


Row and column equivalence: the metric is invariant to transposing the rows and columns of both a prediction and ground truth (i.e. rows and columns are of equal importance).


- 5.


Cell position invariance: the credit given to a correctly predicted cell is the same regardless of its absolute row-column position.


The first two attributes are considered in [Tab. 1](https://ar5iv.labs.arxiv.org/html/2203.12555#S2.T1).
In [Sec. 4](https://ar5iv.labs.arxiv.org/html/2203.12555#S4), we test the last two properties for different proposed metrics.
While not essential, we note again that in practice we believe it is also useful for a TSR metric to define partial correctness for cells and to have the same general form for both cell content recognition and cell location recognition.
In the remainder of this section we describe a new class of evaluation metric that meets all of the above criteria.


<a id="source-section-11"></a>

### 3.1 2D-LCS


We first note that Property [3](https://ar5iv.labs.arxiv.org/html/2203.12555#S3.I1.i3) is difficult to enforce for cells that can span multiple rows and columns.
To account for this, we instead consider the matrix of grid cells of a table.
Exactly one grid cell occupies the intersection of each row and each column of a table.
Note that as a spanning cell occupies multiple grid cells, its text content logically repeats at every grid cell location that the cell spans.


To enforce Property [3](https://ar5iv.labs.arxiv.org/html/2203.12555#S3.I1.i3) for grid cells, we consider the generalization of the longest common subsequence (LCS) problem to two dimensions, which is called the two-dimensional largest (or longest) common substructure (2D-LCS) problem [[1](https://ar5iv.labs.arxiv.org/html/2203.12555#bib.bib1)].
Let $\mathbf{M}[R,C]$ be a matrix with $R=[r_{1},\dots,r_{m}]$ representing its rows and $C=[c_{1},\dots,c_{n}]$ representing its columns.
Let $R^{\prime}\mid R$ be a subsequence of rows of $R$, and $C^{\prime}\mid C$ be a subsequence of columns of $C$.
Then a substructure $\mathbf{M^{\prime}}\mid\mathbf{M}$ is such that,


| 列 1 | 列 2 | 列 3 |
| --- | --- | --- |
| | $$<br>\mathbf{M^{\prime}}\mid\mathbf{M}=\mathbf{M}[R^{\prime},C^{\prime}].<br>$$ | |


2D-LCS operates on two matrices, $\mathbf{A}$ and $\mathbf{B}$, and determines the largest two-dimensional substructures, $\mathbf{\hat{A}}\mid\mathbf{A}=\mathbf{\hat{B}}\mid\mathbf{B}$, the two have in common.
In other words,


| 列 1 | 列 2 | 列 3 | 列 4 | 列 5 |
| --- | --- | --- | --- | --- |
| | $\displaystyle\textrm{2D-LCS}(\mathbf{A},\mathbf{B})$ | $\displaystyle=\operatorname*{arg\,max}_{\mathbf{A}^{\prime}\mid\mathbf{A},\mathbf{B}^{\prime}\mid\mathbf{B}}{\sum_{i,j}f(\mathbf{A}^{\prime}_{i,j},\mathbf{B}^{\prime}_{i,j})}$ | | (1) |
| | | $\displaystyle=\mathbf{\hat{A}},\mathbf{\hat{B}},$ | | (2) |


where,


| 列 1 | 列 2 | 列 3 |
| --- | --- | --- |
| | $$<br>f(e_{1},e_{2})=\begin{cases}1,&\text{if }e_{1}=e_{2}\\<br>0,&\text{otherwise}\end{cases}.<br>$$ | |


<a id="source-section-12"></a>

### 3.2 2D-MSS


While a solution to the 2D-LCS problem satisfies Property [3](https://ar5iv.labs.arxiv.org/html/2203.12555#S3.I1.i3) for grid cells, it assumes an exact match between matrix elements.
To let cells partially match, an extension to 2D-LCS is to relax the exact match constraint and instead determine the two most similar two-dimensional substructures, $\mathbf{\tilde{A}}$ and $\mathbf{\tilde{B}}$.
We define this by replacing equality between two entries $\mathbf{A}_{i,j}$ and $\mathbf{B}_{k,l}$ with a more general choice of similarity function between them. In other words,


| 列 1 | 列 2 | 列 3 | 列 4 | 列 5 |
| --- | --- | --- | --- | --- |
| | $\displaystyle\textrm{2D-MSS}_{f}(\mathbf{A},\mathbf{B})$ | $\displaystyle=\operatorname*{arg\,max}_{\mathbf{A}^{\prime}\mid\mathbf{A},\mathbf{B}^{\prime}\mid\mathbf{B}}{\sum_{i,j}f(\mathbf{A}^{\prime}_{i,j},\mathbf{B}^{\prime}_{i,j})}$ | | (3) |
| | | $\displaystyle=\mathbf{\tilde{A}},\mathbf{\tilde{B}},$ | | (4) |


where,


| 列 1 | 列 2 | 列 3 |
| --- | --- | --- |
| | $$<br>0\leq f(e_{1},e_{2})\leq 1\qquad\forall e_{1},e_{2}.<br>$$ | |


Taking inspiration from the standard F-score, we define a general similarity measure between two matrices based on this as,


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $\displaystyle\tilde{S}_{f}(\mathbf{A},\mathbf{B})=\frac{2\sum_{i,j}f(\mathbf{\tilde{A}}_{i,j},\mathbf{\tilde{B}}_{i,j})}{{\|\mathbf{A}\|}+{\|\mathbf{B}\|}},$ | | (5) |


where $|\mathbf{M}_{m\times n}|=m\cdot n$.


<a id="source-section-13"></a>

### 3.3 Grid table similarity (GriTS)


Finally, to define a similarity between tables, we use [Eq. 5](https://ar5iv.labs.arxiv.org/html/2203.12555#S3.E5) with a particular choice of similarity function and a particular matrix of entries to compare.
This has the general form,


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $\displaystyle\text{GriTS}_{f}(\mathbf{A},\mathbf{B})=\frac{2\sum_{i,j}f(\mathbf{\tilde{A}}_{i,j},\mathbf{\tilde{B}}_{i,j})}{{\|\mathbf{A}\|}+{\|\mathbf{B}\|}},$ | | (6) |


where $\mathbf{A}$ and $\mathbf{B}$ now represent tables—matrices of grid cells—and $f$ is a similarity function between the grid cells’ properties.
Interpreting [Eq. 6](https://ar5iv.labs.arxiv.org/html/2203.12555#S3.E6) as an F-score, then letting $\mathbf{A}$ be the ground truth matrix and $\mathbf{B}$ be the predicted matrix, we can also define the following quantities, which we interpret as recall and precision: $\text{GriTS-Rec}_{f}(\mathbf{A},\mathbf{B})=\frac{\sum_{i,j}f(\mathbf{\tilde{A}}_{i,j},\mathbf{\tilde{B}}_{i,j})}{|\mathbf{A}|}$ and $\text{GriTS-Prec}_{f}(\mathbf{A},\mathbf{B})=\frac{\sum_{i,j}f(\mathbf{\tilde{A}}_{i,j},\mathbf{\tilde{B}}_{i,j})}{|\mathbf{B}|}$.


A specific choice of grid cell property and the similarity function between them defines a particular GriTS metric.
We define three of these: $\text{GriTS}_{\text{Top}}$ for cell topology recognition, $\text{GriTS}_{\text{Con}}$ for cell text content recognition, and $\text{GriTS}_{\text{Loc}}$ for cell location recognition.
Each evaluates table structure recognition from a different perspective.


[图片：Refer to caption]


(a) An example presentation table from the PubTables-1M dataset.


(b) $\text{GriTS}_{\text{Con}}$


(c) $\text{GriTS}_{\text{Top}}$


(d) $\text{GriTS}_{\text{Loc}}$


Figure 2: An example presentation table from the PubTables-1M dataset, along with corresponding ground truth grid cell matrices for different GriTS metrics. Each matrix entry corresponds to one grid cell. Entries that correspond to spanning cells are shaded darker for illustrative purposes.


The matrices used for each metric are visualized in [Fig. 2](https://ar5iv.labs.arxiv.org/html/2203.12555#S3.F2).
For cell location, $\mathbf{A}_{i,j}$ contains the bounding box of the cell at row $i$, column $j$, and we use IoU to compute similarity between bounding boxes.
For cell text content, $\mathbf{A}_{i,j}$ contains the text content of the cell at row $i$, column $j$, and we use normalized longest common subsequence (LCS) to compute similarity between text sequences.


For cell topology, we use the same similarity function as cell location but on bounding boxes with size and relative position given in the grid cell coordinate system.
For the cell at row $i$, column $j$, let $\alpha_{i,j}$ be its rowspan, let $\beta_{i,j}$ be its colspan, let $\rho_{i,j}$ be the minimum row it occupies, and let $\theta_{i,j}$ be the minimum column it occupies.
Then for cell topology recognition, $\mathbf{A}_{i,j}$ contains the bounding box $[\theta_{i,j}-j,\rho_{i,j}-i,\theta_{i,j}-j+\beta_{i,j},\rho_{i,j}-i+\alpha_{i,j}]$.
Note that for any cell with rowspan of 1 and colspan of 1, this box is $[0,0,1,1]$.


<a id="source-section-14"></a>

### 3.4 Factored 2D-MSS algorithm


Computing the 2D-LCS of two matrices is NP-hard [[1](https://ar5iv.labs.arxiv.org/html/2203.12555#bib.bib1)].
This suggests that all metrics for TSR may necessarily be an approximation to what could be considered the ideal metric.
We propose a heuristic approach to determine the 2D-MSS by factoring the problem.
Instead of determining the optimal subsequences of rows and columns jointly for each matrix, we determine the optimal subsequences of rows and the optimal subsequences of columns independently.
This uses dynamic programming (DP) in a nested manner, which is run twice: once to determine the optimal subsequences of rows and a second time to determine the optimal subsequences of columns.
For the case of rows, an inner DP operates on sequences of grid cells in a row, computing the best possible sequence alignment of cells between any two rows.
The inner DP is executed over all pairs of predicted and ground truth rows to score how well each predicted row can be aligned with each ground truth row.
An outer DP operates on the sequences of rows from each matrix, using the pairwise scores computed by the inner DP to determine the best alignment of subsequences of rows between the predicted and ground truth matrices.
For the case of columns, the procedure is identical, merely substituting columns for rows.
The nested DP procedure is $O(|\mathbf{A}|\cdot|\mathbf{B}|)$.
Our implementation uses extensive memoization [[11](https://ar5iv.labs.arxiv.org/html/2203.12555#bib.bib11)] to maximize the efficiency of the procedure.


This factored procedure is similar to the RowColAlign algorithm [[8](https://ar5iv.labs.arxiv.org/html/2203.12555#bib.bib8)] proposed for spreadsheet diffing.
Both procedures decouple the optimization of rows and columns and use DP in a nested manner.
However, RowColAlign attempts to optimize the number of pair-wise exact matches between substructures, whereas Factored 2D-MSS attempts to optimize the pair-wise similarity between substructures.
These differing objectives result in the two procedures having differing outcomes given the same two input matrices.


<a id="source-section-15"></a>

### 3.5 Approximation bounds


The outcome of the procedure is a valid 2D substructure of each matrix—these just may not be the most similar substructures possible, given that the rows and columns are optimized separately.
However, given that these are valid substructures, it follows that the similarity between matrices $\mathbf{A}$ and $\mathbf{B}$ computed by this procedure is a lower bound on their true similarity.
It similarly follows that because constraints are relaxed during the optimization procedure, the lowest similarity determined when computing the optimal subsequences of rows and the optimal subsequences of columns serves as an upper bound on the true similarity between matrices.
We define GriTS as the value of the lower bound, as it always corresponds to a valid substructure.
However, the upper bound score can also be reported to indicate if there is any uncertainty in the true value.
As we show in [Sec. 4](https://ar5iv.labs.arxiv.org/html/2203.12555#S4), little difference is observed between these bounds in practice .


<a id="source-section-16"></a>

## 4 Experiments


Table 2: In this experiment we evaluate models of varying strengths on the PubTables-1M test set and compare the upper and lower bounds for $\textrm{GriTS}_{\textrm{Con}}$ that are produced. This shows how closely GriTS approximates the true similarity between two tables using the Factored 2D-MSS algorithm.


| Epoch | $\textbf{GriTS}_{\textbf{Con}}$ | Upper Bound | Difference | Equal Instances (%) |
| --- | --- | --- | --- | --- |
| 1 | 0.8795 | 0.8801 | 0.0005 | 81.2% |
| 2 | 0.9347 | 0.9348 | 0.0002 | 87.6% |
| 3 | 0.9531 | 0.9532 | 0.0001 | 91.1% |
| 4 | 0.9640 | 0.9641 | < 0.0001 | 93.0% |
| 5 | 0.9683 | 0.9683 | < 0.0001 | 93.2% |
| 10 | 0.9794 | 0.9795 | < 0.0001 | 96.1% |
| 15 | 0.9829 | 0.9829 | < 0.0001 | 96.9% |
| 20 | 0.9850 | 0.9850 | < 0.0001 | 97.4% |


In this section, we report several experiments that assess and compare GriTS and other metrics for TSR.
Given that due to computational intractability no algorithm perfectly implements the ideal metric for TSR as outlined in [Sec. 3](https://ar5iv.labs.arxiv.org/html/2203.12555#S3), the main goal of these experiments is to assess how well each proposed metric matches the behavior that we would expect in the theoretically optimal metric.


GriTS computes a similarity between two tables using the Factored 2D-MSS algorithm, which produces both an upper and a lower bound on the true similarity.
The difference between the two values represents the uncertainty, or maximum possible error, there is with respect to the true similarity.
In the first experiment, we measure how well GriTS approximates the true similarity between predicted and ground truth tables in practice.
To do this, we train the Table Transformer (TATR) model [[15](https://ar5iv.labs.arxiv.org/html/2203.12555#bib.bib15)] on the PubTables-1M dataset, which contains nearly one million tables, for 20 epochs and save the model produced after each epoch.
This effectively creates 20 different TSR models of varying strengths with which to evaluate predictions.
We evaluate each model on the entire PubTables-1M test set and measure the difference between $\textrm{GriTS}_{\textrm{Con}}$ and $\textrm{GriTS}_{\textrm{Con}}$ upper bound, as well as the percentage of individual instances for which these bounds are equal.


We present the results of this experiment in [Tab. 2](https://ar5iv.labs.arxiv.org/html/2203.12555#S4.T2).
As can be seen, there is very little difference between the upper and lower bounds across all models.
The uncertainty in the true table similarity peaks at the worst-performing model, which is trained for only one epoch.
For this model, $\textrm{GriTS}_{\textrm{Con}}$ is 0.8795 and the measured uncertainty in the true similarity is 0.0005.
Above a certain level of model performance, the average difference between the bounds and the percentage of instances for which the bounds differ both decrease quickly as performance improves, with the difference approaching 0 as GriTS approaches a score of 1.
By epoch 4, the difference between the bounds is already less than 0.0001, and for at least 93% of instances tested the GriTS score is in fact the true similarity.
These results strongly suggest that in practice there is almost no difference between GriTS and the true similarity between tables.


(a) $\text{DAR}_{\text{Con}}$


(b) $\text{TEDS}_{\text{Con}}$


(c) $\text{GriTS}_{\text{Con}}$


Figure 3: In this experiment, we compare the response of each metric to a predicted table with half of its rows missing (blue) or half of its columns missing (red) under different schemes for creating missing rows and columns. Results are averaged over 44,381 tables from the test set of PubTables-1M.


(a)


(b)


Figure 4: In this experiment, we compare the response of each metric to a predicted table with random rows missing ([Fig. 4(a)](https://ar5iv.labs.arxiv.org/html/2203.12555#S4.F4.sf1)) or random columns missing ([Fig. 4(b)](https://ar5iv.labs.arxiv.org/html/2203.12555#S4.F4.sf2)) as we vary the probability that a row or column is missing. Results are averaged over 44,381 tables from the test set of PubTables-1M.


Figure 5: In this experiment, we compare the response of each metric split into precision and recall to a predicted table with random rows missing or random columns missing as we vary the probability that a row or column is missing. Results are averaged over 44,381 tables from the test set of PubTables-1M.


In the next set of experiments, we compare GriTS and other metrics for TSR with respect to the properties outlined in [Sec. 3](https://ar5iv.labs.arxiv.org/html/2203.12555#S3).
The goal of these experiments is to assess how well each metric matches the behavior that we would expect in the theoretically optimal metric.
We evaluate each metric on the original ground truth (GT) and versions of the ground truth that are modified or corrupted in controlled ways, which shows the sensitivity or insensitivity of each metric to different underlying properties.
To create corrupted versions of the GT, we select either a subset of the GT table’s rows or a subset of its columns, where each row or each column from the GT is selected with probability $\textrm{p}(x)$, preserving their original order and discarding the rest.


For the experiments we use tables from the test set of the PubTables-1M dataset, which provides text content and location information for every cell, including blank cells.
To make sure each remaining grid cell has well-defined content and location after removing a subset of rows and columns from a table, we use the 44,381 tables that do not have any spanning cells.
In order to make the metrics more comparable, we define a version of TEDS that removes functional analysis from the evaluation, called $\text{TEDS}_{\text{Con}}$, by removing all header information in the ground truth.


In the first experiment, the goal is to test for each metric if rows and columns are given equal importance and if every cell is credited equally regardless of its absolute position.
To test these, we create missing columns or missing rows in a predicted table according to three different selection schemes.
In the first part of the experiment we select each row with probability 0.5 using the following three different selection schemes:


- •


First: select the first 50% of rows.


- •


Alternating: select either every odd-numbered row or every even-numbered row.


- •


Random: select 50% of rows within a table at random.


In the second part of the experiment, we select columns using the same three schemes as for rows.
In all six cases, exactly half of the cells are missing in a predicted table whenever the table has an even number of rows and columns.


We compare the impact of each selection scheme on the metrics in [Fig. 3](https://ar5iv.labs.arxiv.org/html/2203.12555#S4.F3).
For a metric to give equal credit to rows and columns, it should be insensitive to (produce the same value) whether half the rows are missing or half the columns are missing.
For a metric to give equal credit to each cell regardless of absolute position, it should be insensitive to which row or column the missing cell occurs in.
The results show that $\text{DAR}_{\text{Con}}$ is sensitive both to whether rows or columns are selected and to which rows or columns are selected.
$\text{TEDS}_{\text{Con}}$ is sensitive to whether rows are selected or columns are selected, but is not sensitive to which rows or columns are selected.
On the other hand, $\text{GriTS}_{\text{Con}}$ produces a nearly identical value no matter which scheme is used to select half of the rows or half of the columns.


In the second experiment, we select rows and columns randomly, but vary the probability $\text{p}(x)$ from 0 to 1.
We also expand the results to include all three GriTS metrics.
We show the results of this experiment in [Fig. 4](https://ar5iv.labs.arxiv.org/html/2203.12555#S4.F4).
Like in the first experiment, $\text{DAR}_{\text{Con}}$ produces a similar value when randomly selecting rows or randomly selecting columns, and we see that this holds for all values of $\text{p}(x)$.
Likewise, this is true not just for $\text{GriTS}_{\text{Con}}$ but all GriTS metrics.
On the other hand, for $\text{TEDS}_{\text{Con}}$, we see that the metric has a different sensitivity to randomly missing columns than to randomly missing rows, and that the relative magnitude of this sensitivity varies as we vary $\text{p}(x)$.


In [Fig. 5](https://ar5iv.labs.arxiv.org/html/2203.12555#S4.F5), we further split the results for DAR and GriTS by their precision and recall values.
Here we see that for GriTS, not only are rows and columns equivalent, but recall and precision match the probability of rows and columns being in the prediction and ground truth, respectively.
On the other hand, DAR has a less clear interpretation in terms of precision and recall.
Further, for DAR we notice that there is a slight sensitivity that shows up to the choice of rows versus columns for precision, which was not noticeable when considering F-score.
Overall these results show that GriTS closely resembles the ideal metric for TSR and exhibits more desirable behavior than prior metrics for this task.


<a id="source-section-17"></a>

## 5 Conclusion


In this paper we introduced grid table similarity (GriTS), a new class of evaluation metric for table structure recognition (TSR).
GriTS unifies all three perspectives of the TSR task within a single class of metric and evaluates model predictions in a table’s natural matrix form.
As the foundation for GriTS, we derived a similarity measure between matrices by generalizing the two-dimensional largest common substructure problem, which is NP-hard, to 2D most-similar substructures (2D-MSS).
We proposed a polynomial-time heuristic that produces both an upper and a lower bound on the true similarity and we showed that in practice these bounds are tight, nearly always guaranteeing the optimal solution.
We compared GriTS to other metrics and demonstrated using a large dataset that GriTS exhibits more desirable behavior for table structure recognition evaluation.
Overall, we believe these contributions improve the interpretability and stability of evaluation for table structure recognition and make it easier to compare results across different types of modeling approaches.


<a id="source-section-18"></a>

## 6 Acknowledgements


We would like to thank Pramod Sharma, Natalia Larios Delgado, Joseph N. Wilson, Mandar Dixit, John Corring, Ching Pui WAN, and the anonymous reviewers for helpful discussions and feedback while preparing this manuscript.


<a id="source-section-19"></a>

## References


- [1]

Amir, A., Hartman, T., Kapah, O., Shalom, B.R., Tsur, D.: Generalized LCS.
Theoretical computer science 409(3), 438–449 (2008)


- [2]

Chambers, C., Erwig, M., Luckey, M.: SheetDiff: A tool for identifying
changes in spreadsheets. In: 2010 IEEE Symposium on Visual Languages and
Human-Centric Computing. pp. 85–92. IEEE (2010)


- [3]

Corrêa, A.S., Zander, P.O.: Unleashing tabular content to open data: A
survey on pdf table extraction methods and tools. In: Proceedings of the 18th
Annual International Conference on Digital Government Research. pp. 54–63
(2017)


- [4]

Gao, L., Huang, Y., Déjean, H., Meunier, J.L., Yan, Q., Fang, Y., Kleber,
F., Lang, E.: ICDAR 2019 competition on table detection and recognition
(cTDaR). In: 2019 International Conference on Document Analysis and
Recognition (ICDAR). pp. 1510–1515. IEEE (2019)


- [5]

Gatterbauer, W., Bohunsky, P., Herzog, M., Krüpl, B., Pollak, B.: Towards
domain-independent information extraction from web tables. In: Proceedings of
the 16th international conference on World Wide Web. pp. 71–80 (2007)


- [6]

Göbel, M., Hassan, T., Oro, E., Orsi, G.: A methodology for evaluating
algorithms for table understanding in PDF documents. In: Proceedings of the
2012 ACM symposium on Document engineering. pp. 45–48 (2012)


- [7]

Göbel, M., Hassan, T., Oro, E., Orsi, G.: ICDAR 2013 table competition.
In: 2013 12th International Conference on Document Analysis and Recognition.
pp. 1449–1453. IEEE (2013)


- [8]

Harutyunyan, A., Borradaile, G., Chambers, C., Scaffidi, C.: Planted-model
evaluation of algorithms for identifying differences between spreadsheets.
In: 2012 IEEE Symposium on Visual Languages and Human-Centric Computing
(VL/HCC). pp. 7–14. IEEE (2012)


- [9]

Hassan, T.: Towards a common evaluation strategy for table structure
recognition algorithms. In: Proceedings of the 10th ACM symposium on Document
engineering. pp. 255–258 (2010)


- [10]

Hu, J., Kashi, R., Lopresti, D., Nagy, G., Wilfong, G.: Why table
ground-truthing is hard. In: Proceedings of Sixth International Conference on
Document Analysis and Recognition. pp. 129–133. IEEE (2001)


- [11]

Jaffar, J., Santosa, A.E., Voicu, R.: Efficient memoization for dynamic
programming with ad-hoc constraints. In: AAAI. vol. 8, pp. 297–303 (2008)


- [12]

Li, M., Cui, L., Huang, S., Wei, F., Zhou, M., Li, Z.: Tablebank: Table
benchmark for image-based table detection and recognition. In: Proceedings of
The 12th Language Resources and Evaluation Conference. pp. 1918–1925 (2020)


- [13]

Oro, E., Ruffolo, M.: TREX: An approach for recognizing and extracting tables
from PDF documents. In: 2009 10th International Conference on Document
Analysis and Recognition. pp. 906–910. IEEE (2009)


- [14]

Pinto, D., McCallum, A., Wei, X., Croft, W.B.: Table extraction using
conditional random fields. In: Proceedings of the 26th annual international
ACM SIGIR conference on Research and development in informaion retrieval. pp.
235–242 (2003)


- [15]

Smock, B., Pesala, R., Abraham, R.: PubTables-1M: Towards comprehensive
table extraction from unstructured documents. In: Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR). pp. 4634–4642
(June 2022)


- [16]

Wang, X.: Tabular abstraction, editing, and formatting (1996)


- [17]

Yildiz, B., Kaiser, K., Miksch, S.: pdf2table: A method to extract table
information from pdf files. In: IICAI. pp. 1773–1785. Citeseer (2005)


- [18]

Zhong, X., ShafieiBavani, E., Yepes, A.J.: Image-based table recognition: data,
model, and evaluation. arXiv preprint arXiv:1911.10683 (2019)
