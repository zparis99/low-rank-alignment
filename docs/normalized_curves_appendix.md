# Appendix: Normalized performance curves

We use normalized performance curves to compare how quickly RRR and PCA
representations approach their best observed predictive performance as rank
increases. Each dataset, model-layer, and method curve is normalized independently,
including within-dataset and transferred representations. This allows comparison
across curves with different absolute correlations and tested rank ranges.

Normalization is needed to average convergence effectively across layers. Layers
can differ both in the correlation they achieve and in the range of ranks tested.
Averaging raw correlations would mix the shape of convergence with differences
in absolute prediction strength, while averaging by raw rank would compare
different fractions of each layer's available rank range. Normalizing performance
to each layer's own peak and rank to its tested range puts the curves on common
coordinates. Their average then answers a consistent question: what fraction of
a layer's best tested performance is reached at a given fraction of its tested
rank range? Each layer contributes equally to this relative convergence summary,
regardless of its absolute peak correlation.

Let \(r(k_i)\) be the mean prediction correlation at tested rank \(k_i\), ordered
by increasing rank. Correlations are averaged in Fisher-z space and transformed
back before normalization:

\[
\bar r=\tanh\left(\operatorname{mean}[\operatorname{arctanh}(r)]\right).
\]

Evaluation folds are combined within each model layer. For leave-one-subject-out
convergence curves, subject-level correlations are also combined before
normalization. These curves therefore describe the convergence of aggregated
correlations, rather than an average of independently normalized subject curves.

We clip negative correlations to zero, take the cumulative maximum across ranks,
and divide by the resulting peak:

\[
m_i=\max_{j\leq i}\max\{r(k_j),0\},
\qquad M=m_n,\qquad y_i=\frac{m_i}{M}.
\]

This monotonic envelope retains the best performance reached so far whenever
performance decreases at a higher rank. The vertical axis reports the fraction
of the curve's own best tested mean correlation: a value of 0.9 means 90% of
that peak correlation, not a correlation of 0.9 or 90% of explained variance.
The peak is the best value among tested ranks, not an independently estimated
prediction ceiling.

We also map each curve's tested rank range to the interval \([0,1]\):

\[
x_i=\frac{k_i-k_1}{k_n-k_1}.
\]

Zero on this axis denotes the first tested rank, usually rank 1, rather than a
representation with zero dimensions. The curve need not begin at zero performance,
but every usable curve ends at one. Curves without a positive peak or at least
two distinct finite rank–correlation pairs are omitted.

Faint lines show individual model-layer curves. Thick lines are calculated by
linearly interpolating each normalized curve onto a shared relative-rank grid
containing zero and 199 geometrically spaced points from \(10^{-4}\) to 1, then
taking their arithmetic mean. Each contributing model layer receives equal
weight, so model families with more included layers contribute more curves.
The horizontal axis uses a symmetric logarithmic display scale, with a linear
region near zero (threshold 0.01), to show early convergence. Interpolation is
linear in relative rank; the display scale does not change the averaging.

## Area under the normalized curve

The associated AUC uses the same clipped, monotonic, peak-normalized performance
but integrates over a separate normalized logarithmic rank coordinate:

\[
u_i=\frac{\log k_i}{\log k_n},
\qquad
\mathrm{AUC}=\sum_{i=1}^{n-1}
\frac{y_i+y_{i+1}}{2}(u_{i+1}-u_i).
\]

This trapezoidal calculation requires a tested rank of 1 and at least two
distinct positive ranks. AUC lies between zero and one; larger values indicate
that a curve reaches more of its own peak at lower log-relative ranks. It is
not the geometric area under the displayed symmetric-log plot. Positive
RRR-minus-PCA AUC differences indicate faster relative convergence for RRR;
within-dataset-minus-transfer differences have the analogous interpretation.
Because each curve has its own peak and rank range, these comparisons do not
establish which method has the higher absolute prediction correlation. Absolute
correlations and saturation ranks provide complementary information.

## Target-quantile curves and saturation

Normalized target-quantile curves apply the same monotonic, peak normalization
to quantiles of electrode or voxel correlations. Each target's correlations
are first averaged across evaluation folds in Fisher-z space; quantiles across
targets are then computed at each rank. Each quantile curve is divided by its
own peak, with a default eligibility requirement that the peak exceed 0.005.
The paper notebook displays the 5th, 10th, 25th, 50th, 75th, and 90th percentiles.
Mean normalized quantile curves use the shared relative-rank grid and equal
weighting of eligible model layers, separately for each quantile.

The primary saturation statistic uses all seven loaded quantiles, including the
95th percentile. For each eligible quantile, we find its first crossing of 99%
of its own peak, linearly interpolating between the bracketing tested ranks in
the original rank coordinate. Saturation is the latest of these first crossings:
all eligible quantiles have reached their threshold by that rank. If no quantile
is eligible, saturation is undefined. For leave-one-subject-out analyses,
saturation is calculated separately for each subject and then averaged within
a model layer, whereas convergence curves aggregate subjects before normalization.
Sensitivity figures vary the threshold or peak cutoff, and the interactive
notebook allows selection of the saturation percentage.
