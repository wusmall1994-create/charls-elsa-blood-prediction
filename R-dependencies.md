# R figure environment

The transport decision-curve figure was rendered using R 4.6.1 with ggplot2 4.0.3, patchwork 1.3.2, svglite 2.2.2 and ragg 1.5.2. Arial is used for figure labels. The script accepts an aggregate source CSV and an output stem; it writes a 600-dpi PNG and an editable SVG. It does not fit prediction models.

```text
Rscript scripts/figures/transport_decision_curves.R decision_curve_source.csv Figure_4
```
