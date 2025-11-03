## LRTrend

**LRTrend** is a method for detecting outbreaks and uptrends in real time. Given one or more public health data streams (such as case counts, deaths, search trends, etc.) LRTrend will detect if the estimated growth rate is exceedingly high within a recent window, and can be done with multiple data streams to increase detection power. LRTrend can also learn cross-region epidemic similarity, constructing a disease-specific epidemic network, and aggregate across this network to further increase power.


![LRTrend overview](figs/method_overview.png)


## How to use LRTrend

### Installation

```bash
pip install -r requirements.txt
```

### Experiments

To reproduce the main experiments in our paper, see the experiments folder. Some file paths may need to be adjusted. whole_pipeline.ipynb will reproduce the power/delay statistics on the provided data. power_delay_plots.ipynb will reproduce the main power/delay figures. TrendVisualization.ipynb will reproduce the longitudinal figures visualizing trends. geo_figs.ipynb will reproduce all figures corresponding to the geographic section. cross_data_correlation.ipynb will reproduce the correlation figures.

---

## Citation
