""" 
Notes:

1) Since the development of modern portfolio theory by Markowitz (1952), mean-variance optimization (MVO) has received considerable attention.
Unfortunately, it faces a number of shortcomings, including high sensitivity to the input parameters (expected returns and covariance), weight concentration, high turnover, and poor out-of-sample performance.
It is well-known that naive allocation (1/N, inverse-vol, etc.) tends to outperform MVO out-of-sample (DeMiguel, 2007).
Numerous approaches have been developed to alleviate these shortcomings (shrinkage, additional constraints, regularization, uncertainty set, higher moments, Bayesian approaches, coherent risk measures, left-tail risk optimization, distributionally robust optimization, factor model, risk-parity, hierarchical clustering, ensemble methods, pre-selection, etc.).
Given the large number of methods, and the fact that they can be combined, there is a need for a unified framework with a machine-learning approach to perform model selection, validation, and parameter tuning while mitigating the risk of data leakage and overfitting.
This framework (skfolio) is built on scikit-learn's API.

2) The
    config = {
        "_index": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
        "Open": st.column_config.NumberColumn("Open ($)", format="$.2f"),
        "High": st.column_config.NumberColumn("High ($)", format="$.2f"),
        "Low": st.column_config.NumberColumn("Low ($)", format="$.2f"),
        "Close": st.column_config.NumberColumn("Close ($)", format="$"),
        "Adj Close": st.column_config.NumberColumn("Adj Close ($)", format="$.2f"),
        "Volume": st.column_config.NumberColumn("Volume", format=",") 
    }





3)


"""