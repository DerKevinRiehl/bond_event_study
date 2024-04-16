# Corporate Bond Market Event Studies
This is the repository of the publication "Corporate Bond Market Event Studies: Event-Induced Variance and Liquidity".

In this repository you will find Python code that allowsyou to conduct bond market event studies.


## Contents of this repository

- ./full_analysis_code
    - *Code_Report.py* the full original code to reproduce our tables from the paper
- ./example_data
    - *cusips_sample_filter.csv*
    - *trace_daily_r_non_NA_new_ALL_ex_50k.csv*
- ./src
    - *bond_event_study_tools.py* functions to conduct bond event study analysis
    - *synthetic_data_tools.py* function to generate mockup data
    - *example_synthetic.py* example how to generate synthetic data and produce bond study
    - *example_real.py* example how to load real data from csv and produce bond study

## How to get started

```# Imports
import pandas as pd
from bond_event_study_tools import loadRawTraceBondTradesData, calculateBondPrices, calculateBondReturns, calculateBenchmarkReturns, calculateAbnormalBondReturns, calculateFirmAbnormalReturns, calculateAbnormalStandardizedReturns, calculateAbnormalPreStandardizedReturns, calculatePreStandardizedBenchmarkReturns, calculateStandardizedBenchmarkReturns, calculateStandardizedReturns, calculateSigmasWeeks, calculateSigmas

# Define Parameters
weekly_basis = False
min_num_cusips_in_benchmark_filter = 5
WINSORIZE_THRESHOLD = 0.01
```

### Step 1: Bond Trades Dataset

First we need to prepare the bond trades dataset.
Either, we can load a real dataset from files, or we can generate a synthetic dataset.
Both options are outlined below.

#### Load Real Dataset
```# Load Real Trades Input Data
df_input, df_events, df_cusip_permno, lst_cusips, lst_permnos, lst_dates, experiment_montecarlo_selection, df_mat_rat_group = loadRawTraceBondTradesData(
    raw_input_csv_file="./input_data/trace_daily_raw.csv", 
    cusip_filter_input_file="./input_data/cusips_sample_filter.csv",
    n_monte_carlo_experiment_repetitions=10000,
    n_monte_carlo_samples=300)
```

##### Input File Structure (*trace_daily_raw.csv*)

| cusip_id | trd_exctn_dt | TWP | TWY | vol_day | n | maturity | RATING_SPR | RATING_FR | RATING_MR | moody_numeric | sp_numeric | fitch_numeric | RTNG | PERMNO |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | 
| "1" | "00033AAA6" | 2019-07-26 | 93 | 11.645925 | 54000 | 1 | 1816 | "B-" | NA | "B3" | "7.000000" | "7" | NA | 7 | NA |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |



#### Generate Synthetic Dataset
```# Generate Synthetic Trades Input Data
df_input, df_events, df_cusip_permno, lst_cusips, lst_permnos, lst_dates, experiment_montecarlo_selection = generateSyntheticTradesDataset(
    number_of_cusips=100, 
    number_of_permnos=20, 
    start_date=generateDateFromString("20100101"), 
    number_of_days=365*3, 
    number_of_trades_per_day_per_cusip_min=5, 
    number_of_trades_per_day_per_cusip_max=20, 
    min_num_days_to_maturity_filter=365,
    weekly_basis=weekly_basis, 
    n_monte_carlo_experiment_repetitions=10000,
    n_monte_carlo_samples=300)
```

The synthetic trades dataset has following structure.
 - df_input
     ['Date', 'CUSIP', 'FirmID_PERMNO', 'MaturityDate', 'Rating_numeric', 'Price_USD', 'Qty']
 - df_events
     ['Date', 'CUSIP']
 - df_cusip_permno
     ['Date', 'CUSIP', 'FirmID_PERMNO']
 - lst_cusips
     List of all Bond CUSIPS
 - lst_permnos
     List of all Firm PERMNOs
 - lst_dates
     List of all dates (only working days, no weekends, holidays not considered)
 - experiment_montecarlo_selection
     Selection of firm and date combinations for Tables 2 and 4
Here is an example of the randomly generated trades dataset.

|       |       Date    |  CUSIP | FirmID_PERMNO | MaturityDate | Rating_numeric | Price_USD | Qty|
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
|0|       2010-01-01 | IR9CM68QT |    CSJT3O024 | 2014-03-30     |        11 | 29.681576  | 96|
|1|       2010-01-01|  IR9CM68QT |    CSJT3O024|  2014-03-30     |        11 | 28.361588  | 77|
|2|       2010-01-01|  IR9CM68QT |    CSJT3O024 | 2014-03-30     |        11 | 26.609789  | 34|
|3|       2010-01-01|  IR9CM68QT |    CSJT3O024 | 2014-03-30     |        11 | 28.153327  | 19|
|4|       2010-01-01|  IR9CM68QT |    CSJT3O024 | 2014-03-30     |        11 | 30.367110  | 43|
 |  |        ...    |    ...    |       ...  | ...     |       ...    |    ... | ...|
|1365179 |2013-03-01|  BNFX8HZGW    | B7LHQ6E0C | 2014-03-02       |      12 | 21.323571 |  82|

### Step 2: Calculate Event Study Processed Tables
In the next step, let us calculate processed tables, such as prices, returns, abnormal returns etc.
```df_mat_rat_group = determineMaturityRatingGroupClassification(df_input)
df_prices = calculateBondPrices(df_input)
df_returns = calculateBondReturns(lst_dates, lst_cusips, df_prices, df_mat_rat_group, df_cusip_permno, weekly_basis=weekly_basis)
df_benchmarks = calculateBenchmarkReturns(df_returns, df_mat_rat_group, df_events, min_num_cusips_in_benchmark_filter=min_num_cusips_in_benchmark_filter)
df_abn_returns = calculateAbnormalBondReturns(df_returns, df_mat_rat_group, df_benchmarks)
df_abn_returns_firms = calculateFirmAbnormalReturns(df_abn_returns, df_cusip_permno)

# The calculation of Sigma Tables takes lots of time, we recommend to compute and save, so you can load it for later analysis
SIGMA_N_AT_LEAST_OBSV = 6
if weekly_basis:
    SIGMA_WEEK_BOUNDARY_1 = 11 # for ABSR AND SABR
    SIGMA_WEEK_BOUNDARY_2 = 2
    SIGMA_WEEK_BOUNDARY_3 = 2
    SIGMA_WEEK_BOUNDARY_4 = 11
    SIGMA_WEEK_BOUNDARY_5 = 22 # for pre-ABSR
    SIGMA_WEEK_BOUNDARY_6 = 2
    df_raw_sigma_n_t, df_abn_sigma_n_t, df_raw_pre_sigma_n_t = calculateSigmasWeeks(df_returns, df_abn_returns, SIGMA_WEEK_BOUNDARY_1, SIGMA_WEEK_BOUNDARY_2, SIGMA_WEEK_BOUNDARY_3, 
                        SIGMA_WEEK_BOUNDARY_4, SIGMA_WEEK_BOUNDARY_5, SIGMA_WEEK_BOUNDARY_6, lst_cusips, SIGMA_N_AT_LEAST_OBSV, WINSORIZE_THRESHOLD)
    df_raw_sigma_n_t.to_csv("df_raw_sigma_n_t_weeks.csv")
    df_abn_sigma_n_t.to_csv("df_abn_sigma_n_t_weeks.csv")
    df_raw_pre_sigma_n_t.to_csv("df_raw_pre_sigma_n_t_weeks.csv")
else:
    SIGMA_TIME_BOUNDARY_1 = 55 # for ABSR AND SABR
    SIGMA_TIME_BOUNDARY_2 = 6
    SIGMA_TIME_BOUNDARY_3 = 6
    SIGMA_TIME_BOUNDARY_4 = 55
    SIGMA_TIME_BOUNDARY_5 = 101 # for pre-ABSR
    SIGMA_TIME_BOUNDARY_6 = 6
    df_raw_sigma_n_t, df_abn_sigma_n_t, df_raw_pre_sigma_n_t = calculateSigmas(lst_dates, df_returns, df_abn_returns, SIGMA_TIME_BOUNDARY_1, SIGMA_TIME_BOUNDARY_2, SIGMA_TIME_BOUNDARY_3, 
                        SIGMA_TIME_BOUNDARY_4, SIGMA_TIME_BOUNDARY_5, SIGMA_TIME_BOUNDARY_6, lst_cusips, SIGMA_N_AT_LEAST_OBSV, WINSORIZE_THRESHOLD)
    df_raw_sigma_n_t.to_csv("df_raw_sigma_n_t.csv")
    df_abn_sigma_n_t.to_csv("df_abn_sigma_n_t.csv")
    df_raw_pre_sigma_n_t.to_csv("df_raw_pre_sigma_n_t.csv")
# This is how you would load them again
"""
if weekly_basis:
    df_raw_sigma_n_t = pd.read_csv("df_raw_sigma_n_t_weeks.csv")
    df_abn_sigma_n_t = pd.read_csv("df_abn_sigma_n_t_weeks.csv")
    df_raw_pre_sigma_n_t = pd.read_csv("df_raw_pre_sigma_n_t_weeks.csv")
else:
    df_raw_sigma_n_t = pd.read_csv("df_raw_sigma_n_t.csv")
    df_abn_sigma_n_t = pd.read_csv("df_abn_sigma_n_t.csv")
    df_raw_pre_sigma_n_t = pd.read_csv("df_raw_pre_sigma_n_t.csv")
del df_raw_sigma_n_t[df_raw_sigma_n_t.columns[0]]
del df_abn_sigma_n_t[df_abn_sigma_n_t.columns[0]]
del df_raw_pre_sigma_n_t[df_raw_pre_sigma_n_t.columns[0]]
if not weekly_basis:
    df_raw_sigma_n_t["Date"] = pd.to_datetime(df_raw_sigma_n_t["Date"])
    df_abn_sigma_n_t["Date"] = pd.to_datetime(df_abn_sigma_n_t["Date"])
    df_raw_pre_sigma_n_t["Date"] = pd.to_datetime(df_raw_pre_sigma_n_t["Date"])
"""

df_sabr, df_srr, df_srr_pre = calculateStandardizedReturns(df_returns, df_raw_sigma_n_t, df_raw_pre_sigma_n_t, df_abn_returns, df_abn_sigma_n_t, WINSORIZE_THRESHOLD, noised=False)
df_std_benchmarks = calculateStandardizedBenchmarkReturns(df_srr, df_mat_rat_group, min_num_cusips_in_benchmark_filter)
df_std_benchmarks_pre = calculatePreStandardizedBenchmarkReturns(df_srr_pre, df_mat_rat_group, min_num_cusips_in_benchmark_filter)
df_absr = calculateAbnormalStandardizedReturns(df_returns, df_raw_sigma_n_t, df_mat_rat_group, df_std_benchmarks, WINSORIZE_THRESHOLD)
df_absr_pre = calculateAbnormalPreStandardizedReturns(df_returns, df_raw_pre_sigma_n_t, df_mat_rat_group, df_std_benchmarks_pre, WINSORIZE_THRESHOLD)
```

The processed tables have following structure.
 - df_mat_rat_group
     ['Date', 'CUSIP', 'ratingGroup', 'maturityGroup']
     Bond_Maturity_Rating_Groups
 - df_prices
     ['Date', 'CUSIP', 'Price_WA']
     Bond Prices for each CUSIP and day based on weighted average traded volumes
 - df_returns
     ['Date', 'CUSIP', 'R_t']
     R_t = R(t-1, t+1)_n , Raw Bond Returns for each CUSIP and day
 - df_benchmarks
     ['Date', 'ratingGroup', 'maturityGroup', 'BM_t']
     Benchmark Returns_t based on rating and maturity group
 - df_abn_returns
     ['Date', 'CUSIP', 'ABR_t', 'ratingGroup', 'maturityGroup']
     Abnormal Bond Returns for each CUSIP and day
 - df_abn_returns_firms
     ['Date', 'FirmID_PERMNO', 'F_ABR_t']
     Abnormal Firm Returns for each PERMNO and day
 - df_raw_sigma_n_t
     ['CUSIP', 'Date', 'SIGMA_N_T']
     Standard deviations of raw returns of bond N at date t
 - df_abn_sigma_n_t
     ['CUSIP', 'Date', 'SIGMA_N_T']
     Standard deviations of abnormal returns of bond N at date t
 - df_raw_pre_sigma_n_t
     ['CUSIP', 'Date', 'SIGMA_N_T']
     Standard deviations of abnormal returns of bond N at date t with pre dates
 - df_sabr
     ['Date', 'CUSIP', 'SABR_t']
     SABR(t-1, t+1)_n , Standardized Abnormal Returns (SABR)
 - df_srr
     ['Date', 'CUSIP', 'SRR_t']
     SRR(t-1, t+1)_n , Standardized Raw Returns (SRR)
 - df_srr_pre
     ['Date', 'CUSIP', 'SRR_pre_t']
     SRR(t-1, t+1)_n , Pre-Standardized Raw Returns (SRR)
 - df_std_benchmarks
     ['Date', 'ratingGroup', 'maturityGroup', 'SBM_t']
     SBM(t-1,t+1)_n  , Calculate Standardized Benchmark Returns_t
 - df_std_benchmarks_pre
     ['Date', 'ratingGroup', 'maturityGroup', 'SBM_pre_t']
     SBM(t-1,t+1)_n  , Calculate Pre-Standardized Benchmark Returns_t
 - df_absr
     ['Date', 'CUSIP', 'ABSR_t', 'ratingGroup', 'maturityGroup']
     ABSR(t-1, t+1)_n , Abnormal Standardized Returns (ABSR)
 - df_absr_pre
     ['Date', 'CUSIP', 'ABSR_pre_t', 'ratingGroup', 'maturityGroup']
     ABSR(t-1, t+1)_n , Abnormal Pre-Standardized Returns (ABSR_PRE)

## Citations

Please cite our paper if you find this work useful:

[APA-LIKE-CITATION]

```
@article{mueller2024bondmarket,
    title={Corporate Bond Market Event Studies: Event-Induced Variance and Liquidity},
    author={Müller, Lukas and Riehl, Kevin and Buschulte, Sonja and Weiss, Patrick}
    journal={TBD},
    year={2024}
}
```

## Acknowledgements

We would like to thank ???. 
(e.g. Dirk Schiereck, Chair of Corporate Finance, TU Darmstadt, provision of data ?)