###############################################################################
# Source Code for Publication "Corporate Bond Market Event Studies: Event-Induced Variance and Liquidity"
# Authors: Lukas Müller, Kevin Riehl, Sonja Buschulte, Patrick Weiss
# Code Author: Kevin Riehl
###############################################################################




###############################################################################
# Imports
###############################################################################
import numpy as np
import pandas as pd
import datetime
import random

from scipy.stats import kurtosis, skew, wilcoxon
from scipy import stats
from statsmodels.stats.descriptivestats import sign_test




###############################################################################
# Methods
###############################################################################


################### IO Methods
def loadListOfCusips(file, num=-1):
    f = open(file, "r")
    content = f.read()
    f.close()
    cusips = content.split("\n")
    if(num==-1):
        return cusips
    else:
        return cusips[1:num+1]

def loadRawTraceBondTradesData(raw_input_csv_file, cusip_filter_input_file, n_monte_carlo_experiment_repetitions, n_monte_carlo_samples):
    """
    This function loads a real trades dataset.
    
    Parameters
    ----------
    raw_input_csv_file : str
    cusip_filter_input_file : str
    n_monte_carlo_experiment_repetitions : int
    n_monte_carlo_samples : int
    
    Returns
    -------
    df_input : pandas.DataFrame
         ['Date', 'CUSIP', 'FirmID_PERMNO', 'MaturityDate', 'Rating_numeric', 'Price_USD', 'Qty']
    df_events : pandas.DataFrame
         ['Date', 'CUSIP']
    df_cusip_permno : pandas.DataFrame
         ['Date', 'CUSIP', 'FirmID_PERMNO']
    lst_cusips : lst[str]
         List of all Bond CUSIPS
    lst_permnos : lst[str]
         List of all Firm PERMNOs
    lst_dates : lst[datetime.datetime]
         List of all dates (only working days, no weekends, holidays not considered)
    experiment_montecarlo_selection
         Selection of firm and date combinations for Monte Carlo Experiments
    """
    # Load raw data
    df_input = pd.read_csv(raw_input_csv_file)
    df_input = df_input[["trd_exctn_dt", "cusip_id", "PERMNO", "maturity", "RTNG", "TWP", "vol_day"]]
    df_input = df_input.rename(columns={"trd_exctn_dt":"Date", "cusip_id":"CUSIP", "PERMNO":"FirmID_PERMNO", "maturity":"remaining_days_until_maturity", "RTNG":"Rating_numeric", "TWP":"Price_USD", "vol_day":"Qty"})
    df_input["Date"] = pd.to_datetime(df_input['Date'], errors="coerce")
    df_input = df_input.drop_duplicates()
    df_input = df_input[~df_input["FirmID_PERMNO"].isna()]
    if(cusip_filter_input_file is not None):
        df_cusip_filter = pd.read_csv(cusip_filter_input_file)
        del df_cusip_filter["Unnamed: 0"]
        df_input = df_input.merge(df_cusip_filter, left_on=["CUSIP"], right_on=["cusip"], how="left")
        df_input = df_input[~df_input["cusip"].isna()]
        del df_input["cusip"]
    # Generate CUSIP / PERMNO Loopup Table
    df_cusip_permno = df_input[["Date", "CUSIP", "FirmID_PERMNO"]]
    df_cusip_permno = df_cusip_permno.drop_duplicates()
    df_cusip_permno = df_cusip_permno.dropna()
    # Prepare List
    lst_cusips = df_input[["CUSIP"]].dropna().drop_duplicates()["CUSIP"].tolist()
    lst_permnos = df_input[["FirmID_PERMNO"]].dropna().drop_duplicates()["FirmID_PERMNO"].tolist()
    lst_dates = df_input[["Date"]].dropna().drop_duplicates()["Date"].tolist()
    # Generate Empty Events
    df_events = pd.DataFrame([], columns=["Date", "CUSIP"])
    df_events["TO_BE_EXCLUDED"] = 1
    # Determine Bond Maturity and Rating Groups
    df_mat_rat_group = df_input.groupby(["Date", "CUSIP", "remaining_days_until_maturity", "Rating_numeric"]).mean()
    df_mat_rat_group = df_mat_rat_group.reset_index()
    del df_mat_rat_group["Price_USD"]
    del df_mat_rat_group["Qty"]
    df_mat_rat_group = df_mat_rat_group.drop_duplicates()
    df_mat_rat_group["ratingGroup"] = df_mat_rat_group.apply(determineRatingGroup, axis=1)
    df_mat_rat_group["maturityGroup"] = df_mat_rat_group.apply(determineMaturityGroup_raw, axis=1)
    df_mat_rat_group = df_mat_rat_group[["Date", "CUSIP", "ratingGroup", "maturityGroup"]]
    df_mat_rat_group = df_mat_rat_group.dropna()
    ## P_n,t                 >>  Calculate Bond Prices for each CUSIP and day 
    df_prices = df_input.copy()
    df_prices = df_prices[["Date", "CUSIP", "Price_USD"]]
    df_prices["Price_WA"] = df_prices["Price_USD"]
    del df_prices["Price_USD"]
    # experiment_montecarlo_selection
    experiment_montecarlo_selection = []
    for i in range(0, n_monte_carlo_experiment_repetitions):
        random_firms = [random.choice(lst_permnos) for x in range(0,n_monte_carlo_samples)]
        random_days = [random.choice(lst_dates) for x in range(0,n_monte_carlo_samples)]
        data = {}
        data["firms"] = random_firms
        data["days"] = random_days
        experiment_montecarlo_selection.append(data)
    return df_input, df_events, df_cusip_permno, lst_cusips, lst_permnos, lst_dates, experiment_montecarlo_selection, df_mat_rat_group


################### Time Related Methods
def getNextWorkingDay(dt):
    next_date = dt + datetime.timedelta(days=1)
    if(next_date.weekday()==5):
        next_date = dt + datetime.timedelta(days=3)
    return next_date

def getPreviousWorkingDay(dt):
    prev_date = dt + datetime.timedelta(days=-1)
    if(prev_date.weekday()==6):
        prev_date = dt + datetime.timedelta(days=-3)
    return prev_date

def getPreviousWorkingDayLambda(row):
    return getPreviousWorkingDay(row["Date"])

def getNextWorkingDayLambda(row):
    return getNextWorkingDay(row["Date"])
    
def getListOfDays(start_date, number_of_days):
    lst_dates = []
    current_date = start_date
    for day in range(0, number_of_days):
        lst_dates.append(current_date)
        current_date = getNextWorkingDay(current_date)
    return lst_dates

def getListOfDaysFromTo(start_date, end_date):
    lst_dates = []
    current_date = start_date
    while True:
        lst_dates.append(current_date)
        if(current_date == end_date):
            break
        current_date = getNextWorkingDay(current_date)
    return lst_dates

def getDaysFromDateTimeDifference(row):
    return row["remaining_days_until_maturity"].days


################### Rating and Maturity Group Functions
def determineRatingGroup(row):
    if(row["Rating_numeric"]>=19):   # AAA to AA
        return 0
    elif(row["Rating_numeric"]>=16): # A
        return 1
    elif(row["Rating_numeric"]>=13): # BAA
        return 2
    elif(row["Rating_numeric"]>=10): # BA
        return 3
    elif(row["Rating_numeric"]>=7):  # B
        return 4
    else:                            # Below B
        return 5
    
def determineMaturityGroup(row):
    if(row["remaining_days_until_maturity"].days<365):       # below one year
        return np.nan
    elif(row["remaining_days_until_maturity"].days<=3*365):  # 1-3 years
        return 0
    elif(row["remaining_days_until_maturity"].days<=5*365):  # 3-5 years
        return 1
    elif(row["remaining_days_until_maturity"].days<=10*365): # 5-10 years
        return 2
    else:                                                    # Over 10 years
        return 3
    
def determineMaturityGroup_raw(row):
    if(row["remaining_days_until_maturity"]<365):           # below one year
        return np.nan
    elif(row["remaining_days_until_maturity"]<=3*365):      # 1-3 years
        return 0
    elif(row["remaining_days_until_maturity"]<=5*365):      # 3-5 years
        return 1
    elif(row["remaining_days_until_maturity"]<=10*365):     # 5-10 years
        return 2
    else:                                                   # Over 10 years
        return 3
    

################### DataFrame Calculation Functions
def determineMaturityRatingGroupClassification(df_input):
    df_input = df_input[['Date', 'CUSIP', 'MaturityDate', 'Rating_numeric', 'Price_USD', 'Qty']]
    df_mat_rat_group = df_input.groupby(["Date", "CUSIP", "MaturityDate", "Rating_numeric"]).mean()
    df_mat_rat_group = df_mat_rat_group.reset_index()
    del df_mat_rat_group["Price_USD"]
    del df_mat_rat_group["Qty"]
    df_mat_rat_group = df_mat_rat_group.drop_duplicates()
    df_mat_rat_group["remaining_days_until_maturity"] = df_mat_rat_group["MaturityDate"] - df_mat_rat_group["Date"]
    df_mat_rat_group["ratingGroup"] = df_mat_rat_group.apply(determineRatingGroup, axis=1)
    df_mat_rat_group["maturityGroup"] = df_mat_rat_group.apply(determineMaturityGroup, axis=1)
    df_mat_rat_group = df_mat_rat_group[["Date", "CUSIP", "ratingGroup", "maturityGroup"]]
    df_mat_rat_group = df_mat_rat_group.dropna()
    return df_mat_rat_group

def calculateBondPrices(df_input):
    df_prices = df_input.groupby(["Date", "CUSIP"]).apply(weighted_average, 'Price_USD', 'Qty')
    df_prices = df_prices.reset_index()
    df_prices["Price_WA"] = df_prices[0]
    del df_prices[0]
    return df_prices

def calculateBondReturns(lst_dates, lst_cusips, df_prices, df_mat_rat_group, df_cusip_permno, weekly_basis):
    """
    # R_t = R(t-1, t+1)_n    >>  Calculate Bond Returns for each CUSIP and day
    """
    if(weekly_basis):
        # Generate Weeks instead of dates
        lst_date_week_map = []
        for d in lst_dates:
            lst_date_week_map.append([d, "{0:0=4d}".format(d.isocalendar()[0])+"-"+"{0:0=2d}".format(d.isocalendar()[1])])
        date_week_map = pd.DataFrame(lst_date_week_map, columns=["date", "week"])
        
        date_week_map_first = date_week_map.groupby(["week"])["date"].agg("min")
        date_week_map_first = date_week_map_first.reset_index()
        date_week_map_first = date_week_map_first.rename(columns={"date":"first"})
        date_week_map_last = date_week_map.groupby(["week"])["date"].agg("max")
        date_week_map_last = date_week_map_last.reset_index()
        date_week_map_last = date_week_map_last.rename(columns={"date":"last"})
        week_map = date_week_map_first.merge(date_week_map_last, on="week", how="left")
        week_map["dist"] = week_map["last"]-week_map["first"]
        week_map["filt"] = week_map["dist"].astype("timedelta64[D]") >=2
        week_map = week_map[week_map["filt"]]
        week_map = week_map[["week", "first", "last"]]
        
        df_prices = df_prices.merge(date_week_map, left_on=["Date"], right_on=["date"])
        df_prices = df_prices.merge(week_map, left_on=["week"], right_on=["week"])
        df_prices = df_prices[["Date", "CUSIP", "Price_WA", "week", "first", "last"]]
        df_prices["first_filt"] = df_prices["Date"]==df_prices["first"]
        df_prices["last_filt"] = df_prices["Date"]==df_prices["last"]
        
        df_prices_first = df_prices.copy()
        df_prices_first = df_prices_first[df_prices_first["first_filt"]]
        df_prices_first = df_prices_first[["week", "CUSIP", "Price_WA"]]
        
        df_prices_last = df_prices.copy()
        df_prices_last = df_prices_last[df_prices_last["last_filt"]]
        df_prices_last = df_prices_last[["week", "CUSIP", "Price_WA"]]
        
        week_cusip_comb = []
        for week in list(set(week_map["week"].tolist())):
            for cus in lst_cusips:
                week_cusip_comb.append([week, cus])
        df_returns = pd.DataFrame(week_cusip_comb, columns=["Date", "CUSIP"])
        df_returns = df_returns.merge(df_prices_first, left_on=["Date", "CUSIP"], right_on=["week", "CUSIP"], how="left")
        df_returns = df_returns.rename(columns={"Price_WA":"Price_first"})
        df_returns = df_returns.merge(df_prices_last, left_on=["Date", "CUSIP"], right_on=["week", "CUSIP"], how="left")
        df_returns = df_returns.rename(columns={"Price_WA":"Price_last"})
        df_returns = df_returns[["Date", "CUSIP", "Price_first", "Price_last"]]
        df_returns = df_returns.dropna()
        df_returns["R_t"] = 100*(df_returns["Price_last"] - df_returns["Price_first"]) / df_returns["Price_first"]
        df_returns = df_returns[["Date", "CUSIP", "R_t"]]
        
        df_mat_rat_group = df_mat_rat_group.merge(date_week_map, left_on=["Date"], right_on=["date"])
        df_mat_rat_group2 = df_mat_rat_group.groupby(["week", "CUSIP"])["ratingGroup", "maturityGroup"].agg("min")
        df_mat_rat_group2 = df_mat_rat_group2.reset_index()
        df_mat_rat_group2 = df_mat_rat_group2.rename(columns={"week":"Date"})
        df_mat_rat_group = df_mat_rat_group2
        
        df_cusip_permno = df_cusip_permno.merge(date_week_map, left_on="Date", right_on="date")
        del df_cusip_permno["Date"]
        del df_cusip_permno["date"]
        df_cusip_permno = df_cusip_permno.rename(columns={"week":"Date"})
        df_cusip_permno2 = df_cusip_permno.groupby(["Date", "CUSIP"])["FirmID_PERMNO"].agg("min")
        df_cusip_permno2 = df_cusip_permno2.reset_index()
        df_cusip_permno = df_cusip_permno2
        
    else:
        df_returns = df_prices.copy()
        del df_returns["Price_WA"]
        df_returns["Date_prev"] = df_returns.apply(getPreviousWorkingDayLambda, axis=1)
        df_returns["Date_after"] = df_returns.apply(getNextWorkingDayLambda, axis=1)
        df_returns = df_returns.merge(df_prices, left_on=["Date_prev", "CUSIP"], right_on=["Date", "CUSIP"], how="left")
        df_returns["Price_WA_prev"] = df_returns["Price_WA"]
        del df_returns["Price_WA"]
        df_returns = df_returns.merge(df_prices, left_on=["Date_after", "CUSIP"], right_on=["Date", "CUSIP"], how="left")
        df_returns["Price_WA_after"] = df_returns["Price_WA"]
        del df_returns["Price_WA"]
        del df_returns["Date"]
        del df_returns["Date_y"]
        df_returns["Date"] = df_returns["Date_x"]
        del df_returns["Date_x"]
        df_returns["R_t"] = 100*(df_returns["Price_WA_after"] - df_returns["Price_WA_prev"]) / df_returns["Price_WA_prev"]
        df_returns = df_returns.dropna()
        df_returns = df_returns[["Date", "CUSIP", "R_t"]]
    print("Finished df_returns")
    return df_returns

def calculateBenchmarkReturns(df_returns, df_mat_rat_group, df_events, min_num_cusips_in_benchmark_filter):
    """
    # BM_t = BM(t-1,t+1)_n  >> Calculate Benchmark Returns_t
    """
    df_benchmarks = df_returns.copy()
    df_benchmarks = df_benchmarks[["Date", "R_t", "CUSIP"]]
    df_benchmarks = df_benchmarks.merge(df_mat_rat_group, on=["Date", "CUSIP"])
    df_benchmarks = df_benchmarks[["Date", "CUSIP", "R_t", "ratingGroup", "maturityGroup"]]
    df_benchmarks = df_benchmarks.merge(df_events, on=["Date", "CUSIP"], how="left")
    df_benchmarks = df_benchmarks[~df_benchmarks["TO_BE_EXCLUDED"].notnull()]
    del df_benchmarks["TO_BE_EXCLUDED"]
    df_benchmarks = df_benchmarks.groupby(["Date", "ratingGroup", "maturityGroup"]).agg(count=("CUSIP", "size"), mean=("R_t", "mean"))
    df_benchmarks = df_benchmarks.reset_index()
    df_benchmarks = df_benchmarks[df_benchmarks["count"]>=min_num_cusips_in_benchmark_filter]
    del df_benchmarks["count"]
    df_benchmarks["BM_t"] = df_benchmarks["mean"]
    del df_benchmarks["mean"]
    df_benchmarks = df_benchmarks.dropna()
    print("Finished df_benchmarks")
    return df_benchmarks


def calculateAbnormalBondReturns(df_returns, df_mat_rat_group, df_benchmarks):
    """
    # ABR_t = ABR(t-1, t+1)_n  >>  Calculate Abnormal Bond Returns for each CUSIP and day
    """
    df_abn_returns = generateShocked_ABR(0, df_returns, df_mat_rat_group, df_benchmarks)
    print("Finished df_abn_returns")
    return df_abn_returns

def calculateFirmAbnormalReturns(df_abn_returns, df_cusip_permno):
    """
    # F_ABR_t = ABR(t-1, t+1)_n  >>  Calculate Firm Abnormal Returns for each PERMNO and day
    """
    df_abn_returns_firms = aggregateToFirmLevel(df_abn_returns, df_cusip_permno, "ABR_t")
    print("Finished df_abn_returns_firms")
    return df_abn_returns_firms


def calculateStandardizedReturns(df_returns, df_raw_sigma_n_t, df_raw_pre_sigma_n_t, df_abn_returns, df_abn_sigma_n_t, WINSORIZE_THRESHOLD, noised=False):
    if noised:
        # SABR_t = SABR(t-1, t+1)_n >> Standardized Abnormal Returns (SABR)
        df_sabr = generateShocked_SABR_noise(0, df_abn_returns, df_abn_sigma_n_t, WINSORIZE_THRESHOLD)
        print("Finished df_sabr")
        # SRR_t = SRR(t-1, t+1)_n >> Standardized Raw Returns (SRR)
        df_srr = generateShocked_SSR_noise(0, df_returns, df_raw_sigma_n_t)
        print("Finished df_srr")
        # SRR_pre_t = SRR(t-1, t+1)_n >> Pre-Standardized Raw Returns (SRR)
        df_srr_pre = generateShocked_SSR_pre_noise(0, df_returns, df_raw_pre_sigma_n_t)
        print("Finished df_srr_pre")
    else:
        # SABR_t = SABR(t-1, t+1)_n >> Standardized Abnormal Returns (SABR)
        df_sabr = generateShocked_SABR(0, df_abn_returns, df_abn_sigma_n_t,WINSORIZE_THRESHOLD)
        print("Finished df_sabr")
        # SRR_t = SRR(t-1, t+1)_n >> Standardized Raw Returns (SRR)
        df_srr = generateShocked_SSR(0, df_returns, df_raw_sigma_n_t)
        print("Finished df_srr")
        # SRR_pre_t = SRR(t-1, t+1)_n >> Pre-Standardized Raw Returns (SRR)
        df_srr_pre = generateShocked_SSR_pre(0, df_returns, df_raw_pre_sigma_n_t)
        print("Finished df_srr_pre")
    return df_sabr, df_srr, df_srr_pre


def calculateStandardizedBenchmarkReturns(df_srr, df_mat_rat_group, min_num_cusips_in_benchmark_filter):
    """
    # SBM_t = SBM(t-1,t+1)_n  >> Calculate Standardized Benchmark Returns_t
    """
    df_std_benchmarks = df_srr.copy()
    df_std_benchmarks = df_std_benchmarks[["Date", "SRR_t", "CUSIP"]]
    df_std_benchmarks = df_std_benchmarks.merge(df_mat_rat_group, on=["Date", "CUSIP"])
    df_std_benchmarks = df_std_benchmarks[["Date", "CUSIP", "SRR_t", "ratingGroup", "maturityGroup"]]
    df_std_benchmarks = df_std_benchmarks.groupby(["Date", "ratingGroup", "maturityGroup"]).agg(count=("CUSIP", "size"), mean=("SRR_t", "mean"))
    df_std_benchmarks = df_std_benchmarks.reset_index()
    df_std_benchmarks = df_std_benchmarks[df_std_benchmarks["count"]>=min_num_cusips_in_benchmark_filter]
    del df_std_benchmarks["count"]
    df_std_benchmarks["SBM_t"] = df_std_benchmarks["mean"]
    del df_std_benchmarks["mean"]
    df_std_benchmarks = df_std_benchmarks.dropna()
    print("Finished df_std_benchmarks")
    return df_std_benchmarks

def calculatePreStandardizedBenchmarkReturns(df_srr_pre, df_mat_rat_group, min_num_cusips_in_benchmark_filter):
    """
    # SBM_pre_t = SBM(t-1,t+1)_n  >> Calculate Pre-Standardized Benchmark Returns_t
    """
    df_std_benchmarks_pre = df_srr_pre.copy()
    df_std_benchmarks_pre = df_std_benchmarks_pre[["Date", "SRR_pre_t", "CUSIP"]]
    df_std_benchmarks_pre = df_std_benchmarks_pre.merge(df_mat_rat_group, on=["Date", "CUSIP"])
    df_std_benchmarks_pre = df_std_benchmarks_pre[["Date", "CUSIP", "SRR_pre_t", "ratingGroup", "maturityGroup"]]
    df_std_benchmarks_pre = df_std_benchmarks_pre.groupby(["Date", "ratingGroup", "maturityGroup"]).agg(count=("CUSIP", "size"), mean=("SRR_pre_t", "mean"))
    df_std_benchmarks_pre = df_std_benchmarks_pre.reset_index()
    df_std_benchmarks_pre = df_std_benchmarks_pre[df_std_benchmarks_pre["count"]>=min_num_cusips_in_benchmark_filter]
    del df_std_benchmarks_pre["count"]
    df_std_benchmarks_pre["SBM_pre_t"] = df_std_benchmarks_pre["mean"]
    del df_std_benchmarks_pre["mean"]
    df_std_benchmarks_pre = df_std_benchmarks_pre.dropna()
    print("Finished df_std_benchmarks_pre")
    return df_std_benchmarks_pre

def calculateAbnormalStandardizedReturns(df_returns, df_raw_sigma_n_t, df_mat_rat_group, df_std_benchmarks, WINSORIZE_THRESHOLD):
    """
    # ABSR_t = ABSR(t-1, t+1)_n >> Abnormal Standardized Returns (ABSR)
    """
    df_absr = generateShocked_ABSR(0, df_returns, df_raw_sigma_n_t, df_mat_rat_group, df_std_benchmarks, WINSORIZE_THRESHOLD)
    print("Finished df_absr")
    return df_absr

def calculateAbnormalPreStandardizedReturns(df_returns, df_raw_pre_sigma_n_t, df_mat_rat_group, df_std_benchmarks_pre, WINSORIZE_THRESHOLD):
    # ABSR_pre_t = ABSR(t-1, t+1)_n >> Abnormal Pre-Standardized Returns (ABSR_Pre)
    df_absr_pre = generateShocked_ABSR_pre(0, df_returns, df_raw_pre_sigma_n_t, df_mat_rat_group, df_std_benchmarks_pre, WINSORIZE_THRESHOLD)
    print("Finished df_absr_pre")
    return df_absr_pre


################### Descriptive Statistics
def getPopulationValues(lst_pop, pop_func):
    lst_vals = []
    for df_pop in lst_pop:
        lst_vals.append(pop_func(df_pop))
    return lst_vals

def getMean(df_pop):
    return np.mean(df_pop)

def getMedian(df_pop):
    return np.median(df_pop)

def getSTD(df_pop):
    return np.std(df_pop)

def getSkew(df_pop):
    return skew(df_pop)

def getKurtosis(df_pop):
    return kurtosis(df_pop)

def getSharePositive(df_pop):
    b = np.sign(df_pop).value_counts(normalize=True).rename({0:'zero', 1:'pos', -1:'neg'})
    return b[1]*100

def getN(df_pop):
    return df_pop.shape[0]


################### Statistical Test Functions
def doTtest(pop, BOEHMER_T_TEST, SIGNIFICANCE_LEVEL):
    if BOEHMER_T_TEST:
        std = np.std(pop)
        pop = pop/std
    t_test = stats.ttest_1samp(pop, popmean=0.0)
    experiment_t = t_test[0]
    experiment_p = t_test[1]
    experiment_res = 0
    if(experiment_p<SIGNIFICANCE_LEVEL):
        if(experiment_t>0):
            experiment_res = +1
        else:
            experiment_res = -1
    return experiment_res

def doSRtest(pop, SIGNIFICANCE_LEVEL):
    sr_test_ts = wilcoxon(pop, alternative='two-sided')
    experiment_p = sr_test_ts[1]
    experiment_res = 0
    if(experiment_p<SIGNIFICANCE_LEVEL):
        if(np.median(pop)>0):
            experiment_res = +1
        else:
            experiment_res = -1
    return experiment_res

def doSGtest(pop, SIGNIFICANCE_LEVEL):
    sg_test_res = sign_test(pop)
    experiment_p = sg_test_res[1]
    experiment_res = 0
    if(experiment_p<SIGNIFICANCE_LEVEL):
        if(np.sum(pop>0)/len(pop)>0.5):
            experiment_res = +1
        else:
            experiment_res = -1
    return experiment_res

def getTrippleTestResult(pop):
    test_results = {}
    test_results["t"] = doTtest(pop)
    test_results["sr"] = doSRtest(pop)
    test_results["sg"] = doSGtest(pop)
    return test_results
    
    

################### Others



################### Others
def weighted_average(df, values, weights):
    return sum(df[values] * df[weights]) / df[weights].sum()


def winsorize(dat, WINSORIZE_THRESHOLD):
    dat = np.asarray(dat)
    lower_limit_perc = np.percentile(dat, WINSORIZE_THRESHOLD*100)
    upper_limit_perc = np.percentile(dat, (1 - WINSORIZE_THRESHOLD)*100)
    dat[dat<lower_limit_perc] = lower_limit_perc
    dat[dat>upper_limit_perc] = upper_limit_perc
    return dat

def aggregateToFirmLevel(df_cusip_level, df_cusip_permno, col_name_to_be_aggregated):
    df_output = df_cusip_level.copy()
    df_output = df_output.merge(df_cusip_permno, left_on=["Date", "CUSIP"], right_on=["Date", "CUSIP"], how="left")
    df_output = df_output.groupby(["Date", "FirmID_PERMNO"])[col_name_to_be_aggregated].mean()
    df_output = df_output.reset_index()
    df_output["F_"+col_name_to_be_aggregated] = df_output[col_name_to_be_aggregated]
    del df_output[col_name_to_be_aggregated]
    return df_output




def meanExperiment(experiment_data, target_val):
    return np.sum(experiment_data==target_val)/len(experiment_data)*100



def generateShock(shock_base_points):
    return (shock_base_points/100/100)*100 # shock in percent

def generateShocked_ABR(shock, df_returns, df_mat_rat_group, df_benchmarks):
    df_abn_returns = df_returns.copy()
    df_abn_returns = df_abn_returns[["Date", "CUSIP", "R_t"]]
    df_abn_returns = df_abn_returns.merge(df_mat_rat_group, on=["Date", "CUSIP"])
    df_abn_returns = df_abn_returns.merge(df_benchmarks, left_on=["Date", "ratingGroup", "maturityGroup"], right_on=["Date", "ratingGroup", "maturityGroup"], how="left")
    df_abn_returns["ABR_t"] = (df_abn_returns["R_t"]+shock) - df_abn_returns["BM_t"]
    df_abn_returns = df_abn_returns.dropna()
    df_abn_returns = df_abn_returns[["Date", "CUSIP", "ABR_t", "ratingGroup", "maturityGroup"]]
    return df_abn_returns

def generateShocked_ABR_noise(shock, df_raw_sigma_n_t, df_returns, df_mat_rat_group, df_benchmarks,noise_fac=1.0):
    df_abn_returns = df_returns.copy()
    df_abn_returns = df_abn_returns[["Date", "CUSIP", "R_t"]]
    df_abn_returns = df_abn_returns.merge(df_mat_rat_group, on=["Date", "CUSIP"])
    df_abn_returns = df_abn_returns.merge(df_benchmarks, left_on=["Date", "ratingGroup", "maturityGroup"], right_on=["Date", "ratingGroup", "maturityGroup"], how="left")
    df_abn_returns = df_abn_returns.merge(df_raw_sigma_n_t, on=["Date", "CUSIP"], how="left")
    df_abn_returns["ABR_t"] = (df_abn_returns["R_t"]+np.random.normal(shock, noise_fac*df_abn_returns["SIGMA_N_T"])) - df_abn_returns["BM_t"]
    df_abn_returns = df_abn_returns.dropna()
    df_abn_returns = df_abn_returns[["Date", "CUSIP", "ABR_t", "ratingGroup", "maturityGroup"]]
    return df_abn_returns

def generateShocked_SSR(shock, df_returns, df_raw_sigma_n_t):
    df_srr = df_returns.copy()
    df_srr = df_srr.merge(df_raw_sigma_n_t, on=["Date", "CUSIP"], how="left")
    df_srr["SRR_t"] = (df_srr["R_t"]+shock) / df_srr["SIGMA_N_T"]
    df_srr = df_srr[["Date", "CUSIP", "SRR_t"]]
    df_srr = df_srr.dropna()
    return df_srr

def generateShocked_SSR_noise(shock, df_returns, df_raw_sigma_n_t, noiseFac=1.0):
    df_srr = df_returns.copy()
    df_srr = df_srr.merge(df_raw_sigma_n_t, on=["Date", "CUSIP"], how="left")
    df_srr["SRR_t"] = (df_srr["R_t"]+np.random.normal(shock, noiseFac*df_srr["SIGMA_N_T"])) / df_srr["SIGMA_N_T"]
    df_srr = df_srr[["Date", "CUSIP", "SRR_t"]]
    df_srr = df_srr.dropna()
    return df_srr

def generateShocked_SSR_pre(shock, df_returns, df_raw_pre_sigma_n_t):
    df_srr_pre = df_returns.copy()
    df_srr_pre = df_srr_pre.merge(df_raw_pre_sigma_n_t, on=["Date", "CUSIP"], how="left")
    df_srr_pre["SRR_pre_t"] = (df_srr_pre["R_t"]+shock) / df_srr_pre["SIGMA_N_T"]
    df_srr_pre = df_srr_pre[["Date", "CUSIP", "SRR_pre_t"]]
    df_srr_pre = df_srr_pre.dropna()
    return df_srr_pre

def generateShocked_SSR_pre_noise(shock, df_returns, df_raw_pre_sigma_n_t,noise_fac=1.0):
    df_srr_pre = df_returns.copy()
    df_srr_pre = df_srr_pre.merge(df_raw_pre_sigma_n_t, on=["Date", "CUSIP"], how="left")
    df_srr_pre["SRR_pre_t"] = (df_srr_pre["R_t"]+np.random.normal(shock, noise_fac*df_srr_pre["SIGMA_N_T"])) / df_srr_pre["SIGMA_N_T"]
    df_srr_pre = df_srr_pre[["Date", "CUSIP", "SRR_pre_t"]]
    df_srr_pre = df_srr_pre.dropna()
    return df_srr_pre

def generateShocked_SABR(shock, df_abn_returns, df_abn_sigma_n_t, WINSORIZE_THRESHOLD):
    df_sabr = df_abn_returns.copy()
    df_sabr = df_sabr.merge(df_abn_sigma_n_t, on=["Date", "CUSIP"], how="left")
    df_sabr["SABR_t"] = (df_sabr["ABR_t"] + shock) / df_sabr["SIGMA_N_T"]
    df_sabr = df_sabr[["Date", "CUSIP", "SABR_t"]]
    df_sabr = df_sabr.dropna()
    df_sabr["SABR_t"] = winsorize(df_sabr["SABR_t"], WINSORIZE_THRESHOLD)
    return df_sabr

def generateShocked_SABR_noise(shock, df_abn_returns, df_abn_sigma_n_t, WINSORIZE_THRESHOLD, noise_fac=1.0):
    df_sabr = df_abn_returns.copy()
    df_sabr = df_sabr.merge(df_abn_sigma_n_t, on=["Date", "CUSIP"], how="left")
    df_sabr["SABR_t"] = (df_sabr["ABR_t"] + np.random.normal(shock, noise_fac*df_sabr["SIGMA_N_T"])) / df_sabr["SIGMA_N_T"]
    df_sabr = df_sabr[["Date", "CUSIP", "SABR_t"]]
    df_sabr = df_sabr.dropna()
    df_sabr["SABR_t"] = winsorize(df_sabr["SABR_t"], WINSORIZE_THRESHOLD)
    return df_sabr

def generateShocked_ABSR(shock, df_returns, df_raw_sigma_n_t, df_mat_rat_group, df_std_benchmarks, WINSORIZE_THRESHOLD):
    df_absr = generateShocked_SSR(shock, df_returns, df_raw_sigma_n_t)
    df_absr = df_absr[["Date", "CUSIP", "SRR_t"]]
    df_absr = df_absr.merge(df_mat_rat_group, on=["Date", "CUSIP"])
    df_absr = df_absr.merge(df_std_benchmarks, left_on=["Date", "ratingGroup", "maturityGroup"], right_on=["Date", "ratingGroup", "maturityGroup"], how="left")
    df_absr["ABSR_t"] = df_absr["SRR_t"] - df_absr["SBM_t"]
    df_absr = df_absr.dropna()
    df_absr = df_absr[["Date", "CUSIP", "ABSR_t", "ratingGroup", "maturityGroup"]]
    df_absr["ABSR_t"] = winsorize(df_absr["ABSR_t"], WINSORIZE_THRESHOLD)
    return df_absr

def generateShocked_ABSR_noise(shock, df_returns, df_raw_sigma_n_t, df_mat_rat_group, df_std_benchmarks, WINSORIZE_THRESHOLD, noiseFac=1.0):
    df_absr = generateShocked_SSR_noise(shock, df_returns, df_raw_sigma_n_t, noiseFac)
    df_absr = df_absr[["Date", "CUSIP", "SRR_t"]]
    df_absr = df_absr.merge(df_mat_rat_group, on=["Date", "CUSIP"])
    df_absr = df_absr.merge(df_std_benchmarks, left_on=["Date", "ratingGroup", "maturityGroup"], right_on=["Date", "ratingGroup", "maturityGroup"], how="left")
    df_absr["ABSR_t"] = df_absr["SRR_t"] - df_absr["SBM_t"]
    df_absr = df_absr.dropna()
    df_absr = df_absr[["Date", "CUSIP", "ABSR_t", "ratingGroup", "maturityGroup"]]
    df_absr["ABSR_t"] = winsorize(df_absr["ABSR_t"], WINSORIZE_THRESHOLD)
    return df_absr

def generateShocked_ABSR_pre(shock, df_returns, df_raw_pre_sigma_n_t, df_mat_rat_group, df_std_benchmarks_pre, WINSORIZE_THRESHOLD):
    df_absr_pre = generateShocked_SSR_pre(shock, df_returns, df_raw_pre_sigma_n_t)
    df_absr_pre = df_absr_pre[["Date", "CUSIP", "SRR_pre_t"]]
    df_absr_pre = df_absr_pre.merge(df_mat_rat_group, on=["Date", "CUSIP"])
    df_absr_pre = df_absr_pre.merge(df_std_benchmarks_pre, left_on=["Date", "ratingGroup", "maturityGroup"], right_on=["Date", "ratingGroup", "maturityGroup"], how="left")
    df_absr_pre["ABSR_pre_t"] = df_absr_pre["SRR_pre_t"] - df_absr_pre["SBM_pre_t"]
    df_absr_pre = df_absr_pre.dropna()
    df_absr_pre = df_absr_pre[["Date", "CUSIP", "ABSR_pre_t", "ratingGroup", "maturityGroup"]]
    df_absr_pre["ABSR_pre_t"] = winsorize(df_absr_pre["ABSR_pre_t"], WINSORIZE_THRESHOLD)
    return df_absr_pre

def generateShocked_ABSR_pre_noise(shock, df_returns, df_raw_pre_sigma_n_t, df_mat_rat_group, df_std_benchmarks_pre, WINSORIZE_THRESHOLD, noise_fac=1.0):
    df_absr_pre = generateShocked_SSR_pre_noise(shock, df_returns, df_raw_pre_sigma_n_t,noise_fac)
    df_absr_pre = df_absr_pre[["Date", "CUSIP", "SRR_pre_t"]]
    df_absr_pre = df_absr_pre.merge(df_mat_rat_group, on=["Date", "CUSIP"])
    df_absr_pre = df_absr_pre.merge(df_std_benchmarks_pre, left_on=["Date", "ratingGroup", "maturityGroup"], right_on=["Date", "ratingGroup", "maturityGroup"], how="left")
    df_absr_pre["ABSR_pre_t"] = df_absr_pre["SRR_pre_t"] - df_absr_pre["SBM_pre_t"]
    df_absr_pre = df_absr_pre.dropna()
    df_absr_pre = df_absr_pre[["Date", "CUSIP", "ABSR_pre_t", "ratingGroup", "maturityGroup"]]
    df_absr_pre["ABSR_pre_t"] = winsorize(df_absr_pre["ABSR_pre_t"], WINSORIZE_THRESHOLD)
    return df_absr_pre

def getSampleWithDataForT4B(df_sample, df_values):
    df_sample_data = df_sample.copy()
    df_sample_data = df_sample_data.merge(df_values, on=["Date", "FirmID_PERMNO"], how="left")
    df_sample_data = df_sample_data.dropna()
    return df_sample_data

def getListElementRobust(lst, idx):
    if(idx<0):
        date = lst[0]
    elif(idx>len(lst)-1):
        date = lst[-1]
    else:
        date = lst[idx]
    return (date - pd.to_datetime(100000000, unit='s')).days
    
def determineSigmas(df_source, col_name, lst_date_interval_1_beg, lst_date_interval_1_end, lst_date_interval_2_beg, lst_date_interval_2_end, lst_cusips, lst_dates, SIGMA_N_AT_LEAST_OBSV, WINSORIZE_THRESHOLD):
    df_returns_selection = df_source.copy()
    df_returns_selection = df_returns_selection[["CUSIP", "Date", col_name]]
    df_returns_selection["DateX"] = pd.to_datetime(100000000, unit='s')
    df_returns_selection["DateX2"] = (df_returns_selection["Date"]-df_returns_selection["DateX"]).dt.days
    df_returns_selection["Date"] = df_returns_selection["DateX2"]
    df_returns_selection = df_returns_selection[["CUSIP", "Date", col_name]]
    data_sigma_n_t = []
    for specific_cusip in lst_cusips:#[:5]:
        print(col_name, specific_cusip, lst_cusips.index(specific_cusip), len(lst_cusips), 100*lst_cusips.index(specific_cusip)/len(lst_cusips))
        df_cusip_selection = df_returns_selection[df_returns_selection["CUSIP"]==specific_cusip]
        for idx_date in range(0, len(lst_dates)):
            df_selection = df_cusip_selection[ 
                    ((df_cusip_selection['Date'] > lst_date_interval_1_beg[idx_date]) & (df_cusip_selection['Date'] < lst_date_interval_1_end[idx_date]))
                  | ((df_cusip_selection['Date'] > lst_date_interval_2_beg[idx_date]) & (df_cusip_selection['Date'] < lst_date_interval_2_end[idx_date]))]
            if(len(df_selection)>SIGMA_N_AT_LEAST_OBSV):
                winsorized_std = np.std(winsorize(df_selection[col_name], WINSORIZE_THRESHOLD))
                data_sigma_n_t.append([lst_dates[idx_date], specific_cusip,  winsorized_std])
    del df_returns_selection
    df_sigma_n_t = pd.DataFrame(data_sigma_n_t, columns=["Date", "CUSIP", "SIGMA_N_T"])
    return df_sigma_n_t
    
def calculateSigmas(lst_dates, df_returns, df_abn_returns, SIGMA_TIME_BOUNDARY_1, SIGMA_TIME_BOUNDARY_2, SIGMA_TIME_BOUNDARY_3, 
                    SIGMA_TIME_BOUNDARY_4, SIGMA_TIME_BOUNDARY_5, SIGMA_TIME_BOUNDARY_6, lst_cusips, SIGMA_N_AT_LEAST_OBSV, WINSORIZE_THRESHOLD):
    # Prepare dates
    lst_date_interval_1_beg = []
    lst_date_interval_1_end = []
    lst_date_interval_2_beg = []
    lst_date_interval_2_end = []
    lst_date_interval_3_beg = []
    lst_date_interval_3_end = []
    for idx_date in range(0, len(lst_dates)):
        lst_date_interval_1_beg.append(getListElementRobust(lst_dates, idx_date - SIGMA_TIME_BOUNDARY_1))
        lst_date_interval_1_end.append(getListElementRobust(lst_dates, idx_date - SIGMA_TIME_BOUNDARY_2))
        lst_date_interval_2_beg.append(getListElementRobust(lst_dates, idx_date + SIGMA_TIME_BOUNDARY_3))
        lst_date_interval_2_end.append(getListElementRobust(lst_dates, idx_date + SIGMA_TIME_BOUNDARY_4))
        lst_date_interval_3_beg.append(getListElementRobust(lst_dates, idx_date - SIGMA_TIME_BOUNDARY_5))
        lst_date_interval_3_end.append(getListElementRobust(lst_dates, idx_date - SIGMA_TIME_BOUNDARY_6))
    # Determine Standard Deviations of Abnormal and Raw Returns
    df_raw_sigma_n_t = determineSigmas(df_returns,     "R_t",   lst_date_interval_1_beg, lst_date_interval_1_end, lst_date_interval_2_beg, lst_date_interval_2_end, lst_cusips, lst_dates, SIGMA_N_AT_LEAST_OBSV, WINSORIZE_THRESHOLD)
    print("Finished df_raw_sigma_n_t")
    df_abn_sigma_n_t = determineSigmas(df_abn_returns, "ABR_t", lst_date_interval_1_beg, lst_date_interval_1_end, lst_date_interval_2_beg, lst_date_interval_2_end, lst_cusips, lst_dates, SIGMA_N_AT_LEAST_OBSV, WINSORIZE_THRESHOLD)
    print("Finished df_abn_sigma_n_t")
    df_raw_pre_sigma_n_t = determineSigmas(df_returns, "R_t",   lst_date_interval_3_beg, lst_date_interval_3_end, lst_date_interval_3_beg, lst_date_interval_3_end, lst_cusips, lst_dates, SIGMA_N_AT_LEAST_OBSV, WINSORIZE_THRESHOLD)
    print("Finished df_raw_pre_sigma_n_t")
    return df_raw_sigma_n_t, df_abn_sigma_n_t, df_raw_pre_sigma_n_t

def getListElementRobust_Week(lst, idx):
    if(idx<0):
        date = lst[0]
    elif(idx>len(lst)-1):
        date = lst[-1]
    else:
        date = lst[idx]
    return date
    
def determineSigmas_Week(df_source, lst_weeks, col_name, lst_date_interval_1_beg, lst_date_interval_1_end, lst_date_interval_2_beg, lst_date_interval_2_end, lst_cusips, SIGMA_N_AT_LEAST_OBSV, WINSORIZE_THRESHOLD):
    df_returns_selection = df_source.copy()
    df_returns_selection = df_returns_selection[["CUSIP", "Date", col_name]]
    data_sigma_n_t = []
    for specific_cusip in lst_cusips:#[:5]:
        print(col_name, specific_cusip, lst_cusips.index(specific_cusip), len(lst_cusips), 100*lst_cusips.index(specific_cusip)/len(lst_cusips))
        df_cusip_selection = df_returns_selection[df_returns_selection["CUSIP"]==specific_cusip]
        for idx_week in range(0, len(lst_weeks)):
            df_selection = df_cusip_selection[ 
                    ((df_cusip_selection['Date'] > lst_date_interval_1_beg[idx_week]) & (df_cusip_selection['Date'] < lst_date_interval_1_end[idx_week]))
                  | ((df_cusip_selection['Date'] > lst_date_interval_2_beg[idx_week]) & (df_cusip_selection['Date'] < lst_date_interval_2_end[idx_week]))]
            if(len(df_selection)>SIGMA_N_AT_LEAST_OBSV):
                winsorized_std = np.std(winsorize(df_selection[col_name], WINSORIZE_THRESHOLD))
                data_sigma_n_t.append([lst_weeks[idx_week], specific_cusip,  winsorized_std])
    del df_returns_selection
    df_sigma_n_t = pd.DataFrame(data_sigma_n_t, columns=["Date", "CUSIP", "SIGMA_N_T"])
    return df_sigma_n_t
    
def calculateSigmasWeeks(df_returns, df_abn_returns, SIGMA_WEEK_BOUNDARY_1, SIGMA_WEEK_BOUNDARY_2, SIGMA_WEEK_BOUNDARY_3, 
                    SIGMA_WEEK_BOUNDARY_4, SIGMA_WEEK_BOUNDARY_5, SIGMA_WEEK_BOUNDARY_6, lst_cusips, SIGMA_N_AT_LEAST_OBSV, WINSORIZE_THRESHOLD):
    lst_weeks = list(set(df_returns["Date"].tolist()))
    lst_weeks.sort()
    # Prepare dates
    lst_week_interval_1_beg = []
    lst_week_interval_1_end = []
    lst_week_interval_2_beg = []
    lst_week_interval_2_end = []
    lst_week_interval_3_beg = []
    lst_week_interval_3_end = []
    for idx_week in range(0, len(lst_weeks)):
        lst_week_interval_1_beg.append(getListElementRobust_Week(lst_weeks, idx_week - SIGMA_WEEK_BOUNDARY_1))
        lst_week_interval_1_end.append(getListElementRobust_Week(lst_weeks, idx_week - SIGMA_WEEK_BOUNDARY_2))
        lst_week_interval_2_beg.append(getListElementRobust_Week(lst_weeks, idx_week + SIGMA_WEEK_BOUNDARY_3))
        lst_week_interval_2_end.append(getListElementRobust_Week(lst_weeks, idx_week + SIGMA_WEEK_BOUNDARY_4))
        lst_week_interval_3_beg.append(getListElementRobust_Week(lst_weeks, idx_week - SIGMA_WEEK_BOUNDARY_5))
        lst_week_interval_3_end.append(getListElementRobust_Week(lst_weeks, idx_week - SIGMA_WEEK_BOUNDARY_6))
    # Determine Standard Deviations of Abnormal and Raw Returns
    df_raw_sigma_n_t = determineSigmas_Week(df_returns, lst_weeks,     "R_t",   lst_week_interval_1_beg, lst_week_interval_1_end, lst_week_interval_2_beg, lst_week_interval_2_end, lst_cusips, SIGMA_N_AT_LEAST_OBSV, WINSORIZE_THRESHOLD)
    print("Finished df_raw_sigma_n_t")
    df_abn_sigma_n_t = determineSigmas_Week(df_abn_returns, lst_weeks, "ABR_t", lst_week_interval_1_beg, lst_week_interval_1_end, lst_week_interval_2_beg, lst_week_interval_2_end, lst_cusips, SIGMA_N_AT_LEAST_OBSV, WINSORIZE_THRESHOLD)
    print("Finished df_abn_sigma_n_t")
    df_raw_pre_sigma_n_t = determineSigmas_Week(df_returns, lst_weeks, "R_t",   lst_week_interval_3_beg, lst_week_interval_3_end, lst_week_interval_3_beg, lst_week_interval_3_end, lst_cusips, SIGMA_N_AT_LEAST_OBSV, WINSORIZE_THRESHOLD)
    print("Finished df_raw_pre_sigma_n_t")
    return df_raw_sigma_n_t, df_abn_sigma_n_t, df_raw_pre_sigma_n_t

