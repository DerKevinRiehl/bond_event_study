###############################################################################
# Imports
###############################################################################
import numpy as np
import pandas as pd
import datetime
import random
import string
from scipy.stats import kurtosis, skew, wilcoxon
from scipy import stats
from statsmodels.stats.descriptivestats import sign_test




###############################################################################
# Methods
###############################################################################
def generateDateFromString(s):
    return  datetime.datetime.strptime(s, '%Y%m%d')

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

def getListOfRandomCusips(num_cusips):
    N_DIGITS_CUSIPS = 9
    lst_cusips = []
    for i in range(0, num_cusips):
        lst_cusips.append("".join(random.choices(string.ascii_uppercase + string.digits, k=N_DIGITS_CUSIPS)))
    return lst_cusips

def loadListOfCusips(file, num=-1):
    f = open(file, "r")
    content = f.read()
    f.close()
    cusips = content.split("\n")
    if(num==-1):
        return cusips
    else:
        return cusips[1:num+1]

def randomlyGenerateMaturityDatesForCUSIPS(start_date, lst_cusips):
    dict_cusip_maturity_dates = {}
    for cusip in lst_cusips:
        random_maturity_group = random.randint(1, 4)
        if(random_maturity_group==1):   # 1 to 3 years
            random_maturity_date = start_date + datetime.timedelta(days=365 + random.randint(50, 100))
        elif(random_maturity_group==2): # 3 to 5 years
            random_maturity_date = start_date + datetime.timedelta(days=365*4 + random.randint(50, 100))
        elif(random_maturity_group==3): # 5 to 10 years
            random_maturity_date = start_date + datetime.timedelta(days=365*6 + random.randint(50, 100))
        else:                           # over ten years
            random_maturity_date = start_date + datetime.timedelta(days=365*12 + random.randint(50, 100))
        dict_cusip_maturity_dates[cusip] = random_maturity_date
    return dict_cusip_maturity_dates

def randomlyGenerateRatingPerCusipDate(lst_cusips, lst_dates):
    RATING_MAX = 22
    RATING_MIN = 1
    dict_rating = {}
    for cusip in lst_cusips:
        start_rating = random.randint(RATING_MIN, RATING_MAX)
        current_rating = start_rating
        dict_dates = {}
        for date in lst_dates:
            dict_dates[date] = current_rating
            if(random.randint(1, 2)==1):
                current_rating += 1
                if(current_rating>RATING_MAX):
                    current_rating = RATING_MAX
            else:
                current_rating -= 1
                if(current_rating<RATING_MIN):
                    current_rating = RATING_MIN
        dict_rating[cusip] = dict_dates
    return dict_rating

def randomlyGeneratePricesPerCusipDate(lst_cusips, lst_dates):
    dict_prices = {}
    for cusip in lst_cusips:
        PRICE_MIN = random.uniform(10, 50)
        PRICE_MAX = random.uniform(60, 100)
        VOLATILITY_UP = random.uniform(1.01, 1.20)
        VOLATILITY_DN = random.uniform(1.01, 1.20)
        start_price = random.uniform(PRICE_MIN, PRICE_MAX)
        current_price = start_price
        dict_dates = {}
        for date in lst_dates:
            dict_dates[date] = current_price
            if(random.randint(1, 2)==1):
                current_price = current_price*VOLATILITY_UP
                if(current_price>PRICE_MAX):
                    current_price = PRICE_MAX
            else:
                current_price = current_price/VOLATILITY_DN
                if(current_price<PRICE_MIN):
                    current_price = PRICE_MIN
        dict_prices[cusip] = dict_dates
    return dict_prices

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
    if(row["remaining_days_until_maturity"]<365):       # below one year
        return np.nan
    elif(row["remaining_days_until_maturity"]<=3*365):  # 1-3 years
        return 0
    elif(row["remaining_days_until_maturity"]<=5*365):  # 3-5 years
        return 1
    elif(row["remaining_days_until_maturity"]<=10*365): # 5-10 years
        return 2
    else:                                                    # Over 10 years
        return 3
    
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

def weighted_average(df, values, weights):
    return sum(df[values] * df[weights]) / df[weights].sum()

def getDaysFromDateTimeDifference(row):
    return row["remaining_days_until_maturity"].days

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

def doTtest(pop):
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

def doSRtest(pop):
    sr_test_ts = wilcoxon(pop, alternative='two-sided')
    experiment_p = sr_test_ts[1]
    experiment_res = 0
    if(experiment_p<SIGNIFICANCE_LEVEL):
        if(np.median(pop)>0):
            experiment_res = +1
        else:
            experiment_res = -1
    return experiment_res

def doSGtest(pop):
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

def generateShocked_SABR(shock, df_abn_returns, df_abn_sigma_n_t):
    df_sabr = df_abn_returns.copy()
    df_sabr = df_sabr.merge(df_abn_sigma_n_t, on=["Date", "CUSIP"], how="left")
    df_sabr["SABR_t"] = (df_sabr["ABR_t"] + shock) / df_sabr["SIGMA_N_T"]
    df_sabr = df_sabr[["Date", "CUSIP", "SABR_t"]]
    df_sabr = df_sabr.dropna()
    df_sabr["SABR_t"] = winsorize(df_sabr["SABR_t"], WINSORIZE_THRESHOLD)
    return df_sabr

def generateShocked_SABR_noise(shock, df_abn_returns, df_abn_sigma_n_t,noise_fac=1.0):
    df_sabr = df_abn_returns.copy()
    df_sabr = df_sabr.merge(df_abn_sigma_n_t, on=["Date", "CUSIP"], how="left")
    df_sabr["SABR_t"] = (df_sabr["ABR_t"] + np.random.normal(shock, noise_fac*df_sabr["SIGMA_N_T"])) / df_sabr["SIGMA_N_T"]
    df_sabr = df_sabr[["Date", "CUSIP", "SABR_t"]]
    df_sabr = df_sabr.dropna()
    df_sabr["SABR_t"] = winsorize(df_sabr["SABR_t"], WINSORIZE_THRESHOLD)
    return df_sabr

def generateShocked_ABSR(shock, df_returns, df_raw_sigma_n_t, df_mat_rat_group, df_std_benchmarks):
    df_absr = generateShocked_SSR(shock, df_returns, df_raw_sigma_n_t)
    df_absr = df_absr[["Date", "CUSIP", "SRR_t"]]
    df_absr = df_absr.merge(df_mat_rat_group, on=["Date", "CUSIP"])
    df_absr = df_absr.merge(df_std_benchmarks, left_on=["Date", "ratingGroup", "maturityGroup"], right_on=["Date", "ratingGroup", "maturityGroup"], how="left")
    df_absr["ABSR_t"] = df_absr["SRR_t"] - df_absr["SBM_t"]
    df_absr = df_absr.dropna()
    df_absr = df_absr[["Date", "CUSIP", "ABSR_t", "ratingGroup", "maturityGroup"]]
    df_absr["ABSR_t"] = winsorize(df_absr["ABSR_t"], WINSORIZE_THRESHOLD)
    return df_absr

def generateShocked_ABSR_noise(shock, df_returns, df_raw_sigma_n_t, df_mat_rat_group, df_std_benchmarks, noiseFac=1.0):
    df_absr = generateShocked_SSR_noise(shock, df_returns, df_raw_sigma_n_t, noiseFac)
    df_absr = df_absr[["Date", "CUSIP", "SRR_t"]]
    df_absr = df_absr.merge(df_mat_rat_group, on=["Date", "CUSIP"])
    df_absr = df_absr.merge(df_std_benchmarks, left_on=["Date", "ratingGroup", "maturityGroup"], right_on=["Date", "ratingGroup", "maturityGroup"], how="left")
    df_absr["ABSR_t"] = df_absr["SRR_t"] - df_absr["SBM_t"]
    df_absr = df_absr.dropna()
    df_absr = df_absr[["Date", "CUSIP", "ABSR_t", "ratingGroup", "maturityGroup"]]
    df_absr["ABSR_t"] = winsorize(df_absr["ABSR_t"], WINSORIZE_THRESHOLD)
    return df_absr

def generateShocked_ABSR_pre(shock, df_returns, df_raw_pre_sigma_n_t, df_mat_rat_group, df_std_benchmarks_pre):
    df_absr_pre = generateShocked_SSR_pre(shock, df_returns, df_raw_pre_sigma_n_t)
    df_absr_pre = df_absr_pre[["Date", "CUSIP", "SRR_pre_t"]]
    df_absr_pre = df_absr_pre.merge(df_mat_rat_group, on=["Date", "CUSIP"])
    df_absr_pre = df_absr_pre.merge(df_std_benchmarks_pre, left_on=["Date", "ratingGroup", "maturityGroup"], right_on=["Date", "ratingGroup", "maturityGroup"], how="left")
    df_absr_pre["ABSR_pre_t"] = df_absr_pre["SRR_pre_t"] - df_absr_pre["SBM_pre_t"]
    df_absr_pre = df_absr_pre.dropna()
    df_absr_pre = df_absr_pre[["Date", "CUSIP", "ABSR_pre_t", "ratingGroup", "maturityGroup"]]
    df_absr_pre["ABSR_pre_t"] = winsorize(df_absr_pre["ABSR_pre_t"], WINSORIZE_THRESHOLD)
    return df_absr_pre

def generateShocked_ABSR_pre_noise(shock, df_returns, df_raw_pre_sigma_n_t, df_mat_rat_group, df_std_benchmarks_pre,noise_fac=1.0):
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
def determineSigmas(df_source, col_name, lst_date_interval_1_beg, lst_date_interval_1_end, lst_date_interval_2_beg, lst_date_interval_2_end):
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
                    SIGMA_TIME_BOUNDARY_4, SIGMA_TIME_BOUNDARY_5, SIGMA_TIME_BOUNDARY_6):
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
    df_raw_sigma_n_t = determineSigmas(df_returns,     "R_t",   lst_date_interval_1_beg, lst_date_interval_1_end, lst_date_interval_2_beg, lst_date_interval_2_end)
    print("Finished df_raw_sigma_n_t")
    df_abn_sigma_n_t = determineSigmas(df_abn_returns, "ABR_t", lst_date_interval_1_beg, lst_date_interval_1_end, lst_date_interval_2_beg, lst_date_interval_2_end)
    print("Finished df_abn_sigma_n_t")
    df_raw_pre_sigma_n_t = determineSigmas(df_returns, "R_t",   lst_date_interval_3_beg, lst_date_interval_3_end, lst_date_interval_3_beg, lst_date_interval_3_end)
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
def determineSigmas_Week(df_source, lst_weeks, col_name, lst_date_interval_1_beg, lst_date_interval_1_end, lst_date_interval_2_beg, lst_date_interval_2_end):
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
                    SIGMA_WEEK_BOUNDARY_4, SIGMA_WEEK_BOUNDARY_5, SIGMA_WEEK_BOUNDARY_6):
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
    df_raw_sigma_n_t = determineSigmas_Week(df_returns, lst_weeks,     "R_t",   lst_week_interval_1_beg, lst_week_interval_1_end, lst_week_interval_2_beg, lst_week_interval_2_end)
    print("Finished df_raw_sigma_n_t")
    df_abn_sigma_n_t = determineSigmas_Week(df_abn_returns, lst_weeks, "ABR_t", lst_week_interval_1_beg, lst_week_interval_1_end, lst_week_interval_2_beg, lst_week_interval_2_end)
    print("Finished df_abn_sigma_n_t")
    df_raw_pre_sigma_n_t = determineSigmas_Week(df_returns, lst_weeks, "R_t",   lst_week_interval_3_beg, lst_week_interval_3_end, lst_week_interval_3_beg, lst_week_interval_3_end)
    print("Finished df_raw_pre_sigma_n_t")
    return df_raw_sigma_n_t, df_abn_sigma_n_t, df_raw_pre_sigma_n_t


###############################################################################
# Parameters from Paper
###############################################################################
    # Technical & Data
SYNTHETIC_DATA = False
CUSIP_FILTER = "Input_Data/cusips_sample_filter.csv"
CUSIP_FILTER2 = None # None # "Input_Data/FilterCUSIPSIC6.csv"
# RAW_DATA_SRC = "Input_Data/trace_daily_r_non_NA_new_ALL_ex_notin.csv"
# RAW_DATA_SRC = "Input_Data/trace_daily_r_non_NA_new_ALL_ex_100k.csv"
RAW_DATA_SRC = "Input_Data/trace_daily_r_non_NA_new_ALL_ex_50k.csv"
# RAW_DATA_SRC = "Input_Data/trace_daily_r_non_NA_new_50k_07_2013.csv"
# RAW_DATA_SRC = "Input_Data/trace_daily_r_non_NA_100k_new.csv"
# RAW_DATA_SRC = "Input_Data/trace_daily_r_non_NA_all_trades.csv"
    # NOISE
NOISE = False # True # False
BOEHMER_T_TEST = False # True # False
    # WEEK BASE
WEEK_BASIS = False # True # False
    # General
FILTER_MIN_NUMBER_CUSIPS_IN_BENCHMARK = 5
FILTER_MINIMUM_NUMBER_OF_DAYS_TO_MATURITY = 365
    # Standard Deviation
SIGMA_TIME_BOUNDARY_1 = 55 # for ABSR AND SABR
SIGMA_TIME_BOUNDARY_2 = 6
SIGMA_TIME_BOUNDARY_3 = 6
SIGMA_TIME_BOUNDARY_4 = 55
SIGMA_TIME_BOUNDARY_5 = 101 # for pre-ABSR
SIGMA_TIME_BOUNDARY_6 = 6
SIGMA_N_AT_LEAST_OBSV = 6

SIGMA_WEEK_BOUNDARY_1 = 11 # for ABSR AND SABR
SIGMA_WEEK_BOUNDARY_2 = 2
SIGMA_WEEK_BOUNDARY_3 = 2
SIGMA_WEEK_BOUNDARY_4 = 11
SIGMA_WEEK_BOUNDARY_5 = 22 # for pre-ABSR
SIGMA_WEEK_BOUNDARY_6 = 2
    # Table 2 - Monte Carlo
N_SAMPLE = 300
N_MONTE_CARLO_EXPERIMENT_REPETITION = 10000
SIGNIFICANCE_LEVEL = 0.01 # 0.05 # 0.01
SHOCK_BASE_POINTS_T2 = 15
    # Table 3- ABSR and SABR
WINSORIZE_THRESHOLD = 0.01
    # Table 4 - Panel C
SHOCK_BASE_POINTS_T4C = 15
SHOCK_BASE_POINTS_T4D1 = 10
SHOCK_BASE_POINTS_T4D2 = 25
SHOCK_BASE_POINTS_T5 = 15




###############################################################################
# Generate Synthetic Input Data
"""
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
"""
###############################################################################
if(SYNTHETIC_DATA):
        # Generation Parameters
    start_date = generateDateFromString("20100101")
    number_of_days = 365*3
    number_of_cusips = 100
    number_of_permnos = 20
    number_of_trades_per_day_per_cusip_min = 5
    number_of_trades_per_day_per_cusip_max = 20
        # Prepare Lists
    lst_cusips = getListOfRandomCusips(number_of_cusips)
    lst_permnos = getListOfRandomCusips(number_of_permnos)
    #lst_cusips = loadListOfCusips("Input_Data/tidyfinance_cusips_WRDS.csv", number_of_cusips)
    lst_dates = getListOfDays(start_date, number_of_days)
    dict_cusip_maturity_dates = randomlyGenerateMaturityDatesForCUSIPS(start_date, lst_cusips)
    dict_ratings = randomlyGenerateRatingPerCusipDate(lst_cusips, lst_dates)
    dict_prices = randomlyGeneratePricesPerCusipDate(lst_cusips, lst_dates)
        # Generate Random DataFrame
    ls_data = []
    for cusip in lst_cusips:
        permno = random.choice(lst_permnos)
        for date in lst_dates:
            n_trades = random.randint(number_of_trades_per_day_per_cusip_min, number_of_trades_per_day_per_cusip_max)
            for i in range(0, n_trades):
                lst_entry = []
                lst_entry.append(date)
                lst_entry.append(cusip)
                lst_entry.append(permno)
                lst_entry.append(dict_cusip_maturity_dates[cusip])
                lst_entry.append(dict_ratings[cusip][date])
                lst_entry.append(dict_prices[cusip][date] * random.uniform(0.9, 1.1))
                lst_entry.append(random.randint(1, 100))
                ls_data.append(lst_entry)
    df_input = pd.DataFrame(ls_data, columns=["Date", "CUSIP", "FirmID_PERMNO", "MaturityDate", "Rating_numeric", "Price_USD", "Qty"])
    df_input["remaining_days_until_maturity"] = df_input["MaturityDate"] - df_input["Date"]
    df_input["remaining_days_until_maturity_int"] = df_input.apply(getDaysFromDateTimeDifference, axis=1)
    df_input = df_input[df_input["remaining_days_until_maturity_int"]>FILTER_MINIMUM_NUMBER_OF_DAYS_TO_MATURITY]
    del df_input["remaining_days_until_maturity"] 
    del df_input["remaining_days_until_maturity_int"] 
        # Generate Empty Events
    df_events = pd.DataFrame([], columns=["Date", "CUSIP"])
    df_events["TO_BE_EXCLUDED"] = 1
        # Generate CUSIP / PERMNO Loopup Table
    df_cusip_permno = df_input[["Date", "CUSIP", "FirmID_PERMNO"]]
    df_cusip_permno = df_cusip_permno.drop_duplicates()

    # experiment_montecarlo_selection
    if WEEK_BASIS:
        lst_weeks = []
        for d in lst_dates:
            lst_weeks.append("{0:0=4d}".format(d.isocalendar()[0])+"-"+"{0:0=2d}".format(d.isocalendar()[1]))
        experiment_montecarlo_selection = []
        for i in range(0,N_MONTE_CARLO_EXPERIMENT_REPETITION):
            random_firms = [random.choice(lst_permnos) for x in range(0,N_SAMPLE)]
            random_days = [random.choice(lst_weeks) for x in range(0,N_SAMPLE)]
            data = {}
            data["firms"] = random_firms
            data["days"] = random_days
            experiment_montecarlo_selection.append(data)
    else:   
        experiment_montecarlo_selection = []
        for i in range(0,N_MONTE_CARLO_EXPERIMENT_REPETITION):
            random_firms = [random.choice(lst_permnos) for x in range(0,N_SAMPLE)]
            random_days = [random.choice(lst_dates) for x in range(0,N_SAMPLE)]
            data = {}
            data["firms"] = random_firms
            data["days"] = random_days
            experiment_montecarlo_selection.append(data)



###############################################################################
# Load Real Data
"""
 - df_raw (provided by lukas müller)
     ["trd_exctn_dt", "cusip_id", "PERMNO", "maturity", "RTNG", "TWP", "vol_day"]
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
"""
###############################################################################
if(not SYNTHETIC_DATA):
    # Load raw data
    df_raw_data = pd.read_csv(RAW_DATA_SRC)
    df_raw_data = df_raw_data[["trd_exctn_dt", "cusip_id", "PERMNO", "maturity", "RTNG", "TWP", "vol_day"]]
    df_raw_data = df_raw_data.rename(columns={"trd_exctn_dt":"Date", "cusip_id":"CUSIP", "PERMNO":"FirmID_PERMNO", "maturity":"remaining_days_until_maturity", "RTNG":"Rating_numeric", "TWP":"TRADE_PRICE", "vol_day":"TRADE_VOLUME"})
    df_raw_data["Date"] = pd.to_datetime(df_raw_data['Date'], errors="coerce")
    df_raw_data = df_raw_data.drop_duplicates()
    df_raw_data = df_raw_data[~df_raw_data["FirmID_PERMNO"].isna()]
    # df_raw_data = df_raw_data[df_raw_data["Date"]<"2017-01-01 00:00:00"]
    
    if(CUSIP_FILTER is not None):
        df_cusip_filter = pd.read_csv(CUSIP_FILTER)
        del df_cusip_filter["Unnamed: 0"]
        df_raw_data = df_raw_data.merge(df_cusip_filter, left_on=["CUSIP"], right_on=["cusip"], how="left")
        df_raw_data = df_raw_data[~df_raw_data["cusip"].isna()]
        del df_raw_data["cusip"]
    
    if(CUSIP_FILTER2 is not None):
        df_cusip_filter = pd.read_csv(CUSIP_FILTER2)
        df_raw_data = df_raw_data.merge(df_cusip_filter, on=["CUSIP"], how="left")
        df_raw_data = df_raw_data[df_raw_data["sic_code"].isna()]
        del df_raw_data["sic_code"]

    
    # Generate CUSIP / PERMNO Loopup Table
    df_cusip_permno = df_raw_data[["Date", "CUSIP", "FirmID_PERMNO"]]
    df_cusip_permno = df_cusip_permno.drop_duplicates()
    df_cusip_permno = df_cusip_permno.dropna()
    
    # Prepare List
    lst_cusips = df_raw_data[["CUSIP"]].dropna().drop_duplicates()["CUSIP"].tolist()
    lst_permnos = df_raw_data[["FirmID_PERMNO"]].dropna().drop_duplicates()["FirmID_PERMNO"].tolist()
    lst_dates = df_raw_data[["Date"]].dropna().drop_duplicates()["Date"].tolist()
    
    # Generate Empty Events
    df_events = pd.DataFrame([], columns=["Date", "CUSIP"])
    df_events["TO_BE_EXCLUDED"] = 1
    
    # Determine Bond Maturity and Rating Groups
    df_mat_rat_group = df_raw_data.groupby(["Date", "CUSIP", "remaining_days_until_maturity", "Rating_numeric"]).mean()
    df_mat_rat_group = df_mat_rat_group.reset_index()
    del df_mat_rat_group["TRADE_PRICE"]
    del df_mat_rat_group["TRADE_VOLUME"]
    df_mat_rat_group = df_mat_rat_group.drop_duplicates()
    df_mat_rat_group["ratingGroup"] = df_mat_rat_group.apply(determineRatingGroup, axis=1)
    df_mat_rat_group["maturityGroup"] = df_mat_rat_group.apply(determineMaturityGroup_raw, axis=1)
    df_mat_rat_group = df_mat_rat_group[["Date", "CUSIP", "ratingGroup", "maturityGroup"]]
    df_mat_rat_group = df_mat_rat_group.dropna()
    
    ## P_n,t                 >>  Calculate Bond Prices for each CUSIP and day 
    df_prices = df_raw_data.copy()
    df_prices = df_prices[["Date", "CUSIP", "TRADE_PRICE"]]
    df_prices["Price_WA"] = df_prices["TRADE_PRICE"]
    del df_prices["TRADE_PRICE"]

    # experiment_montecarlo_selection
    experiment_montecarlo_selection = []
    for i in range(0,N_MONTE_CARLO_EXPERIMENT_REPETITION):
        random_firms = [random.choice(lst_permnos) for x in range(0,N_SAMPLE)]
        random_days = [random.choice(lst_dates) for x in range(0,N_SAMPLE)]
        data = {}
        data["firms"] = random_firms
        data["days"] = random_days
        experiment_montecarlo_selection.append(data)
        
        


###############################################################################
# Generate Supporting Tables
###############################################################################
"""
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
""" 
###############################################################################
if(SYNTHETIC_DATA):
    # Determine Bond Maturity and Rating Groups
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
    
    # P_n,t                 >>  Calculate Bond Prices for each CUSIP and day 
    df_prices = df_input.groupby(["Date", "CUSIP"]).apply(weighted_average, 'Price_USD', 'Qty')
    df_prices = df_prices.reset_index()
    df_prices["Price_WA"] = df_prices[0]
    del df_prices[0]




# R_t = R(t-1, t+1)_n    >>  Calculate Bond Returns for each CUSIP and day

if(WEEK_BASIS):
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


# BM_t = BM(t-1,t+1)_n  >> Calculate Benchmark Returns_t
df_benchmarks = df_returns.copy()
df_benchmarks = df_benchmarks[["Date", "R_t", "CUSIP"]]
df_benchmarks = df_benchmarks.merge(df_mat_rat_group, on=["Date", "CUSIP"])
df_benchmarks = df_benchmarks[["Date", "CUSIP", "R_t", "ratingGroup", "maturityGroup"]]
df_benchmarks = df_benchmarks.merge(df_events, on=["Date", "CUSIP"], how="left")
df_benchmarks = df_benchmarks[~df_benchmarks["TO_BE_EXCLUDED"].notnull()]
del df_benchmarks["TO_BE_EXCLUDED"]
df_benchmarks = df_benchmarks.groupby(["Date", "ratingGroup", "maturityGroup"]).agg(count=("CUSIP", "size"), mean=("R_t", "mean"))
df_benchmarks = df_benchmarks.reset_index()
df_benchmarks = df_benchmarks[df_benchmarks["count"]>=FILTER_MIN_NUMBER_CUSIPS_IN_BENCHMARK]
del df_benchmarks["count"]
df_benchmarks["BM_t"] = df_benchmarks["mean"]
del df_benchmarks["mean"]
df_benchmarks = df_benchmarks.dropna()
print("Finished df_benchmarks")


# ABR_t = ABR(t-1, t+1)_n  >>  Calculate Abnormal Bond Returns for each CUSIP and day
df_abn_returns = generateShocked_ABR(0, df_returns, df_mat_rat_group, df_benchmarks)
print("Finished df_abn_returns")


# F_ABR_t = ABR(t-1, t+1)_n  >>  Calculate Firm Abnormal Returns for each PERMNO and day
df_abn_returns_firms = aggregateToFirmLevel(df_abn_returns, df_cusip_permno, "ABR_t")
print("Finished df_abn_returns_firms")








"""
# ABN_SIGMA_N_T >> Standard deviations of abnormal returns of bond N at date t
# RAW_SIGMA_N_T >> Standard deviations of raw returns of bond N at date t
if WEEK_BASIS:
    df_raw_sigma_n_t, df_abn_sigma_n_t, df_raw_pre_sigma_n_t = calculateSigmasWeeks(df_returns, df_abn_returns, SIGMA_WEEK_BOUNDARY_1, SIGMA_WEEK_BOUNDARY_2, SIGMA_WEEK_BOUNDARY_3, 
                        SIGMA_WEEK_BOUNDARY_4, SIGMA_WEEK_BOUNDARY_5, SIGMA_WEEK_BOUNDARY_6)
    df_raw_sigma_n_t.to_csv("df_raw_sigma_n_t_new_EX50_SEP2023_weeks.csv")
    df_abn_sigma_n_t.to_csv("df_abn_sigma_n_t_new_EX50_SEP2023_weeks.csv")
    df_raw_pre_sigma_n_t.to_csv("df_raw_pre_sigma_n_t_new_EX50_SEP2023_weeks.csv")
else:
    df_raw_sigma_n_t, df_abn_sigma_n_t, df_raw_pre_sigma_n_t = calculateSigmas(lst_dates, df_returns, df_abn_returns, SIGMA_TIME_BOUNDARY_1, SIGMA_TIME_BOUNDARY_2, SIGMA_TIME_BOUNDARY_3, 
                        SIGMA_TIME_BOUNDARY_4, SIGMA_TIME_BOUNDARY_5, SIGMA_TIME_BOUNDARY_6)
    df_raw_sigma_n_t.to_csv("df_raw_sigma_n_t_new_EX50_SEP2023.csv")
    df_abn_sigma_n_t.to_csv("df_abn_sigma_n_t_new_EX50_SEP2023.csv")
    df_raw_pre_sigma_n_t.to_csv("df_raw_pre_sigma_n_t_new_EX50_SEP2023.csv")
"""


if WEEK_BASIS:
    df_raw_sigma_n_t = pd.read_csv("df_raw_sigma_n_t_new_EX50_SEP2023_weeks.csv")
    df_abn_sigma_n_t = pd.read_csv("df_abn_sigma_n_t_new_EX50_SEP2023_weeks.csv")
    df_raw_pre_sigma_n_t = pd.read_csv("df_raw_pre_sigma_n_t_new_EX50_SEP2023_weeks.csv")
else:
    df_raw_sigma_n_t = pd.read_csv("df_raw_sigma_n_t_new_EX50_SEP2023.csv")
    df_abn_sigma_n_t = pd.read_csv("df_abn_sigma_n_t_new_EX50_SEP2023.csv")
    df_raw_pre_sigma_n_t = pd.read_csv("df_raw_pre_sigma_n_t_new_EX50_SEP2023.csv")
    
del df_raw_sigma_n_t[df_raw_sigma_n_t.columns[0]]
del df_abn_sigma_n_t[df_abn_sigma_n_t.columns[0]]
del df_raw_pre_sigma_n_t[df_raw_pre_sigma_n_t.columns[0]]

if not WEEK_BASIS:
    df_raw_sigma_n_t["Date"] = pd.to_datetime(df_raw_sigma_n_t["Date"])
    df_abn_sigma_n_t["Date"] = pd.to_datetime(df_abn_sigma_n_t["Date"])
    df_raw_pre_sigma_n_t["Date"] = pd.to_datetime(df_raw_pre_sigma_n_t["Date"])

if(CUSIP_FILTER2 is not None):
    df_cusip_filter = pd.read_csv(CUSIP_FILTER2)
    df_raw_sigma_n_t = df_raw_sigma_n_t.merge(df_cusip_filter, on=["CUSIP"], how="left")
    df_raw_sigma_n_t = df_raw_sigma_n_t[df_raw_sigma_n_t["sic_code"].isna()]
    del df_raw_sigma_n_t["sic_code"]
    df_abn_sigma_n_t = df_abn_sigma_n_t.merge(df_cusip_filter, on=["CUSIP"], how="left")
    df_abn_sigma_n_t = df_abn_sigma_n_t[df_abn_sigma_n_t["sic_code"].isna()]
    del df_abn_sigma_n_t["sic_code"]
    df_raw_pre_sigma_n_t = df_raw_pre_sigma_n_t.merge(df_cusip_filter, on=["CUSIP"], how="left")
    df_raw_pre_sigma_n_t = df_raw_pre_sigma_n_t[df_raw_pre_sigma_n_t["sic_code"].isna()]
    del df_raw_pre_sigma_n_t["sic_code"]



if NOISE:
    # SABR_t = SABR(t-1, t+1)_n >> Standardized Abnormal Returns (SABR)
    df_sabr = generateShocked_SABR_noise(0, df_abn_returns, df_abn_sigma_n_t)
    print("Finished df_sabr")
    
    # SRR_t = SRR(t-1, t+1)_n >> Standardized Raw Returns (SRR)
    df_srr = generateShocked_SSR_noise(0, df_returns, df_raw_sigma_n_t)
    print("Finished df_srr")
    
    # SRR_pre_t = SRR(t-1, t+1)_n >> Pre-Standardized Raw Returns (SRR)
    df_srr_pre = generateShocked_SSR_pre_noise(0, df_returns, df_raw_pre_sigma_n_t)
    print("Finished df_srr_pre")

else:
    # SABR_t = SABR(t-1, t+1)_n >> Standardized Abnormal Returns (SABR)
    df_sabr = generateShocked_SABR(0, df_abn_returns, df_abn_sigma_n_t)
    print("Finished df_sabr")
    
    # SRR_t = SRR(t-1, t+1)_n >> Standardized Raw Returns (SRR)
    df_srr = generateShocked_SSR(0, df_returns, df_raw_sigma_n_t)
    print("Finished df_srr")
    
    # SRR_pre_t = SRR(t-1, t+1)_n >> Pre-Standardized Raw Returns (SRR)
    df_srr_pre = generateShocked_SSR_pre(0, df_returns, df_raw_pre_sigma_n_t)
    print("Finished df_srr_pre")


# SBM_t = SBM(t-1,t+1)_n  >> Calculate Standardized Benchmark Returns_t
df_std_benchmarks = df_srr.copy()
df_std_benchmarks = df_std_benchmarks[["Date", "SRR_t", "CUSIP"]]
df_std_benchmarks = df_std_benchmarks.merge(df_mat_rat_group, on=["Date", "CUSIP"])
df_std_benchmarks = df_std_benchmarks[["Date", "CUSIP", "SRR_t", "ratingGroup", "maturityGroup"]]
df_std_benchmarks = df_std_benchmarks.groupby(["Date", "ratingGroup", "maturityGroup"]).agg(count=("CUSIP", "size"), mean=("SRR_t", "mean"))
df_std_benchmarks = df_std_benchmarks.reset_index()
df_std_benchmarks = df_std_benchmarks[df_std_benchmarks["count"]>=FILTER_MIN_NUMBER_CUSIPS_IN_BENCHMARK]
# TODO UNCOMMENT LINE ABOVE WHEN WORKING WITH REAL DATA
del df_std_benchmarks["count"]
df_std_benchmarks["SBM_t"] = df_std_benchmarks["mean"]
del df_std_benchmarks["mean"]
df_std_benchmarks = df_std_benchmarks.dropna()
print("Finished df_std_benchmarks")


# SBM_pre_t = SBM(t-1,t+1)_n  >> Calculate Pre-Standardized Benchmark Returns_t
df_std_benchmarks_pre = df_srr_pre.copy()
df_std_benchmarks_pre = df_std_benchmarks_pre[["Date", "SRR_pre_t", "CUSIP"]]
df_std_benchmarks_pre = df_std_benchmarks_pre.merge(df_mat_rat_group, on=["Date", "CUSIP"])
df_std_benchmarks_pre = df_std_benchmarks_pre[["Date", "CUSIP", "SRR_pre_t", "ratingGroup", "maturityGroup"]]
df_std_benchmarks_pre = df_std_benchmarks_pre.groupby(["Date", "ratingGroup", "maturityGroup"]).agg(count=("CUSIP", "size"), mean=("SRR_pre_t", "mean"))
df_std_benchmarks_pre = df_std_benchmarks_pre.reset_index()
df_std_benchmarks_pre = df_std_benchmarks_pre[df_std_benchmarks_pre["count"]>=FILTER_MIN_NUMBER_CUSIPS_IN_BENCHMARK]
# TODO UNCOMMENT LINE ABOVE WHEN WORKING WITH REAL DATA
del df_std_benchmarks_pre["count"]
df_std_benchmarks_pre["SBM_pre_t"] = df_std_benchmarks_pre["mean"]
del df_std_benchmarks_pre["mean"]
df_std_benchmarks_pre = df_std_benchmarks_pre.dropna()
print("Finished df_std_benchmarks_pre")


# ABSR_t = ABSR(t-1, t+1)_n >> Abnormal Standardized Returns (ABSR)
df_absr = generateShocked_ABSR(0, df_returns, df_raw_sigma_n_t, df_mat_rat_group, df_std_benchmarks)
print("Finished df_absr")


# ABSR_pre_t = ABSR(t-1, t+1)_n >> Abnormal Pre-Standardized Returns (ABSR_Pre)
df_absr_pre = generateShocked_ABSR_pre(0, df_returns, df_raw_pre_sigma_n_t, df_mat_rat_group, df_std_benchmarks_pre)
print("Finished df_absr_pre")


































###############################################################################
# Tables
###############################################################################
# Table 1
print("Table 1 - The bond and firm return samples.")
print("Panel A - Unstandardized returns")
lst_group = ["Raw bond returns", "Abnormal bond returns", "Abnormal firm returns"]
populations = [df_returns["R_t"], df_abn_returns["ABR_t"], df_abn_returns_firms["F_ABR_t"]]
lst_mean = getPopulationValues(populations, getMean)
#lst_mean = [x*100 for x in lst_mean]
lst_median = getPopulationValues(populations, getMedian)
#lst_mean = [x*100 for x in lst_median]
lst_std = getPopulationValues(populations, getSTD)
lst_skew = getPopulationValues(populations, getSkew)
lst_kurt = getPopulationValues(populations, getKurtosis)
lst_pstv_share = getPopulationValues(populations, getSharePositive)
lst_n = getPopulationValues(populations, getN)
num_working_days_in_period = len(getListOfDaysFromTo(min(lst_dates), max(lst_dates)))
def getShareOfDaysWithReturn(df, col_id, lst_id):
    return df[["Date", col_id]].drop_duplicates().shape[0]/(num_working_days_in_period*len(lst_id))*100
def getShareOfDaysWithReturn2(df):
    return df[["Date"]].drop_duplicates().shape[0]/(num_working_days_in_period)*100
lst_share_days_with_return = [getShareOfDaysWithReturn(df_returns, "CUSIP", lst_cusips), getShareOfDaysWithReturn(df_abn_returns, "CUSIP", lst_cusips), getShareOfDaysWithReturn(df_abn_returns_firms, "FirmID_PERMNO", lst_permnos)]
lst_share_days_with_return2 = [getShareOfDaysWithReturn2(df_returns), getShareOfDaysWithReturn2(df_abn_returns), getShareOfDaysWithReturn2(df_abn_returns_firms)]
df_table1A = pd.DataFrame(list(zip(lst_group, lst_mean, lst_median, lst_std, lst_skew, lst_kurt, lst_pstv_share, lst_n, lst_share_days_with_return, lst_share_days_with_return2)), 
                         columns =['Group', 'Mean', 'Median', "Standard deviation", "Skewness", "Kurtosis", "% positive", "Observations", "% days with returns", "% days with returns 2"])
print(df_table1A.to_markdown())
print("Panel B - Bond characteristics in the bond return sample")
print("???")
print("Panel C - Rating and maturity distributions for the bond return samples")
df_benchmarks_temp = df_returns.copy()
df_benchmarks_temp = df_benchmarks_temp[["Date", "R_t", "CUSIP"]]
df_benchmarks_temp = df_benchmarks_temp.merge(df_mat_rat_group, left_on=["Date", "CUSIP"], right_on=["Date", "CUSIP"], how="left")
df_benchmarks_temp = df_benchmarks_temp[["Date", "CUSIP", "ratingGroup", "maturityGroup"]]
lst_maturity_group = ["0", "1", "2", "3"]
lst_maturity_group = ["1-3 years [0]", "3-5 years [1]", "5-10 years [2]", "Over 10 years [3]"]
lst_maturity_share = [df_benchmarks_temp[df_benchmarks_temp["maturityGroup"]==x].shape[0] / df_returns.shape[0] for x in range(0,4)]
lst_rating_group = ["0", "1", "2", "3", "4", "5"]
lst_rating_group = ["AAA to AA [0]", "A [1]", "BAA [2]", "BA [3]", "B [4]", "Below B [5]"]
lst_rating_share = [df_benchmarks_temp[df_benchmarks_temp["ratingGroup"]==x].shape[0] / df_returns.shape[0] for x in range(0,6)]
lst_group = [*lst_maturity_group, "", *lst_rating_group]
lst_share = [*lst_maturity_share, "", *lst_rating_share]
df_table1C = pd.DataFrame(list(zip(lst_group, lst_share)), columns =['Group', 'share'])
print(df_table1C.to_markdown())


# Table 2
print("Table 2 - Size and power tests based on unstandardized abnormal returns.")
print("Panel A - Size tests")
shock = generateShock(SHOCK_BASE_POINTS_T2)
df_dfdata_exp2 = {}

if NOISE:
    df_abn_returns_firms = aggregateToFirmLevel(generateShocked_ABR_noise(0, df_raw_sigma_n_t, df_returns, df_mat_rat_group, df_benchmarks), df_cusip_permno, "ABR_t")
    df_dfdata_exp2["df_all_data"] = df_abn_returns_firms.copy()
    df_abn_returns_firms_b_p = aggregateToFirmLevel(generateShocked_ABR_noise(+shock, df_raw_sigma_n_t, df_returns, df_mat_rat_group, df_benchmarks), df_cusip_permno, "ABR_t")
    df_dfdata_exp2["df_all_data_b_p"] = df_abn_returns_firms_b_p.copy()
    df_abn_returns_firms_b_n = aggregateToFirmLevel(generateShocked_ABR_noise(-shock, df_raw_sigma_n_t, df_returns, df_mat_rat_group, df_benchmarks), df_cusip_permno, "ABR_t")
    df_dfdata_exp2["df_all_data_b_n"] = df_abn_returns_firms_b_n.copy()
else:
    df_abn_returns_firms = aggregateToFirmLevel(generateShocked_ABR(0, df_returns, df_mat_rat_group, df_benchmarks), df_cusip_permno, "ABR_t")
    df_dfdata_exp2["df_all_data"] = df_abn_returns_firms.copy()
    df_abn_returns_firms_b_p = aggregateToFirmLevel(generateShocked_ABR(+shock, df_returns, df_mat_rat_group, df_benchmarks), df_cusip_permno, "ABR_t")
    df_dfdata_exp2["df_all_data_b_p"] = df_abn_returns_firms_b_p.copy()
    df_abn_returns_firms_b_n = aggregateToFirmLevel(generateShocked_ABR(-shock, df_returns, df_mat_rat_group, df_benchmarks), df_cusip_permno, "ABR_t")
    df_dfdata_exp2["df_all_data_b_n"] = df_abn_returns_firms_b_n.copy()

experiment_data_glob = {}
def doMonteCarloTable2(idx, start, end, df_dfdata_exp2):
    experiment_data = []
    for i in range(start,end):
        random_firms = experiment_montecarlo_selection[i]["firms"] # [random.choice(lst_permnos) for x in range(0,N_SAMPLE)]
        random_days = experiment_montecarlo_selection[i]["days"] # [random.choice(lst_dates) for x in range(0,N_SAMPLE)]
        df_sample = pd.DataFrame(list(zip(random_firms, random_days)), columns=["FirmID_PERMNO", "Date"])
        df_sample_all = getSampleWithDataForT4B(df_sample, df_dfdata_exp2["df_all_data"])
        df_sample_b_p = getSampleWithDataForT4B(df_sample, df_dfdata_exp2["df_all_data_b_p"])
        df_sample_b_n = getSampleWithDataForT4B(df_sample, df_dfdata_exp2["df_all_data_b_n"])
        # panel a - size tests
        test_a = getTrippleTestResult(df_sample_all["F_ABR_t"])
        # panel b - power tests for 15 basis point return shocks
        test_b_p = getTrippleTestResult(df_sample_b_p["F_ABR_t"])
        test_b_n = getTrippleTestResult(df_sample_b_n["F_ABR_t"])
        # result
        experiment_data.append([len(df_sample_all), 
                                test_a["t"], test_a["sr"], test_a["sg"],
                                test_b_p["t"], test_b_p["sr"], test_b_p["sg"],
                                test_b_n["t"], test_b_n["sr"], test_b_n["sg"]])
    experiment_data_glob[idx] = experiment_data
                
import threading
threads = []
for i in range(0,10):
    t = threading.Thread(target=doMonteCarloTable2, args=(i, 1000*i, 1000*(i+1), df_dfdata_exp2))
    threads.append(t)
for i in range(0,10):
    threads[i].start()
for i in range(0,10):
    threads[i].join()
print("Finished")
experiment_data = []
for i in range(0,10):
    experiment_data = experiment_data + experiment_data_glob[i]
    
experiment_data = np.asarray(experiment_data)
share = np.mean(experiment_data[:,0])
print("Average number of calculable returns in sample of size", N_SAMPLE, "is", share)
lst_group = ["Significance level (%)", "No event null rejection rates (%)"]
t_1_vals = [2.5,  meanExperiment(experiment_data[:,1], -1)]
t_2_vals = [97.5, meanExperiment(experiment_data[:,1], +1)]
sr1_vals = [2.5,  meanExperiment(experiment_data[:,2], -1)]
sr2_vals = [97.5, meanExperiment(experiment_data[:,2], +1)]
sg1_vals = [2.5,  meanExperiment(experiment_data[:,3], -1)]
sg2_vals = [97.5, meanExperiment(experiment_data[:,3], +1)]
lst_share_days_with_return = [x/len(lst_dates) for x in lst_n]
df_table2A = pd.DataFrame(list(zip(lst_group, t_1_vals, t_2_vals, sr1_vals, sr2_vals, sg1_vals, sg2_vals)), 
                         columns =['Group', 't-Test', '', "Signed-rank test", "", "Sign test", ""])
print(df_table2A.to_markdown())

print("Panel B - Power tests for 15 basis point return shocks")
lst_group = ["Positive shock (%)", "Negative shock (%)"]
t_vals  = [meanExperiment(experiment_data[:,4], +1), meanExperiment(experiment_data[:,7], -1)]
sr_vals = [meanExperiment(experiment_data[:,5], +1), meanExperiment(experiment_data[:,8], -1)]
sg_vals = [meanExperiment(experiment_data[:,6], +1), meanExperiment(experiment_data[:,9], -1)]
lst_share_days_with_return = [x/len(lst_dates) for x in lst_n]
df_table2B = pd.DataFrame(list(zip(lst_group, t_vals, sr_vals, sg_vals)), 
                         columns =['Group', 't-Test', "Signed-rank test", "Sign test"])
print(df_table2B.to_markdown())


# Table 3
print("Table 3 - Abnormal bond return standard deviations by rating and maturity.")
data_stds = []
data_n = []
for maturity_group in range(0,4):
    lst_stds = []
    lst_ns = []
    for rating_group in range(0,6):
        lst_stds.append(np.std(df_abn_returns[df_abn_returns["ratingGroup"]==rating_group][df_abn_returns["maturityGroup"]==maturity_group]["ABR_t"]))
        lst_ns.append(df_abn_returns[df_abn_returns["ratingGroup"]==rating_group][df_abn_returns["maturityGroup"]==maturity_group]["ABR_t"].shape[0])
    data_stds.append(lst_stds)
    data_n.append(lst_ns)
lst_group = ["AAA to AA [0]", "A [1]", "BAA [2]", "BA [3]", "B [4]", "Below B [5]"]
df_table3 = pd.DataFrame(list(zip(lst_group, data_stds[0], data_stds[1], data_stds[2], data_stds[3])), 
                         columns =['Rating', "1-3 years [0]", "3-5 years [1]", "5-10 years [2]", "Over 10 years [3]"])
print(df_table3.to_markdown())
df_table3 = pd.DataFrame(list(zip(lst_group, data_n[0], data_n[1], data_n[2], data_n[3])), 
                         columns =['Rating', "1-3 years [0]", "3-5 years [1]", "5-10 years [2]", "Over 10 years [3]"])
print(df_table3.to_markdown())



# Table 4
print("Table 4 - Size and power of event study tests based on standardized returns.")
print("Panel A - Return characteristics")
lst_group = ["Abr. Std. Ret. (ABSR)", "Std. Abr. Ret. (SABR)"]
populations = [df_absr["ABSR_t"], df_sabr["SABR_t"]]
lst_mean = getPopulationValues(populations, getMean)
lst_median = getPopulationValues(populations, getMedian)
lst_std = getPopulationValues(populations, getSTD)
lst_skew = getPopulationValues(populations, getSkew)
lst_kurt = getPopulationValues(populations, getKurtosis)
lst_n = getPopulationValues(populations, getN)
df_table4A = pd.DataFrame(list(zip(lst_group, lst_mean, lst_median, lst_std, lst_skew, lst_kurt, lst_n)), 
                         columns =['Group', 'Mean', 'Median', "Standard deviation", "Skewness", "Kurtosis", "Observations"])
print(df_table4A.to_markdown())
    
print("Panel B - Size tests")
shock_c  = generateShock(SHOCK_BASE_POINTS_T4C)
shock_d1 = generateShock(SHOCK_BASE_POINTS_T4D1)
shock_d2 = generateShock(SHOCK_BASE_POINTS_T4D2)
df_dfdata_exp4 = {}
if NOISE:
    df_dfdata_exp4["df_absr_firms"]             = aggregateToFirmLevel(generateShocked_ABSR_noise(0,  df_returns, df_raw_sigma_n_t, df_mat_rat_group, df_std_benchmarks), df_cusip_permno, "ABSR_t")
    df_dfdata_exp4["df_absr_firms_c_p"]         = aggregateToFirmLevel(generateShocked_ABSR_noise(+shock_c,  df_returns, df_raw_sigma_n_t, df_mat_rat_group, df_std_benchmarks), df_cusip_permno, "ABSR_t")
    df_dfdata_exp4["df_absr_firms_c_n"]         = aggregateToFirmLevel(generateShocked_ABSR_noise(-shock_c,  df_returns, df_raw_sigma_n_t, df_mat_rat_group, df_std_benchmarks), df_cusip_permno, "ABSR_t")
    df_dfdata_exp4["df_absr_firms_d1_p"]        = aggregateToFirmLevel(generateShocked_ABSR_noise(+shock_d1, df_returns, df_raw_sigma_n_t, df_mat_rat_group, df_std_benchmarks), df_cusip_permno, "ABSR_t")
    df_dfdata_exp4["df_absr_firms_d1_n"]        = aggregateToFirmLevel(generateShocked_ABSR_noise(-shock_d1, df_returns, df_raw_sigma_n_t, df_mat_rat_group, df_std_benchmarks), df_cusip_permno, "ABSR_t")
    df_dfdata_exp4["df_absr_firms_d2_p"]        = aggregateToFirmLevel(generateShocked_ABSR_noise(+shock_d2, df_returns, df_raw_sigma_n_t, df_mat_rat_group, df_std_benchmarks), df_cusip_permno, "ABSR_t")
    df_dfdata_exp4["df_absr_firms_d2_n"]        = aggregateToFirmLevel(generateShocked_ABSR_noise(-shock_d2, df_returns, df_raw_sigma_n_t, df_mat_rat_group, df_std_benchmarks), df_cusip_permno, "ABSR_t")
    df_dfdata_exp4["df_sabr_firms"]             = aggregateToFirmLevel(generateShocked_SABR_noise(0, df_abn_returns, df_abn_sigma_n_t), df_cusip_permno, "SABR_t")
    df_dfdata_exp4["df_sabr_firms_c_p"]         = aggregateToFirmLevel(generateShocked_SABR_noise(+shock_c, df_abn_returns, df_abn_sigma_n_t),  df_cusip_permno, "SABR_t")
    df_dfdata_exp4["df_sabr_firms_c_n"]         = aggregateToFirmLevel(generateShocked_SABR_noise(-shock_c, df_abn_returns, df_abn_sigma_n_t),  df_cusip_permno, "SABR_t")
    df_dfdata_exp4["df_abn_returns_firms"]      = aggregateToFirmLevel(generateShocked_ABR_noise(0, df_raw_sigma_n_t, df_returns, df_mat_rat_group, df_benchmarks), df_cusip_permno, "ABR_t")
    df_dfdata_exp4["df_abn_returns_firms_c_p"]  = aggregateToFirmLevel(generateShocked_ABR_noise(+shock_c, df_raw_sigma_n_t, df_returns, df_mat_rat_group, df_benchmarks), df_cusip_permno, "ABR_t")
    df_dfdata_exp4["df_abn_returns_firms_c_n"]  = aggregateToFirmLevel(generateShocked_ABR_noise(-shock_c, df_raw_sigma_n_t, df_returns, df_mat_rat_group, df_benchmarks), df_cusip_permno, "ABR_t")
    df_dfdata_exp4["df_abn_returns_firms_d1_p"] = aggregateToFirmLevel(generateShocked_ABR_noise(+shock_d1, df_raw_sigma_n_t, df_returns, df_mat_rat_group, df_benchmarks), df_cusip_permno, "ABR_t")
    df_dfdata_exp4["df_abn_returns_firms_d1_n"] = aggregateToFirmLevel(generateShocked_ABR_noise(-shock_d1, df_raw_sigma_n_t, df_returns, df_mat_rat_group, df_benchmarks), df_cusip_permno, "ABR_t")
    df_dfdata_exp4["df_abn_returns_firms_d2_p"] = aggregateToFirmLevel(generateShocked_ABR_noise(+shock_d2, df_raw_sigma_n_t, df_returns, df_mat_rat_group, df_benchmarks), df_cusip_permno, "ABR_t")
    df_dfdata_exp4["df_abn_returns_firms_d2_n"] = aggregateToFirmLevel(generateShocked_ABR_noise(-shock_d2, df_raw_sigma_n_t, df_returns, df_mat_rat_group, df_benchmarks), df_cusip_permno, "ABR_t")
    df_dfdata_exp4["df_absr_pre_firms"]         = aggregateToFirmLevel(generateShocked_ABSR_pre_noise(+0, df_returns, df_raw_pre_sigma_n_t, df_mat_rat_group, df_std_benchmarks_pre), df_cusip_permno, "ABSR_pre_t")
    df_dfdata_exp4["df_absr_pre_firms_c_p"]     = aggregateToFirmLevel(generateShocked_ABSR_pre_noise(+shock_c, df_returns, df_raw_pre_sigma_n_t, df_mat_rat_group, df_std_benchmarks_pre), df_cusip_permno, "ABSR_pre_t")
    df_dfdata_exp4["df_absr_pre_firms_c_n"]     = aggregateToFirmLevel(generateShocked_ABSR_pre_noise(-shock_c, df_returns, df_raw_pre_sigma_n_t, df_mat_rat_group, df_std_benchmarks_pre), df_cusip_permno, "ABSR_pre_t")
else:
    df_dfdata_exp4["df_absr_firms"]             = aggregateToFirmLevel(df_absr, df_cusip_permno, "ABSR_t")
    df_dfdata_exp4["df_absr_firms_c_p"]         = aggregateToFirmLevel(generateShocked_ABSR(+shock_c,  df_returns, df_raw_sigma_n_t, df_mat_rat_group, df_std_benchmarks), df_cusip_permno, "ABSR_t")
    df_dfdata_exp4["df_absr_firms_c_n"]         = aggregateToFirmLevel(generateShocked_ABSR(-shock_c,  df_returns, df_raw_sigma_n_t, df_mat_rat_group, df_std_benchmarks), df_cusip_permno, "ABSR_t")
    df_dfdata_exp4["df_absr_firms_d1_p"]        = aggregateToFirmLevel(generateShocked_ABSR(+shock_d1, df_returns, df_raw_sigma_n_t, df_mat_rat_group, df_std_benchmarks), df_cusip_permno, "ABSR_t")
    df_dfdata_exp4["df_absr_firms_d1_n"]        = aggregateToFirmLevel(generateShocked_ABSR(-shock_d1, df_returns, df_raw_sigma_n_t, df_mat_rat_group, df_std_benchmarks), df_cusip_permno, "ABSR_t")
    df_dfdata_exp4["df_absr_firms_d2_p"]        = aggregateToFirmLevel(generateShocked_ABSR(+shock_d2, df_returns, df_raw_sigma_n_t, df_mat_rat_group, df_std_benchmarks), df_cusip_permno, "ABSR_t")
    df_dfdata_exp4["df_absr_firms_d2_n"]        = aggregateToFirmLevel(generateShocked_ABSR(-shock_d2, df_returns, df_raw_sigma_n_t, df_mat_rat_group, df_std_benchmarks), df_cusip_permno, "ABSR_t")
    df_dfdata_exp4["df_sabr_firms"]             = aggregateToFirmLevel(df_sabr, df_cusip_permno, "SABR_t")
    df_dfdata_exp4["df_sabr_firms_c_p"]         = aggregateToFirmLevel(generateShocked_SABR(+shock_c, df_abn_returns, df_abn_sigma_n_t),  df_cusip_permno, "SABR_t")
    df_dfdata_exp4["df_sabr_firms_c_n"]         = aggregateToFirmLevel(generateShocked_SABR(-shock_c, df_abn_returns, df_abn_sigma_n_t),  df_cusip_permno, "SABR_t")
    df_dfdata_exp4["df_abn_returns_firms"]      = aggregateToFirmLevel(df_abn_returns, df_cusip_permno, "ABR_t")
    df_dfdata_exp4["df_abn_returns_firms_c_p"]  = aggregateToFirmLevel(generateShocked_ABR(+shock_c, df_returns, df_mat_rat_group, df_benchmarks), df_cusip_permno, "ABR_t")
    df_dfdata_exp4["df_abn_returns_firms_c_n"]  = aggregateToFirmLevel(generateShocked_ABR(-shock_c, df_returns, df_mat_rat_group, df_benchmarks), df_cusip_permno, "ABR_t")
    df_dfdata_exp4["df_abn_returns_firms_d1_p"] = aggregateToFirmLevel(generateShocked_ABR(+shock_d1, df_returns, df_mat_rat_group, df_benchmarks), df_cusip_permno, "ABR_t")
    df_dfdata_exp4["df_abn_returns_firms_d1_n"] = aggregateToFirmLevel(generateShocked_ABR(-shock_d1, df_returns, df_mat_rat_group, df_benchmarks), df_cusip_permno, "ABR_t")
    df_dfdata_exp4["df_abn_returns_firms_d2_p"] = aggregateToFirmLevel(generateShocked_ABR(+shock_d2, df_returns, df_mat_rat_group, df_benchmarks), df_cusip_permno, "ABR_t")
    df_dfdata_exp4["df_abn_returns_firms_d2_n"] = aggregateToFirmLevel(generateShocked_ABR(-shock_d2, df_returns, df_mat_rat_group, df_benchmarks), df_cusip_permno, "ABR_t")
    df_dfdata_exp4["df_absr_pre_firms"]         = aggregateToFirmLevel(df_absr_pre, df_cusip_permno, "ABSR_pre_t")
    df_dfdata_exp4["df_absr_pre_firms_c_p"]     = aggregateToFirmLevel(generateShocked_ABSR_pre(+shock_c, df_returns, df_raw_pre_sigma_n_t, df_mat_rat_group, df_std_benchmarks_pre), df_cusip_permno, "ABSR_pre_t")
    df_dfdata_exp4["df_absr_pre_firms_c_n"]     = aggregateToFirmLevel(generateShocked_ABSR_pre(-shock_c, df_returns, df_raw_pre_sigma_n_t, df_mat_rat_group, df_std_benchmarks_pre), df_cusip_permno, "ABSR_pre_t")

experiment_data_glob = {}
def doMonteCarloTable4(idx, start, end, df_dfdata_exp2):
    experiment_data = []
    for i in range(start, end):
        random_firms = experiment_montecarlo_selection[i]["firms"] # [random.choice(lst_permnos) for x in range(0,N_SAMPLE)]
        random_days = experiment_montecarlo_selection[i]["days"] # [random.choice(lst_dates) for x in range(0,N_SAMPLE)]
        df_sample = pd.DataFrame(list(zip(random_firms, random_days)), columns=["FirmID_PERMNO", "Date"])
        # === Panel B - Size tests
        # === Panel C - Power tests - 15 basis point shocks
        # === Panel D - Power tests for 10 and 25 basis point shocks
        # ABSR
        test_absr       = getTrippleTestResult(getSampleWithDataForT4B(df_sample, df_dfdata_exp4["df_absr_firms"])["F_ABSR_t"])
        test_absr_p     = getTrippleTestResult(getSampleWithDataForT4B(df_sample, df_dfdata_exp4["df_absr_firms_c_p"])["F_ABSR_t"])
        test_absr_n     = getTrippleTestResult(getSampleWithDataForT4B(df_sample, df_dfdata_exp4["df_absr_firms_c_n"])["F_ABSR_t"])
        test_absr_p_d10 = getTrippleTestResult(getSampleWithDataForT4B(df_sample, df_dfdata_exp4["df_absr_firms_d1_p"])["F_ABSR_t"])
        test_absr_n_d10 = getTrippleTestResult(getSampleWithDataForT4B(df_sample, df_dfdata_exp4["df_absr_firms_d1_n"])["F_ABSR_t"])
        test_absr_p_d25 = getTrippleTestResult(getSampleWithDataForT4B(df_sample, df_dfdata_exp4["df_absr_firms_d2_p"])["F_ABSR_t"])
        test_absr_n_d25 = getTrippleTestResult(getSampleWithDataForT4B(df_sample, df_dfdata_exp4["df_absr_firms_d2_n"])["F_ABSR_t"])
        # SABR
        test_sabr       = getTrippleTestResult(getSampleWithDataForT4B(df_sample, df_dfdata_exp4["df_sabr_firms"])["F_SABR_t"])
        test_sabr_p     = getTrippleTestResult(getSampleWithDataForT4B(df_sample, df_dfdata_exp4["df_sabr_firms_c_p"])["F_SABR_t"])
        test_sabr_n     = getTrippleTestResult(getSampleWithDataForT4B(df_sample, df_dfdata_exp4["df_sabr_firms_c_n"])["F_SABR_t"])
        # ABR (unstandardized)
        test_abr        = getTrippleTestResult(getSampleWithDataForT4B(df_sample, df_dfdata_exp4["df_abn_returns_firms"])["F_ABR_t"])
        test_abr_p      = getTrippleTestResult(getSampleWithDataForT4B(df_sample, df_dfdata_exp4["df_abn_returns_firms_c_p"])["F_ABR_t"])
        test_abr_n      = getTrippleTestResult(getSampleWithDataForT4B(df_sample, df_dfdata_exp4["df_abn_returns_firms_c_n"])["F_ABR_t"])
        test_abr_p_d10  = getTrippleTestResult(getSampleWithDataForT4B(df_sample, df_dfdata_exp4["df_abn_returns_firms_d1_p"])["F_ABR_t"])
        test_abr_n_d10  = getTrippleTestResult(getSampleWithDataForT4B(df_sample, df_dfdata_exp4["df_abn_returns_firms_d1_n"])["F_ABR_t"])
        test_abr_p_d25  = getTrippleTestResult(getSampleWithDataForT4B(df_sample, df_dfdata_exp4["df_abn_returns_firms_d2_p"])["F_ABR_t"])
        test_abr_n_d25  = getTrippleTestResult(getSampleWithDataForT4B(df_sample, df_dfdata_exp4["df_abn_returns_firms_d2_n"])["F_ABR_t"])
        # ABSR_pre
        test_absr_pre   = getTrippleTestResult(getSampleWithDataForT4B(df_sample, df_dfdata_exp4["df_absr_pre_firms"])["F_ABSR_pre_t"])
        test_absr_pre_p = getTrippleTestResult(getSampleWithDataForT4B(df_sample, df_dfdata_exp4["df_absr_pre_firms_c_p"])["F_ABSR_pre_t"])
        test_absr_pre_n = getTrippleTestResult(getSampleWithDataForT4B(df_sample, df_dfdata_exp4["df_absr_pre_firms_c_n"])["F_ABSR_pre_t"])
        # result
        experiment_data.append([
                                # for panel b
                                test_absr["t"], test_absr["sr"], test_absr["sg"],
                                test_sabr["t"], test_sabr["sr"], test_sabr["sg"],
                                test_abr["t"],  test_abr["sr"],  test_abr["sg"],
                                test_absr_pre["t"], test_absr_pre["sr"], test_absr_pre["sg"],
                                # for panel c
                                    # n
                                test_absr_n["t"], test_absr_n["sr"], test_absr_n["sg"],
                                test_sabr_n["t"], test_sabr_n["sr"], test_sabr_n["sg"],
                                test_abr_n["t"],  test_abr_n["sr"],  test_abr_n["sg"],
                                test_absr_pre_n["t"], test_absr_pre_n["sr"], test_absr_pre_n["sg"],
                                    # p
                                test_absr_p["t"], test_absr_p["sr"], test_absr_p["sg"],
                                test_sabr_p["t"], test_sabr_p["sr"], test_sabr_p["sg"],
                                test_abr_p["t"],  test_abr_p["sr"],  test_abr_p["sg"],
                                test_absr_pre_p["t"], test_absr_pre_p["sr"], test_absr_pre_p["sg"],
                                # for panel d
                                    # 10 bp unstandardized
                                test_abr_n_d10["t"], test_abr_n_d10["sr"], test_abr_n_d10["sg"],
                                test_abr_p_d10["t"], test_abr_p_d10["sr"], test_abr_p_d10["sg"],
                                    # 10 bp standardized
                                test_absr_n_d10["t"], test_absr_n_d10["sr"], test_absr_n_d10["sg"],
                                test_absr_p_d10["t"], test_absr_p_d10["sr"], test_absr_p_d10["sg"],
                                    # 25 bp unstandardized
                                test_abr_n_d25["t"], test_abr_n_d25["sr"], test_abr_n_d25["sg"],
                                test_abr_p_d25["t"], test_abr_p_d25["sr"], test_abr_p_d25["sg"],
                                    # 25 bp standardized
                                test_absr_n_d25["t"], test_absr_n_d25["sr"], test_absr_n_d25["sg"],
                                test_absr_p_d25["t"], test_absr_p_d25["sr"], test_absr_p_d25["sg"],
                                ])
    experiment_data_glob[idx] = experiment_data

import threading
threads = []
for i in range(0,10):
    t = threading.Thread(target=doMonteCarloTable4, args=(i, 1000*i, 1000*(i+1), df_dfdata_exp2))
    threads.append(t)
for i in range(0,10):
    threads[i].start()
for i in range(0,10):
    threads[i].join()
print("Finished")
experiment_data = []
for i in range(0,10):
    experiment_data = experiment_data + experiment_data_glob[i]
    
experiment_data = np.asarray(experiment_data)
share = np.mean(experiment_data[:,0])
# print("Average number of calculable returns in sample of size", N_SAMPLE, "is", share)
lst_group = ["Significance level (%)", "ABSR", "SABR", "ABR (unstandardized)", "ABSR-Pre"]
t_1_vals = [2.5,    meanExperiment(experiment_data[:,0], -1), meanExperiment(experiment_data[:,3], -1), meanExperiment(experiment_data[:,6], -1), meanExperiment(experiment_data[:,9], -1)]
t_2_vals = [97.5,   meanExperiment(experiment_data[:,0], +1), meanExperiment(experiment_data[:,3], +1), meanExperiment(experiment_data[:,6], +1), meanExperiment(experiment_data[:,9], +1)]
sr1_vals = [2.5,    meanExperiment(experiment_data[:,1], -1), meanExperiment(experiment_data[:,4], -1), meanExperiment(experiment_data[:,7], -1), meanExperiment(experiment_data[:,10], -1)]
sr2_vals = [97.5,   meanExperiment(experiment_data[:,1], +1), meanExperiment(experiment_data[:,4], +1), meanExperiment(experiment_data[:,7], +1), meanExperiment(experiment_data[:,10], +1)]
sg1_vals = [2.5,    meanExperiment(experiment_data[:,2], -1), meanExperiment(experiment_data[:,5], -1), meanExperiment(experiment_data[:,8], -1), meanExperiment(experiment_data[:,11], -1)]
sg2_vals = [97.5,   meanExperiment(experiment_data[:,2], +1), meanExperiment(experiment_data[:,5], +1), meanExperiment(experiment_data[:,8], +1), meanExperiment(experiment_data[:,11], +1)]
lst_share_days_with_return = [x/len(lst_dates) for x in lst_n]
df_table4B = pd.DataFrame(list(zip(lst_group, t_1_vals, t_2_vals, sr1_vals, sr2_vals, sg1_vals, sg2_vals)), 
                         columns =['Group', 't-Test', '', "Signed-rank test", "", "Sign test", ""])
print(df_table4B.to_markdown())

print("Panel C - Power tests - 15 basis point shocks")
lst_group = ["ABSR", "SABR", "ABR (unstandardized)", "ABSR-Pre"]
n_t_vals  = [meanExperiment(experiment_data[:,12], -1), meanExperiment(experiment_data[:,15], -1), meanExperiment(experiment_data[:,18], -1), meanExperiment(experiment_data[:,21], -1)]
n_sr_vals = [meanExperiment(experiment_data[:,13], -1), meanExperiment(experiment_data[:,16], -1), meanExperiment(experiment_data[:,19], -1), meanExperiment(experiment_data[:,22], -1)]
n_sg_vals = [meanExperiment(experiment_data[:,14], -1), meanExperiment(experiment_data[:,17], -1), meanExperiment(experiment_data[:,20], -1), meanExperiment(experiment_data[:,23], -1)]
p_t_vals  = [meanExperiment(experiment_data[:,24], +1), meanExperiment(experiment_data[:,27], +1), meanExperiment(experiment_data[:,30], +1), meanExperiment(experiment_data[:,33], +1)]
p_sr_vals = [meanExperiment(experiment_data[:,25], +1), meanExperiment(experiment_data[:,28], +1), meanExperiment(experiment_data[:,31], +1), meanExperiment(experiment_data[:,34], +1)]
p_sg_vals = [meanExperiment(experiment_data[:,26], +1), meanExperiment(experiment_data[:,29], +1), meanExperiment(experiment_data[:,32], +1), meanExperiment(experiment_data[:,35], +1)]
lst_share_days_with_return = [x/len(lst_dates) for x in lst_n]
df_table4C = pd.DataFrame(list(zip(lst_group, n_t_vals, n_sr_vals, n_sg_vals, p_t_vals, p_sr_vals, p_sg_vals)), 
                         columns =["Group", "t-Test (%) neg", "Signed-rank test (%) neg", "Sign test (%) neg", "t-Test (%) pos", "Signed-rank test (%) pos", "Sign test (%) pos"])
print(df_table4C.to_markdown())

print("Panel D - Power tests for 10 and 25 basis point shocks")
lst_group = ["10 bp - unstandardized", "10 bp - standardized", "25 bp - unstandardized", "25 bp - standardized"]
n_t_vals  = [meanExperiment(experiment_data[:,36], -1), meanExperiment(experiment_data[:,42], -1), meanExperiment(experiment_data[:,48], -1), meanExperiment(experiment_data[:,54], -1)]
n_sr_vals = [meanExperiment(experiment_data[:,37], -1), meanExperiment(experiment_data[:,43], -1), meanExperiment(experiment_data[:,49], -1), meanExperiment(experiment_data[:,55], -1)]
n_sg_vals = [meanExperiment(experiment_data[:,38], -1), meanExperiment(experiment_data[:,44], -1), meanExperiment(experiment_data[:,50], -1), meanExperiment(experiment_data[:,56], -1)]
p_t_vals  = [meanExperiment(experiment_data[:,39], +1), meanExperiment(experiment_data[:,45], +1), meanExperiment(experiment_data[:,51], +1), meanExperiment(experiment_data[:,57], +1)]
p_sr_vals = [meanExperiment(experiment_data[:,40], +1), meanExperiment(experiment_data[:,46], +1), meanExperiment(experiment_data[:,52], +1), meanExperiment(experiment_data[:,58], +1)]
p_sg_vals = [meanExperiment(experiment_data[:,41], +1), meanExperiment(experiment_data[:,47], +1), meanExperiment(experiment_data[:,53], +1), meanExperiment(experiment_data[:,59], +1)]
lst_share_days_with_return = [x/len(lst_dates) for x in lst_n]
df_table4D = pd.DataFrame(list(zip(lst_group, n_t_vals, n_sr_vals, n_sg_vals, p_t_vals, p_sr_vals, p_sg_vals)), 
                         columns =["Group", "t-Test (%) neg", "Signed-rank test (%) neg", "Sign test (%) neg", "t-Test (%) pos", "Signed-rank test (%) pos", "Sign test (%) pos"])
print(df_table4D.to_markdown())

# Table 5
print("Table 5 - Test power stratified by bond rating and maturity for a 15 basis point return shock.")
shock = generateShock(SHOCK_BASE_POINTS_T5)
if NOISE:
    df_absr_p  = generateShocked_ABSR_noise(+shock, df_returns, df_raw_sigma_n_t, df_mat_rat_group, df_std_benchmarks)
    df_absr_n  = generateShocked_ABSR_noise(-shock, df_returns, df_raw_sigma_n_t, df_mat_rat_group, df_std_benchmarks)
else:
    df_absr_p  = generateShocked_ABSR(+shock, df_returns, df_raw_sigma_n_t, df_mat_rat_group, df_std_benchmarks)
    df_absr_n  = generateShocked_ABSR(-shock, df_returns, df_raw_sigma_n_t, df_mat_rat_group, df_std_benchmarks)
experiment_data = {}
experiment_data_n = {}

for maturity_group in range(0,4):
    experiment_data[maturity_group] = {}
    experiment_data_n[maturity_group] = {}
    for rating_group in range(0,6):
        df_data_p = df_absr_p.copy()
        df_data_p = df_data_p[df_data_p["ratingGroup"]==rating_group]
        df_data_p = df_data_p[df_data_p["maturityGroup"]==maturity_group]
        df_data_n = df_absr_n.copy()
        df_data_n = df_data_n[df_data_n["ratingGroup"]==rating_group]
        df_data_n = df_data_n[df_data_n["maturityGroup"]==maturity_group]
        experiment_lst = []
        if(df_data_p.shape[0]>0):
            for i in range(0,N_MONTE_CARLO_EXPERIMENT_REPETITION):
                df_sample = df_data_p.sample(n=N_SAMPLE, replace=True)
                df_sample = df_sample.dropna()            
                p_sr_test_absr = doSRtest(df_sample["ABSR_t"])
                experiment_lst.append(p_sr_test_absr==+1)
        if(df_data_n.shape[0]>0):
            for i in range(0,N_MONTE_CARLO_EXPERIMENT_REPETITION):
                df_sample = df_data_n.sample(n=N_SAMPLE, replace=True)
                df_sample = df_sample.dropna()            
                n_sr_test_absr = doSRtest(df_sample["ABSR_t"])
                experiment_lst.append(n_sr_test_absr==-1)
        experiment_data_n[maturity_group][rating_group] = df_data_p.shape[0]
        if(len(experiment_lst)>0):
            experiment_data[maturity_group][rating_group] = 100*np.sum(experiment_lst)/len(experiment_lst)
        else:
            experiment_data[maturity_group][rating_group] = -1
lst_group = ["AAA to AA [0]", "A [1]", "BAA [2]", "BA [3]", "B [4]", "Below B [5]"]
data_mat = [ [experiment_data[y][x] for x in range(0,6)] for y in range(0,4)]
df_table5 = pd.DataFrame(list(zip(lst_group, data_mat[0], data_mat[1], data_mat[2], data_mat[3])), 
                         columns =['Rating', "1-3 years [0]", "3-5 years [1]", "5-10 years [2]", "Over 10 years [3]"])
print(df_table5.to_markdown())
data_mat = [ [experiment_data_n[y][x] for x in range(0,6)] for y in range(0,4)]
df_table5 = pd.DataFrame(list(zip(lst_group, data_mat[0], data_mat[1], data_mat[2], data_mat[3])), 
                         columns =['Rating', "1-3 years [0]", "3-5 years [1]", "5-10 years [2]", "Over 10 years [3]"])
print(df_table5.to_markdown())


# Table 6
print("Table 6 - The effect of trade sampling on test power.")
print("Panel A - Large trades")
print("Across-the-board 15 bp shocks")
print("Proportional shocks")
print("Panel B - Interdealer trades")
print("Across-the-board 15 bp shocks")
print("Proportional shocks")


# Table 7
print("Table 7 - Comparing the power of tests based on different measures of average daily prices.")
print("Panel A - Power tests – 15 bp across-the-board shocks")
print("Panel B - Power tests – proportional shocks")


# Table 8
print("Table 8 - Comparing the size and power of tests based on different abnormal standardized return measures over a four-day window.")
print("Panel A - Size tests")
print("Panel B - Power tests - 15 bp across-the-board shocks")
print("Panel C - Power tests - proportional shocks")


# Table 9
print("Table 9 - Comparing the size and power of tests based on 2-day and composite 4-day, 6-day, 8-day and 10-day event windows.")
print("Panel A - Size tests")
print("Panel B - Power tests - 15 bp across-the-board-shocks")
print("Panel C - Power tests - proportional shocks")


# Table 10
print("Table 10 - Measures of test bias when firms share a common event day.")
print("Panel A - ABSR{t - 3, t + 3} abnormal standardized returns")
print("Panel B - ABSR{t - 3, t + 3} composite returns")
