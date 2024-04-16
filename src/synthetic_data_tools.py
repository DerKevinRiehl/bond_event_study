###############################################################################
# Source Code for Publication "Corporate Bond Market Event Studies: Event-Induced Variance and Liquidity"
# Authors: Lukas Müller, Kevin Riehl, Sonja Buschulte, Patrick Weiss
# Code Author: Kevin Riehl
###############################################################################




###############################################################################
# Imports
###############################################################################
import datetime
import random
import string
import pandas as pd

from bond_event_study_tools import getListOfDays, getDaysFromDateTimeDifference, determineMaturityRatingGroupClassification



###############################################################################
# Methods
###############################################################################
def generateDateFromString(text, format_str='%Y%m%d'):
    """
    This function converts a given string to a datetime object.
    Example Input:   text="20100101"
    Example Output:  datetime.datetime(2010, 1, 1, 0, 0)

    Parameters
    ----------
    text : str
        Date as input string in specific format.
    format: str
        Input date string format, default: '%Y%m%d'
        
    Returns
    -------
    datetime.datetime
        The string converted to datetime object.
    """
    return  datetime.datetime.strptime(text, '%Y%m%d')
    
def getListOfRandomCUSIPs(num_cusips, n_digits_cusips=9):
    """
    This function generates a list of random CUSIPs.
    Example Input:   num_cusips=2
    Example Output:  ['GCEHLPBIT', 'ADWF9MYW8']
    Parameters
    ----------
    num_cusips : int
        The number of CUSIPs that shall be returned.
    n_digits_cusips: int
        The number of digits for each CUSIP.
        
    Returns
    -------
    lst[str]
        A list of random CUSIPs strings.
    """
    lst_cusips = []
    for i in range(0, num_cusips):
        lst_cusips.append("".join(random.choices(string.ascii_uppercase + string.digits, k=n_digits_cusips)))
    return lst_cusips

def generateRandomMaturityDatesForCUSIPs(start_date, lst_cusips):
    """
    This function generates a random map of maturity dates for the given CUSIPs.

    Parameters
    ----------
    start_date : datetime.datetime
        A start date.
    lst_cusips: lst[str]
        A list of CUSIPs strings.
        
    Returns
    -------
    dict[str -> datetime.datetime]
        A map from the CUSIPs to their maturity date.
    """
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

def generateRandomRatingPerCUSIPDate(lst_cusips, lst_dates, rating_max=22, rating_min=1):
    """
    This function generates a random map of ratings for each CUSIP and date.

    Parameters
    ----------
    lst_cusips: lst[str]
        A list of CUSIPs strings.
    lst_dates : lst[datetime.datetime]
        A list of dates.
    rating_max : int
        The maximum of the numeric rating scale.
    rating_min : int
        The minimum of the numeric rating scale.
    
    Returns
    -------
    dict[cusip -> dict[date -> rating]]
        A map of ratings for each CUSIP and date.
    """
    dict_rating = {}
    for cusip in lst_cusips:
        start_rating = random.randint(rating_min, rating_max)
        current_rating = start_rating
        dict_dates = {}
        for date in lst_dates:
            dict_dates[date] = current_rating
            if(random.randint(1, 2)==1):
                current_rating += 1
                if(current_rating>rating_max):
                    current_rating = rating_max
            else:
                current_rating -= 1
                if(current_rating<rating_min):
                    current_rating = rating_min
        dict_rating[cusip] = dict_dates
    return dict_rating

def generateRandomPricesPerCUSIPDate(lst_cusips, lst_dates):
    """
    This function generates a random map of prices for each CUSIP and date.

    Parameters
    ----------
    lst_cusips: lst[str]
        A list of CUSIPs strings.
    lst_dates : lst[datetime.datetime]
        A list of dates.
    
    Returns
    -------
    dict[cusip -> dict[date -> price]]
        A map of prices for each CUSIP and date.
    """
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


def generateSyntheticTradesDataset(number_of_cusips, number_of_permnos, start_date, 
                             number_of_days, number_of_trades_per_day_per_cusip_min, 
                             number_of_trades_per_day_per_cusip_max, 
                             min_num_days_to_maturity_filter,
                             weekly_basis, n_monte_carlo_experiment_repetitions,
                             n_monte_carlo_samples):
    """
    This function generates a synthetic trades dataset for demonstration purposes.
    
    Parameters
    ----------
    number_of_cusips : int
    number_of_permnos : int
    start_date : datetime.datetime
    number_of_days : int
    number_of_trades_per_day_per_cusip_min : int
    number_of_trades_per_day_per_cusip_max : int
    min_num_days_to_maturity_filter : int
    weekly_basis : bool
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
    df_mat_rat_group
        ['Date', 'CUSIP', 'ratingGroup', 'maturityGroup'] Bond_Maturity_Rating_Groups
    """
        # Prepare Lists
    lst_cusips = getListOfRandomCUSIPs(number_of_cusips)
    lst_permnos = getListOfRandomCUSIPs(number_of_permnos)
    #lst_cusips = loadListOfCusips("Input_Data/tidyfinance_cusips_WRDS.csv", number_of_cusips)
    lst_dates = getListOfDays(start_date, number_of_days)
    dict_cusip_maturity_dates = generateRandomMaturityDatesForCUSIPs(start_date, lst_cusips)
    dict_ratings = generateRandomRatingPerCUSIPDate(lst_cusips, lst_dates)
    dict_prices = generateRandomPricesPerCUSIPDate(lst_cusips, lst_dates)
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
    df_input = df_input[df_input["remaining_days_until_maturity_int"]>min_num_days_to_maturity_filter]
    del df_input["remaining_days_until_maturity"] 
    del df_input["remaining_days_until_maturity_int"] 
        # Generate Empty Events
    df_events = pd.DataFrame([], columns=["Date", "CUSIP"])
    df_events["TO_BE_EXCLUDED"] = 1
        # Generate CUSIP / PERMNO Loopup Table
    df_cusip_permno = df_input[["Date", "CUSIP", "FirmID_PERMNO"]]
    df_cusip_permno = df_cusip_permno.drop_duplicates()
    # Experiment_montecarlo_selection
    if weekly_basis:
        lst_weeks = []
        for d in lst_dates:
            lst_weeks.append("{0:0=4d}".format(d.isocalendar()[0])+"-"+"{0:0=2d}".format(d.isocalendar()[1]))
        experiment_montecarlo_selection = []
        for i in range(0,n_monte_carlo_experiment_repetitions):
            random_firms = [random.choice(lst_permnos) for x in range(0,n_monte_carlo_samples)]
            random_days = [random.choice(lst_weeks) for x in range(0,n_monte_carlo_samples)]
            data = {}
            data["firms"] = random_firms
            data["days"] = random_days
            experiment_montecarlo_selection.append(data)
    else:   
        experiment_montecarlo_selection = []
        for i in range(0,n_monte_carlo_experiment_repetitions):
            random_firms = [random.choice(lst_permnos) for x in range(0,n_monte_carlo_samples)]
            random_days = [random.choice(lst_dates) for x in range(0,n_monte_carlo_samples)]
            data = {}
            data["firms"] = random_firms
            data["days"] = random_days
            experiment_montecarlo_selection.append(data)
    df_mat_rat_group = determineMaturityRatingGroupClassification(df_input)
    return df_input, df_events, df_cusip_permno, lst_cusips, lst_permnos, lst_dates, experiment_montecarlo_selection, df_mat_rat_group
            