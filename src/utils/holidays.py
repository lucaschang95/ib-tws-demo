import pandas as pd

# Define US market holidays for the date range
def get_market_holidays(year):
    # Memorial Day (Last Monday in May)
    memorial_day = pd.Timestamp(f"{year}-05-01")
    while memorial_day.month == 5:
        if memorial_day.weekday() == 0:  # Monday
            last_monday = memorial_day
        memorial_day += pd.Timedelta(days=1)
    
    holidays = [
        # New Year's Day
        pd.Timestamp(f"{year}-01-01"),
        # Martin Luther King Jr. Day (3rd Monday in January)
        pd.Timestamp(f"{year}-01-01") + pd.offsets.DateOffset(weekday=0) + pd.offsets.Week(2),
        # Presidents Day (3rd Monday in February)
        pd.Timestamp(f"{year}-02-01") + pd.offsets.DateOffset(weekday=0) + pd.offsets.Week(2),
        # Good Friday (approximate - using fixed date for simplicity)
        pd.Timestamp(f"{year}-03-29"),
        # Memorial Day
        last_monday,
        # Juneteenth
        pd.Timestamp(f"{year}-06-19"),
        # Independence Day
        pd.Timestamp(f"{year}-07-04"),
        # Labor Day (1st Monday in September)
        pd.Timestamp(f"{year}-09-01") + pd.offsets.DateOffset(weekday=0),
        # Thanksgiving (4th Thursday in November)
        pd.Timestamp(f"{year}-11-01") + pd.offsets.DateOffset(weekday=3) + pd.offsets.Week(3),
        # Christmas
        pd.Timestamp(f"{year}-12-25")
    ]
    
    # Adjust for weekends
    adjusted_holidays = []
    for holiday in holidays:
        if holiday.weekday() == 5:  # Saturday
            adjusted_holidays.append(holiday - pd.Timedelta(days=1))
        elif holiday.weekday() == 6:  # Sunday
            adjusted_holidays.append(holiday + pd.Timedelta(days=1))
        else:
            adjusted_holidays.append(holiday)
    
    return adjusted_holidays