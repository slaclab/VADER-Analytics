from sklearn import linear_model
import pandas as pd
import numpy as np
from skimage.util.shape import view_as_windows

__author__ = 'Jiafan Yu'


class ErcotRegression:
    def __init__(self):
        self.model = linear_model.LinearRegression(copy_X=True)
        self.features = None
        self.window_size = 168
        pass

    def generate_x(self, X, dates, stepAhead):
        """

        :param X: a numpy array, with size (number of users, 2 x window size)
        :param dates: a pandas timestamp series
        :param stepAhead: forecasting horizon
        :return:
        """

        number_of_samples = len(dates)

        daily_start_index_list = dates.to_datetime()

        # Build calendar variables:

        # Day of the week variables, daily and hourly fractions share the same
        df_dow = pd.DataFrame(index=range(number_of_samples),
                              columns=['Monday', 'TWT', 'Friday', 'Saturday', 'Sunday'])
        df_dow['Monday'] = [int(start_index.isoweekday() == 1) for start_index in daily_start_index_list]
        df_dow['TWT'] = [int(start_index.isoweekday() in (2, 3, 4)) for start_index in daily_start_index_list]
        df_dow['Friday'] = [int(start_index.isoweekday() == 5) for start_index in daily_start_index_list]
        df_dow['Saturday'] = [int(start_index.isoweekday() == 6) for start_index in daily_start_index_list]
        df_dow['Sunday'] = [int(start_index.isoweekday() == 7) for start_index in daily_start_index_list]

        # Holiday variables, daily and hourly fractions share the same
        df_holidays = pd.DataFrame(index=range(number_of_samples),
                                   columns=['NewYearsHoliday', 'MartinLKing', 'PresidentDay', 'MemorialDay',
                                            'July4thHol',
                                            'LaborDay', 'Thanksgiving', 'FridayAfterThanks', 'ChristmasHoliday',
                                            'XMasWkB4',
                                            'XMasWkAft'])

        df_holidays['NewYearsHoliday'] = [self.is_new_year(start_index) for start_index in daily_start_index_list]
        df_holidays['MartinLKing'] = [self.is_mlk(start_index) for start_index in daily_start_index_list]
        df_holidays['PresidentDay'] = [self.is_president_day(start_index) for start_index in daily_start_index_list]
        df_holidays['MemorialDay'] = [self.is_memorial_day(start_index) for start_index in daily_start_index_list]
        df_holidays['July4thHol'] = [self.is_july_4th(start_index) for start_index in daily_start_index_list]
        df_holidays['LaborDay'] = [self.is_labor_day(start_index) for start_index in daily_start_index_list]
        df_holidays['Thanksgiving'] = [self.is_thanksgiving(start_index) for start_index in daily_start_index_list]
        df_holidays['FridayAfterThanks'] = [self.is_friday_after_thanksgiving(start_index) for start_index in
                                            daily_start_index_list]
        df_holidays['ChristmasHoliday'] = [self.is_christmas(start_index) for start_index in daily_start_index_list]
        df_holidays['XMasWkB4'] = [self.is_christmas_week_before(start_index) for start_index in daily_start_index_list]
        df_holidays['XMasWkAft'] = [self.is_christmas_week_after(start_index) for start_index in daily_start_index_list]

        # Major holiday variables, daily and hourly fractions share the same
        df_major_holidays = pd.DataFrame(index=range(number_of_samples), columns=['MajorHolidays'])
        df_major_holidays['MajorHolidays'] = (df_holidays.ix[:, ['NewYearsHoliday',
                                                                 'MemorialDay',
                                                                 'LaborDay',
                                                                 'Thanksgiving',
                                                                 'FridayAfterThanks',
                                                                 'ChristmasHoliday']].sum(axis=1) > 0).apply(int).values

        # Weekday and weekend variables, daily and hourly fractions share the same
        df_wkdays = pd.DataFrame(index=range(number_of_samples), columns=['WkDay', 'WkEnd'])
        df_wkdays['WkEnd'] = df_dow['Saturday'] + df_dow['Sunday'] + df_major_holidays['MajorHolidays']
        df_wkdays['WkEnd'] = (df_wkdays['WkEnd'] > 0).apply(int).values
        df_wkdays['WkDay'] = 1 - df_wkdays['WkEnd']

        # Season variables, daily and hourly fractions share the same
        df_seasons = pd.DataFrame(index=range(number_of_samples), columns=['Summer', 'Winter'])
        df_seasons['Summer'] = [int(6 <= start_index.month <= 9) for start_index in daily_start_index_list]
        df_seasons['Winter'] = [int((start_index.month <= 2) or start_index.month == 12) for start_index in
                                daily_start_index_list]

        # Season/Day-Type interaction variables, daily and hourly fractions share the same
        season_list = df_seasons.columns
        dow_list = df_dow.columns

        season_dow_list = []
        for season in season_list:
            for dow in dow_list:
                season_dow_list.append(season + dow)

        df_season_dow = pd.DataFrame(index=range(number_of_samples), columns=season_dow_list)

        for season in season_list:
            for dow in dow_list:
                df_season_dow[season + dow] = [df_dow.ix[index, dow] * df_seasons.ix[index, season] for index in
                                               df_dow.index]

        pass

        recent_daily_temperature = X[:, range(self.window_size - 24, self.window_size)]
        recent_daily_start_hour = (dates.hour - stepAhead + 1) % 24

        temp_start = recent_daily_start_hour[0]
        temp_prefix = np.array(range(temp_start - 23, temp_start))

        recent_daily_start_hour = np.concatenate([temp_prefix, recent_daily_start_hour])
        recent_daily_start_hour = recent_daily_start_hour % 24

        recent_daily_hour_matrix = view_as_windows(recent_daily_start_hour, (24, ))

        # Weather Variables

        # AveDB, MornDB, AftDB, EveDB, daily and hourly fractions share the same
        df_dbs = pd.DataFrame(index=range(number_of_samples), columns=['AveDB', 'MornDB', 'AftDB', 'EveDB'])
        df_dbs['AveDB'] = recent_daily_temperature.mean(axis=1)

        for idx in range(len(df_dbs)):
            recent_daily_hours = recent_daily_hour_matrix[idx]

            recent_morning = recent_daily_hours[recent_daily_hours < 9]
            recent_morning = recent_morning[recent_morning >= 4]

            recent_afternoon = recent_daily_hours[recent_daily_hours < 17]
            recent_afternoon = recent_afternoon[recent_afternoon >= 11]

            recent_evening = recent_daily_hours[recent_daily_hours < 22]
            recent_evening = recent_evening[recent_evening >= 18]

            df_dbs.ix[idx, 'MornDB'] = recent_morning.min()
            df_dbs.ix[idx, 'AftDB'] = recent_afternoon.max()
            df_dbs.ix[idx, 'EveDB'] = recent_evening.max()

        # Slopes variables, daily and hourly fractions share the same, all from AveDB
        df_slopes = pd.DataFrame(index=range(number_of_samples),
                                 columns=['XColdSlope', 'ColdSlope', 'MidSlope', 'HotSlope', 'XHotSlope', 'XXHotSlope'])
        df_slopes['XColdSlope'] = 50 - df_dbs['AveDB']
        df_slopes['ColdSlope'] = 60 - df_dbs['AveDB']
        df_slopes['MidSlope'] = df_dbs['AveDB'] - 60
        df_slopes['MidSlope'].where(cond=df_slopes['MidSlope'] > 0, other=0, inplace=True)
        df_slopes['MidSlope'].where(cond=df_slopes['MidSlope'] < 10, other=10, inplace=True)
        df_slopes['HotSlope'] = df_dbs['AveDB'] - 70
        df_slopes['XHotSlope'] = df_dbs['AveDB'] - 80
        df_slopes['XXHotSlope'] = df_dbs['AveDB'] - 85
        df_slopes.where(cond=df_slopes > 0, other=0, inplace=True)

        # Weekend slope release variables, daily and hourly fractions share the same
        df_weekend_slopes = pd.DataFrame(index=range(number_of_samples), columns=['HotSlopeWkEnd', 'ColdSlopeWkEnd'])
        df_weekend_slopes['HotSlopeWkEnd'] = df_slopes['HotSlope'] * df_wkdays['WkEnd']
        df_weekend_slopes['ColdSlopeWkEnd'] = df_slopes['ColdSlope'] * df_wkdays['WkEnd']

        # Weather-based day-types, daily and hourly fractions share the same
        df_weather_based_day_types = pd.DataFrame(index=range(number_of_samples),
                                                  columns=['HotDay', 'ColdDay', 'MildDay'])

        df_weather_based_day_types['HotDay'] = (df_dbs['AveDB'] > 70).apply(int).values
        df_weather_based_day_types['ColdDay'] = (df_dbs['AveDB'] < 60).apply(int).values
        df_weather_based_day_types['MildDay'] = 1 - df_weather_based_day_types['HotDay'] - df_weather_based_day_types[
            'ColdDay']

        # Temperature Gain Variables, daily and hourly fractions share the same

        df_temp_gain = pd.DataFrame(index=range(number_of_samples), columns=['TempGain'])

        df_temp_gain['TempGain'] = df_dbs['AftDB'] - df_dbs['MornDB']

        df_temp_gain['HotTempGain'] = df_weather_based_day_types['HotDay'] * (df_temp_gain['TempGain'] - 17.7)
        df_temp_gain['ColdTempGain'] = df_weather_based_day_types['ColdDay'] * (df_temp_gain['TempGain'] - 17.7)
        df_temp_gain['MildTempGain'] = df_weather_based_day_types['MildDay'] * (df_temp_gain['TempGain'] - 17.7)

        # Time-of-Day Temperature Variables,
        # the full set of temperature variables used in the hourly fraction equation is shown below

        df_time_of_day = pd.DataFrame(index=range(number_of_samples),
                                      columns=['HotMornDB', 'HotAftDB', 'HotEveDB', 'WkEndHotMornDB', 'WkEndHotAftDB',
                                               'WkEndHotEveDB', 'MildMornDB', 'MildAftDB', 'MildEveDB', 'ColdMornDB',
                                               'ColdAftDB', 'ColdEveDB', 'WkEndColdMornDB', 'WkEndColdAftDB',
                                               'WkEndColdEveDB'])

        for temp_condition in ['Hot', 'Mild', 'Cold']:
            for tod in ['Morn', 'Aft', 'Eve']:
                df_time_of_day[temp_condition + tod + 'DB'] = df_weather_based_day_types[temp_condition + 'Day'] * \
                                                              df_dbs[
                                                                  tod + 'DB']

        for temp_condition in ['Hot', 'Cold']:
            for tod in ['Morn', 'Aft', 'Eve']:
                df_time_of_day['WkEnd' + temp_condition + tod + 'DB'] = df_wkdays['WkEnd'] * df_time_of_day[
                    temp_condition + tod + 'DB']

        df_features = pd.concat(
            [df_dow, df_holidays, df_major_holidays, df_wkdays, df_seasons, df_season_dow, df_dbs, df_slopes,
             df_weekend_slopes, df_weather_based_day_types, df_temp_gain, df_time_of_day], axis=1)

        df_features = df_features.loc[:, df_features.apply(pd.Series.nunique) != 1]

        recent_load_indices = [self.window_size * 2 - 24 * 7 + stepAhead - 1] + range(self.window_size - 24, self.window_size)
        X_loads = X[:, recent_load_indices]

        return np.hstack([df_features.as_matrix(), X_loads])

    def fit(self, X, y):
            self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    # New year: Jan 1st
    @staticmethod
    def is_new_year(date_time):
        if date_time.month == 1 and date_time.day == 1 and date_time.isoweekday() not in (6, 7):
            return 1
        elif date_time.month == 12 and date_time.day == 31 and date_time.isoweekday() == 5:
            return 1
        elif date_time.month == 1 and date_time.day == 2 and date_time.isoweekday() == 1:
            return 1
        return 0

    # MLK day: 3rd Monday in Jan
    @staticmethod
    def is_mlk(date_time):
        if date_time.month == 1 and date_time.isoweekday() == 1 and 14 < date_time.day < 22:
            return 1
        return 0

    # Presidents' day: 3rd Monday in Feb
    @staticmethod
    def is_president_day(date_time):
        if date_time.month == 2 and date_time.isoweekday() == 1 and 14 < date_time.day < 22:
            return 1
        return 0

    # Memorial day: last Monday in May
    @staticmethod
    def is_memorial_day(date_time):
        if date_time.month == 5 and date_time.isoweekday() == 1 and date_time.day > 24:
            return 1
        return 0

    # July 4th: July 4th
    @staticmethod
    def is_july_4th(date_time):
        if date_time.month == 7 and date_time.day == 4 and date_time.isoweekday() not in (6, 7):
            return 1
        elif date_time.month == 7 and date_time.day == 3 and date_time.isoweekday() == 5:
            return 1
        elif date_time.month == 7 and date_time.day == 5 and date_time.isoweekday() == 1:
            return 1
        return 0

    # Labor day: 1st monday in September
    @staticmethod
    def is_labor_day(date_time):
        if date_time.month == 9 and date_time.isoweekday() == 1 and date_time.day < 8:
            return 1
        return 0

    # Thanksgiving: 4th Thursday in November
    @staticmethod
    def is_thanksgiving(date_time):
        if date_time.month == 11 and date_time.isoweekday() == 4 and 21 < date_time.day < 29:
            return 1
        return 0

    # Friday after Thanksgiving: Friday after 4th Thursday in November
    @staticmethod
    def is_friday_after_thanksgiving(date_time):
        if date_time.month == 11 and date_time.isoweekday() == 5 and 22 < date_time.day < 30:
            return 1
        return 0

    # Christmas day: Dec 25th
    @staticmethod
    def is_christmas(date_time):
        if date_time.month == 12 and date_time.day == 25 and date_time.isoweekday() not in (6, 7):
            return 1
        elif date_time.month == 12 and date_time.day == 24 and date_time.isoweekday() == 5:
            return 1
        elif date_time.month == 12 and date_time.day == 26 and date_time.isoweekday() == 1:
            return 1
        return 0

    # Whole week before Christmas day: Dec 18th to Dec Dec 24th
    @staticmethod
    def is_christmas_week_before(date_time):
        if date_time.month == 12 and 18 <= date_time.day < 25:
            return 1
        return 0

    # Whole week after Christmas day: Dec 26th to next year's Jan 1st
    @staticmethod
    def is_christmas_week_after(date_time):
        if (date_time.month == 12 and 25 < date_time.day) or (date_time.month == 1 and date_time.day == 1):
            return 1
        return 0