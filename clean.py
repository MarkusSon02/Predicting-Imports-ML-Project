import pandas as pd
from IPython.display import display
import numpy as np

# Ingesting data
GEM_df = pd.read_excel("data/GEM_data.xlsx")
WDI_df = pd.read_excel("data/WDI_data.xlsx")

GEM_df = GEM_df.drop(columns=['Time Code', 'CPI Price, % y-o-y, median weighted, seas. adj., [CPTOTSAXMZGY]', 'Exports Merchandise, Customs, current US$, millions, seas. adj. [DXGSRMRCHSACD]', 'Imports Merchandise, Customs, current US$, millions, seas. adj. [DMGSRMRCHSACD]', 'Industrial Production, constant US$, seas. adj.,, [IPTOTSAKD]'])
WDI_df = WDI_df.drop(columns=['Time Code', 'Country Name', "Educational attainment, at least Master's or equivalent, population 25+, total (%) (cumulative) [SE.TER.CUAT.MS.ZS]", "Educational attainment, Doctoral or equivalent, population 25+, total (%) (cumulative) [SE.TER.CUAT.DO.ZS]", "Educational attainment, at least Bachelor's or equivalent, population 25+, total (%) (cumulative) [SE.TER.CUAT.BA.ZS]", "Educational attainment, at least completed lower secondary, population 25+, total (%) (cumulative) [SE.SEC.CUAT.LO.ZS]", "Educational attainment, at least completed post-secondary, population 25+, total (%) (cumulative) [SE.SEC.CUAT.PO.ZS]", "Educational attainment, at least completed primary, population 25+ years, total (%) (cumulative) [SE.PRM.CUAT.ZS]", "Educational attainment, at least completed short-cycle tertiary, population 25+, total (%) (cumulative) [SE.TER.CUAT.ST.ZS]", "Educational attainment, at least completed upper secondary, population 25+, total (%) (cumulative) [SE.SEC.CUAT.UP.ZS]",])
WDI_df = WDI_df.rename({"Time": "Year"}, axis=1)

GEM_df = GEM_df.reset_index(drop=True)
WDI_df = WDI_df.reset_index(drop=True)

# Reformating data for joining operation
GEM_df['Year'] = GEM_df['Time'].str[:4]
GEM_df['Time'] = GEM_df['Time'].astype(str)
GEM_df["Year"] = pd.to_numeric(GEM_df["Year"], errors='coerce')
GEM_df['Country Code'] = GEM_df['Country Code'].astype(str)
WDI_df["Year"] = pd.to_numeric(WDI_df["Year"], errors='coerce')
WDI_df['Country Code'] = WDI_df['Country Code'].astype(str)

# Joining data
df = pd.merge(GEM_df, WDI_df, on=["Year", "Country Code"], how="left")
df.to_excel("data/joined_data.xlsx")

# print(GEM_df[['Year', 'Country Code']].dtypes)
# print(WDI_df.reset_index()[['Year', 'Country Code']].dtypes)

# print("GEM_df index:", GEM_df.index)
# print("GEM_df columns:", GEM_df.columns)

# print("WDI_df index:", WDI_df.index)
# print("WDI_df columns:", WDI_df.columns)

# count_countries = df["Country Code"].value_counts()
# print(len(count_countries))
# display(df.head(3))
# df.to_excel("data/Joined_data.xlsx")

joined_df = df

for col in joined_df.columns:
    if col not in ["Time", "Country Code", "Country"]:
        joined_df[col] = pd.to_numeric(joined_df[col], errors='coerce')

joined_df['Time'] = joined_df['Time'].astype("string")
joined_df['Country Code'] = joined_df['Country Code'].astype("string")
joined_df['Country'] = joined_df['Country'].astype("string")

count_countries = joined_df["Country Code"].value_counts()
print("Number of countries:", len(count_countries))

# Removing countries with inconsistent data
joined_df = joined_df[~joined_df['Country Code'].isin(['EST', 'MLT', 'LTU', 'HRV', 'CYP', 'CZE', 'GRC', 'SVK', 'AUT', 'FRA', 'HUN', 'YUG', 'CHE', 'GEO', 'RUS', 'BGR', 'SVN', 'DNK', 'ISL', 'GBR'])]
joined_df = joined_df[~joined_df['Country Code'].isin(['ARE', 'ARG', 'AUS', 'BRA', 'JPN', 'MYS', 'NZL', 'PHL', 'USA'])]
joined_df['Year'] = pd.to_numeric(joined_df['Year'], errors='coerce')

# Removing time frames with inconsistent data
joined_df = joined_df[joined_df['Year'] > 2009]
joined_df = joined_df[joined_df['Year'] < 2019]
joined_df = joined_df.sort_values(by=["Country Code", "Time"])
joined_df['Month'] = joined_df["Time"].str[-2:]
joined_df['Month'] = pd.to_numeric(joined_df['Month'], errors='coerce')
joined_df['Time'] = pd.to_datetime(joined_df['Time'].str.replace('M', '-') + '-01', errors='coerce')


print("Count of null values: ", joined_df.isna().sum().sum())

count_countries = joined_df["Country Code"].value_counts()
print("Number of countries:", len(count_countries))
joined_df.to_excel("data/semi_cleaned_data.xlsx", index=False)

# Filling the few cells that have null values (very few cells had null data)
for col in joined_df.columns:
    joined_df[col] = joined_df.groupby(["Country Code"])[col].ffill()


joined_df = joined_df.reset_index()
# print(joined_df.dtypes)
joined_df.to_excel("data/cleaned_data.xlsx", index=False)


# print(joined_df.isna().sum().sum())
# print(joined_df.shape)
# print(joined_df.groupby("Country Code").apply(lambda x: x.isnull().sum().sum()))
# # display(joined_df.head(3))

final_df = joined_df

# Data enrichment and transformation
final_df['Net Exports not seas. adj'] = final_df['Exports Merchandise, Customs, current US$, millions, not seas. adj. [DXGSRMRCHNSCD]'] - final_df['Imports Merchandise, Customs, current US$, millions, not seas. adj. [DMGSRMRCHNSCD]']
final_df['Lag Net Exports not seas. adj'] = final_df.groupby("Country Code")['Net Exports not seas. adj'].shift(1)
final_df['Lag Exports not seas. adj'] = final_df.groupby("Country Code")['Exports Merchandise, Customs, current US$, millions, not seas. adj. [DXGSRMRCHNSCD]'].shift(1)
final_df['Lag Imports not seas. adj'] = final_df.groupby("Country Code")['Imports Merchandise, Customs, current US$, millions, not seas. adj. [DMGSRMRCHNSCD]'].shift(1)
final_df["Official Exchange Rate percent change"] = final_df.groupby("Country Code")["Official exchange rate, LCU per USD, period average,, [DPANUSLCU]"].pct_change() * 100
final_df["Nominal Effective Exchange Rate percent change"] = final_df.groupby("Country Code")["Nominal Effective Exchange Rate,,,, [NEER]"].pct_change() * 100
final_df["Real Effective Exchange Rate percent change"] = final_df.groupby("Country Code")["Real Effective Exchange Rate,,,, [REER]"].pct_change() * 100
final_df["Net Exports percent change"] = final_df.groupby("Country Code")["Net Exports not seas. adj"].pct_change() * 100
final_df["Exports percent change"] = final_df.groupby("Country Code")['Exports Merchandise, Customs, current US$, millions, not seas. adj. [DXGSRMRCHNSCD]'].pct_change() * 100
final_df["Imports percent change"] = final_df.groupby("Country Code")['Imports Merchandise, Customs, current US$, millions, not seas. adj. [DMGSRMRCHNSCD]'].pct_change() * 100
final_df['Lag Official Exchange Rate percent change'] = final_df.groupby("Country Code")['Official Exchange Rate percent change'].shift(1)
final_df.reset_index()
print(final_df.groupby("Country Code").apply(lambda x: x.isnull().sum().sum()))
final_df = final_df.dropna()


# Taking log of variables with very large values
final_df['ln_Labor'] = np.log(final_df['Labor force, total [SL.TLF.TOTL.IN]'])
final_df['ln_Industial_Production not seas. adj'] = np.log(final_df['Industrial Production, constant US$,,, [IPTOTNSKD]'])
final_df['ln_GDP'] = np.log(final_df['GDP (current US$) [NY.GDP.MKTP.CD]'])
final_df['ln_GDP_per_capita'] = np.log(final_df['GDP per capita (current US$) [NY.GDP.PCAP.CD]'])
final_df['ln_exports'] = np.log(final_df['Exports Merchandise, Customs, current US$, millions, not seas. adj. [DXGSRMRCHNSCD]'])
final_df['ln_imports'] = np.log(final_df['Imports Merchandise, Customs, current US$, millions, not seas. adj. [DMGSRMRCHNSCD]'])
final_df['ln_lag_exports'] = np.log(final_df['Lag Exports not seas. adj'])
final_df['ln_lag_imports'] = np.log(final_df['Lag Imports not seas. adj'])
final_df['ln_exports_change'] = final_df['ln_exports'] - final_df['ln_lag_exports']
final_df['ln_imports_change'] = final_df['ln_imports'] - final_df['ln_lag_imports']

final_df = final_df.dropna()
print(final_df.groupby("Country Code").apply(lambda x: x.isnull().sum().sum()))

print("Number of samples", final_df.shape[0])

final_df.to_excel("data/final_data.xlsx")
