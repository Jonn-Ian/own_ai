import pandas as pd

path_file = r"E:\Git\my_repos\own_ai\forecasting\csv\cleansed.csv"

df = pd.read_csv(path_file, header = 0)

df = df.sort_values(by = "year", ascending=True)
result = df.groupby("year")["revenue"].mean().reset_index()

d = []

for arr in range(len(result)-1):
    
    
    current_year = result["year"].iloc[arr + 1]
    current_year_revenue = result["revenue"].iloc[arr + 1]

    
    previous_year = result["year"].iloc[arr]
    previous_year_revenue = result["revenue"].iloc[arr]

    diff = current_year_revenue - previous_year_revenue
    print(f"{current_year} - {previous_year}: {current_year_revenue} - {previous_year_revenue} = {diff}")
    d.append(round(diff))

