import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from Levenshtein import distance
from fastDamerauLevenshtein import damerauLevenshtein

df = pd.read_csv("TMDb_updated.csv")
df = df["title"].to_frame()
df = df["title"].str.lower().to_frame()
df["title"].astype(str)

def search_correction_lev(input_title, input_df):
    lev_distances = np.array([])
    input_title = input_title.lower()
    if len(input_title) <= 3:
        # оставляем все те строки, в которых есть входная строка
        df = input_df.loc[input_df["title"].str.contains(input_title, case=False)]
        # считаем расстояния между ними
        for title in df["title"]:
            lev_distances = np.append(lev_distances, distance(input_title, title))
        df["distances"] = lev_distances
        # сортируем, с наименьшим расстоянием возвращаем
        df = df.sort_values(by="distances")
        return df["title"].iloc[0:5]
    else:
        for title in input_df["title"]:
            lev_distances = np.append(lev_distances, distance(input_title, title))
        input_df["distances"] = lev_distances
        input_df = input_df.sort_values(by="distances")
        return input_df["title"].iloc[0:5]

def search_correction_dam_lev(input_title, input_df):
    lev_distances = np.array([])
    input_title = input_title.lower()
    if len(input_title) <= 3:
        # оставляем все те строки, в которых есть входная строка
        df = input_df.loc[input_df["title"].str.contains(input_title, case=False)]
        # считаем расстояния между ними
        for title in df["title"]:
            lev_distances = np.append(
                lev_distances, damerauLevenshtein(input_title, title, similarity=False)
            )
        df["distances"] = lev_distances
        # сортируем, с наименьшим расстоянием возвращаем
        df = df.sort_values(by="distances")
        return df["title"].iloc[0:5]
    else:
        for title in input_df["title"]:
            lev_distances = np.append(
                lev_distances, damerauLevenshtein(input_title, title, similarity=False)
            )
        input_df["distances"] = lev_distances
        input_df = input_df.sort_values(by="distances")
        return input_df["title"].iloc[0:5]

print(search_correction_lev("Aline", df))
print(search_correction_dam_lev("Aline", df))