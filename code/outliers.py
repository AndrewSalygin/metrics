import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
from sklearn.cluster import DBSCAN
from statsmodels.formula.api import ols

df = pd.read_csv("home_data.csv")

df["bathrooms"] = df["bathrooms"].astype(int)
df["floors"] = df["floors"].astype(int)

dfs = (df.copy(), df.copy())

dfs[0].drop(
    columns=[
        "date",
        "id",
        "sqft_living15",
        "sqft_lot15",
        "zipcode",
        "price",
        "yr_renovated",
    ],
    inplace=True,
)
dfs[1].drop(
    columns=[
        "date",
        "id",
        "sqft_living15",
        "sqft_lot15",
        "zipcode",
        "sqft_living",
        "price",
        "grade",
        "sqft_above",
        "yr_built",
        "waterfront",
        "yr_renovated",
    ],
    inplace=True,
)

target = df["price"]

# очистка от выбросов расстоянием Махаланобиса
def mahalanobisCleaning(dfs):
    # вычисление ковариационной матрицы
    inv_cov_mats = [
        np.linalg.matrix_power(dfs[0].cov(), -1),
        np.linalg.matrix_power(dfs[1].cov(), -1),
    ]
    mean_values = [np.mean(dfs[0], axis=0), np.mean(dfs[1], axis=0)]
    mah_dfs = [dfs[0].copy(), dfs[1].copy()]
    mah_dfs[0]["mah_dist"] = 0
    mah_dfs[1]["mah_dist"] = 0

    columns = [14, 9]

    mah_borders = (chi2.ppf(0.99, columns[0]), chi2.ppf(0.99, columns[1]))
    mah_outliers = [pd.DataFrame(), pd.DataFrame()]

    # вычисление расстояния
    for i in range(0, 2):
        for j, values in mah_dfs[i].iterrows():
            values = values[0 : columns[i]]
            mah_dfs[i].loc[j, "mah_dist"] = (
                mahalanobis(values, mean_values[i], inv_cov_mats[i]) ** 2
            )

        # находятся выбросы
        mah_outliers[i] = mah_dfs[i][mah_dfs[i]["mah_dist"] > mah_borders[i]].index

        # удаляются выбросы
        print("Количество удалённых строк:", len(mah_outliers[i]))
        print(f"Это {len(mah_outliers[i]) / len(mah_dfs[i]) * 100}% датасета")

        mah_dfs[i].drop(index=mah_outliers[i], inplace=True)
        mah_dfs[i].reset_index(inplace=True)
        mah_dfs[i].drop(columns=["index", "mah_dist"], inplace=True)


# очистка от выбросов расстоянием Кука
def cooksCleaning(dfs, target):
    cooks_dfs = (dfs[0].copy(), dfs[1].copy())

    # выбираем колонки
    Xs = [
        cooks_dfs[0][cooks_dfs[0].columns[0:14]],
        cooks_dfs[1][cooks_dfs[1].columns[0:9]],
    ]

    Xs[0] = sm.tools.tools.add_constant(Xs[0])
    Xs[1] = sm.tools.tools.add_constant(Xs[1])

    # обучение моделей
    models = [
        sm.regression.linear_model.OLS(np.asarray(target), Xs[0]).fit(),
        sm.regression.linear_model.OLS(np.asarray(target), Xs[1]).fit(),
    ]

    inflce_lsts = [[], []]

    # получение расстояний Кука
    for i in range(0, 2):
        inflce = models[i].get_influence()
        inflce_lsts[i] = inflce.cooks_distance[0]
        cooks_dfs[i]["influence"] = inflce_lsts[i]

    # удаляются выбросы
    cooks_borders = [4 / len(cooks_dfs[0]), 4 / len(cooks_dfs[1])]
    cooks_outliers = [pd.DataFrame(), pd.DataFrame()]
    for i in range(0, 2):
        cooks_outliers[i] = cooks_dfs[i][cooks_dfs[i]["influence"] > cooks_borders[i]]
        print("Количество удалённых строк:", len(cooks_outliers[i]))
        print(f"Это {len(cooks_outliers[i]) / len(cooks_dfs[i]) * 100}% датасета")

        cooks_dfs[i].drop(index=cooks_outliers[i].index, axis=0, inplace=True)
        cooks_dfs[i].reset_index(inplace=True)
        cooks_dfs[i].drop(columns=["index", "influence"], inplace=True)


# очистка от выбросов методом DBSCAN
def dbscanCleaning(dfs, border1, border2):
    dbscan_1 = DBSCAN()
    dbscan_2 = DBSCAN()

    dbscans = [dbscan_1.fit(dfs[0]), dbscan_2.fit(dfs[1])]
    unique_values = [[], []]
    count_of_outliers = [0, 0]
    removed_indexes = [[], []]
    len_of_mah_outliers = [border1, border2]

    # настройка DBSCAN
    for i in range(0, 2):
        eps = 0.5
        for value in dbscans[i].labels_:
            if value not in unique_values[i]:
                unique_values[i].append(value)
            if value == -1:
                count_of_outliers[i] += 1

        # настройка eps
        while count_of_outliers[i] > len_of_mah_outliers[i]:
            dbscans[i] = DBSCAN(eps=eps)
            dbscans[i].fit(dfs[i])
            labels_df = pd.DataFrame(dbscans[i].labels_, columns=["labels"])
            labels_df.index = dfs[i].index
            removed_df = labels_df[labels_df["labels"] == -1]
            removed_indexes[i] = removed_df.index
            count_of_outliers[i] = len(removed_indexes[i])
            eps += 100

    print("Количество удалённых строк:", count_of_outliers[0])
    print(f"Это {count_of_outliers[0] / len(dfs[0]) * 100}% датасета")

    print("Количество удалённых строк:", count_of_outliers[1])
    print(f"Это {count_of_outliers[1] / len(dfs[1]) * 100}% датасета")

    dbscan_dfs = [pd.DataFrame(), pd.DataFrame()]
    for i in range(0, 2):
        dbscan_dfs[i] = dfs[i].drop(index=removed_indexes[i])
        dbscan_dfs[i].reset_index(inplace=True)
        dbscan_dfs[i].drop(columns="index", inplace=True)

mahalanobisCleaning(dfs)
cooksCleaning(dfs, target)
dbscanCleaning(dfs, 1984, 1106)