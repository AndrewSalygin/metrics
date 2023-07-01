import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from kmodes.kprototypes import KPrototypes
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2

df = pd.read_csv("dataset.csv")

# убираем лишний класс
df = df.drop(df[df.Target == "Enrolled"].index)
df = df.reset_index()
df.drop(columns="index", inplace=True)

df.drop(
    columns=[
        "Curricular units 1st sem (without evaluations)",
        "Curricular units 2nd sem (without evaluations)",
        "Curricular units 1st sem (enrolled)",
        "Curricular units 2nd sem (enrolled)",
    ],
    inplace=True,
)

target = df["Target"]
le = preprocessing.LabelEncoder()

y = le.fit_transform(target)
X = df.drop(columns="Target")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=41
)

# увеличиваем количество данных
X_resampled_train, y_resampled_train = SMOTE(
    k_neighbors=3, random_state=41
).fit_resample(X_train, y_train)
X_resampled_test, y_resampled_test = SMOTE(k_neighbors=3, random_state=41).fit_resample(
    X_test, y_test
)

# нормализуем данные
standart_scaler_train = StandardScaler()
standart_scaler_train.fit(X_resampled_train)
X_standart_train = standart_scaler_train.transform(X_resampled_train)

standart_scaler_test = StandardScaler()
standart_scaler_test.fit(X_resampled_test)
X_standart_test = standart_scaler_test.transform(X_resampled_test)


def mahal(X, y):
    matrix_cov_rev = np.linalg.matrix_power(pd.DataFrame(X).cov(), -1)
    mean_values = np.mean(X, axis=0)
    mah_df99 = pd.DataFrame(X).copy()
    mah_df99["mah_dist"] = 0
    for i, values in mah_df99.iterrows():
        values = values[0:30]
        mah_df99.loc[i, "mah_dist"] = (
            mahalanobis(values, mean_values, matrix_cov_rev) ** 2
        )
    mah_border99 = chi2.ppf(0.99, 29)
    mah_outliers99 = mah_df99[mah_df99["mah_dist"] > mah_border99].index
    mah_df99 = pd.DataFrame(X).drop(index=mah_outliers99)
    mah_df99 = mah_df99.reset_index()
    mah_df99 = mah_df99.drop(columns="index")
    y_mah = pd.DataFrame(y).drop(index=mah_outliers99)
    return mah_df99, y_mah


# очищаем от выбросов
X_standart_train, y_resampled_train_standart = mahal(
    X_standart_train, y_resampled_train
)
X_standart_test, y_resampled_test_standart = mahal(X_standart_test, y_resampled_test)

X_resampled_train, y_resampled_train_n_standart = mahal(
    X_resampled_train, y_resampled_train
)
X_resampled_test, y_resampled_test_n_standart = mahal(
    X_resampled_test, y_resampled_test
)

# KNN
metrics = [
    ["euclidean", "sqeuclidean", "cosine", "manhattan", "canberra", "chebyshev"],
    ["jaccard", "hamming"],
]

for i in range(0, 2):
    knn = KNeighborsClassifier()
    k_range = list(range(1, 20))
    param_grid = dict(
        n_neighbors=k_range,
        algorithm=["auto", "ball_tree", "kd_tree", "brute"],
        metric=metrics[i],
    )

    # ищем наилучший классификатор
    grid = GridSearchCV(
        knn,
        param_grid,
        cv=10,
        scoring="f1_weighted",
        return_train_score=False,
        n_jobs=-1,
    )
    grid.fit(X_standart_train, y_resampled_train_standart[0])
    y_pred = grid.predict(X_standart_test)

# K-Means
kmeans = KMeans(n_clusters=2).fit(X_standart_train)
y_pred = kmeans.predict(X_standart_test)

# K-Prototype
X_prot = X_standart_train
X_prot.columns = X.columns.values

# указываем категориальные признаки
list_of_cat_features = [
    "Marital status",
    "Nacionality",
    "Displaced",
    "Gender",
    "International",
    "Father's qualification",
    "Mother's qualification",
    "Father's occupation",
    "Mother's occupation",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Scholarship holder",
    "Application mode",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
]
for feature in list_of_cat_features:
    X_prot[feature] = X_prot[feature].astype(object)


cat_columns = [
    X_prot.columns.get_loc(col) for col in list(X_prot.select_dtypes("object").columns)
]

k_prot = KPrototypes(n_jobs=-1, n_clusters=2, random_state=41)
k_prot.fit(X_prot, categorical=cat_columns)
y_pred = k_prot.predict(X_standart_test, categorical=cat_columns)

# K-Medoids
km_model = KMedoids(
    n_clusters=2,
    metric="jaccard",
    method="pam",
    init="k-medoids++",
    random_state=41,
).fit(X_resampled_train)
clusters = km_model.predict(X_resampled_test)