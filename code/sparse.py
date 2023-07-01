import numpy as np
import pandas as pd
import nltk
import seaborn as sns
import re
import matplotlib.pyplot as plt
import nltk

from nltk.corpus import stopwords
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import classification_report
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import Counter
from sklearn.preprocessing import Normalizer


def input_data():
    # Загружаем датасет:
    df_init = pd.read_csv("Emotion_final.csv")

    # Выбрасываем пустые строчки:
    df_init = df_init.dropna()

    # Заменяем названия, для объединения датасетов:
    df_init = df_init.rename(columns={"Text": "content", "Emotion": "sentiment"})

    # Загружаем другой датасет:
    df_init_tweet = pd.read_csv("tweet_emotions.csv")
    df_tweet = df_init_tweet.drop(labels="tweet_id", axis=1)

    # Выбрасываем пустые строчки:
    df_tweet = df_tweet.dropna()

    # Заменяем название, для объединения датасетов:
    df_tweet.sentiment = df_tweet.sentiment.replace("happiness", "happy")

    # Соединим датасеты
    df_concatted = pd.DataFrame(pd.concat([df_tweet, df_init]))

    # Перезагружаем индексы:
    df_concatted.reset_index(drop=True, inplace=True)

    # Удалим все те эмоции, которые имеют слишком малое количество:
    to_delete = [
        "empty",
        "boredom",
        "relief",
        "enthusiasm",
        "fun",
        "hate",
        "love",
        "worry",
        "surprise",
    ]

    for val in to_delete:
        df_concatted.drop(
            df_concatted[df_concatted["sentiment"] == val].index, inplace=True
        )

    # Обновим индексы:
    df_concatted.reset_index(drop=True, inplace=True)

    return df_concatted


def clearing_data(X):
    # Представление данных для проверки эффективности расстояний на данных
    # с множественным пропуском
    nltk.download("wordnet")
    ps = PorterStemmer()
    wnl = WordNetLemmatizer()

    cleared_tweets_stem = []
    cleared_tweets_lem = []
    for i in range(0, len(X)):
        # очистка от упоминаний других пользователей
        tweets = re.sub("@[^ ]+", "", X[i])

        # очистка от хештегов
        tweets = re.sub("#[a-zA-Z0-9_]+", "", tweets)

        # очистка от кавычек
        tweets = re.sub("&quot", "", tweets)

        # очистка от внешних ссылок
        tweets = re.sub("https?:\/\/.*?[\s+]", "", tweets)

        # очистка от знаков препинаний
        tweets = re.sub("[^a-zA-Z]", " ", tweets)

        # приведение всего к одному регистру
        tweets = tweets.lower()

        # разделение твита на отдельные слова
        tweets = tweets.split()

        # очистка от шумовых слов (стоп-слов) ( предлоги, цифры, частицы и т. п.)
        # применение стеммера Портера
        tweets_stem = [
            ps.stem(word) for word in tweets if not word in stopwords.words("english")
        ]

        # лемматизация
        tweets_lem = [
            wnl.lemmatize(word)
            for word in tweets
            if not word in stopwords.words("english")
        ]

        # добавление его в общий список
        cleared_tweets_stem.append(tweets_stem)
        cleared_tweets_lem.append(tweets_lem)
    return cleared_tweets_stem, cleared_tweets_lem


def data_transformation(cleared_data):
    words_of_tweets_stem = [a for b in cleared_data[0] for a in b]
    words_of_tweets_lem = [a for b in cleared_data[1] for a in b]

    # подсчитаем количество всех слов
    d_stem = Counter(words_of_tweets_stem)
    d_lem = Counter(words_of_tweets_lem)

    # формируем датасеты
    d_stem_df = pd.DataFrame(list(d_stem))
    d_lem_df = pd.DataFrame(list(d_lem))

    for val in [[d_stem_df, d_stem], [d_lem_df, d_lem]]:
        val[0]["value"] = list(val[1].values())
        val[0].drop(val[0][(val[0].value <= 1)].index, inplace=True)
        val[0].reset_index(inplace=True)
        val[0].rename({0: "word"}, axis=1, inplace=True)
        val[0] = val[0].sort_values(by="value", ascending=False)

    # формируем строки в которых указываем количество повторений слова
    words_of_tweet_lst_stem = []
    words_of_tweet_lst_lem = []

    list_of_d_stem_df = list(d_stem_df.word)
    list_of_d_lem_df = list(d_lem_df.word)

    for val in [
        [words_of_tweet_lst_stem, cleared_data[0], list_of_d_stem_df],
        [words_of_tweet_lst_lem, cleared_data[1], list_of_d_lem_df],
    ]:
        for i in range(0, len(val[1])):
            d = dict()
            for inner_val in val[1][i]:
                # если слово присутствует в словаре
                if inner_val in d:
                    if inner_val in val[2]:
                        d[inner_val] += 1
                # если слово отсутствует в словаре
                else:
                    if inner_val in val[2]:
                        d[inner_val] = 1
            val[0].append(d)
    return words_of_tweet_lst_stem, words_of_tweet_lst_lem


# Представление данных в удобочитаемом виде для модели
df = input_data()
X = df["content"]
y = df["sentiment"]

cleared_data = clearing_data(X)

transformed_data = data_transformation(cleared_data)

# Создание датасетов для обучения:
df_stem = pd.DataFrame.from_dict(transformed_data[0])
df_stem.fillna(0, inplace=True)

df_lem = pd.DataFrame.from_dict(transformed_data[1])
df_lem.fillna(0, inplace=True)

# Кодирование таргета
le = preprocessing.LabelEncoder()
target = le.fit_transform(y)

X_train_stem, X_test_stem, y_train_stem, y_test_stem = train_test_split(
    df_stem, target, test_size=0.1, random_state=41
)

X_train_lem, X_test_lem, y_train_lem, y_test_lem = train_test_split(
    df_lem, target, test_size=0.1, random_state=41
)

# Балансирование классов
X_resampled_stem_train, y_resampled_train_stem = SMOTE(
    k_neighbors=25, random_state=41
).fit_resample(X_train_stem, y_train_stem)
X_resampled_lem_train, y_resampled_train_lem = SMOTE(
    k_neighbors=25, random_state=41
).fit_resample(X_train_lem, y_train_lem)
X_resampled_stem_test, y_resampled_test_stem = SMOTE(
    k_neighbors=25, random_state=41
).fit_resample(X_test_stem, y_test_stem)
X_resampled_lem_test, y_resampled_test_lem = SMOTE(
    k_neighbors=25, random_state=41
).fit_resample(X_test_lem, y_test_lem)

# Нормирование данных
transformer_stem_train = Normalizer().fit(X_resampled_stem_train)
transformer_lem_train = Normalizer().fit(X_resampled_lem_train)
transformer_stem_test = Normalizer().fit(X_resampled_stem_test)
transformer_lem_test = Normalizer().fit(X_resampled_lem_test)

X_normalized_stem_train = transformer_stem_train.transform(X_resampled_stem_train)
X_normalized_lem_train = transformer_lem_train.transform(X_resampled_lem_train)
X_normalized_stem_test = transformer_stem_test.transform(X_resampled_stem_test)
X_normalized_lem_test = transformer_lem_test.transform(X_resampled_lem_test)

metrics = [
    "jaccard",
    "euclidean",
    "sqeuclidean",
    "cosine",
    "manhattan",
    "canberra",
    "chebyshev",
    "correlation",
    "hamming",
]

for val in metrics:
    print("Stemming")
    knn = KNeighborsClassifier(n_neighbors=8, metric=val, n_jobs=-1)
    knn.fit(X_normalized_stem_train, y_resampled_train_stem)
    y_pred = knn.predict(X_normalized_stem_test)
    report = classification_report(y_resampled_test_stem, y_pred)
    print(report)

    print("Lemmatization")
    knn = KNeighborsClassifier(n_neighbors=8, metric=val, n_jobs=-1)
    knn.fit(X_normalized_lem_train, y_resampled_train_lem)
    y_pred = knn.predict(X_normalized_lem_test)
    report = classification_report(y_resampled_test_lem, y_pred)
    print(report)