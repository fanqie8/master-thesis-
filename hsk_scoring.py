import pandas as pd
import jieba
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Load Data
# df_data = pd.read_excel('./data/Dataset_MultipleTopic.xlsx',index_col=None, header=0)
df_data = pd.read_excel('./data/Dataset_Singletopic_greenfood.xlsx',index_col=None, header=0)


# Tokenization
def preprocessing_sentence(x):
    words = jieba.cut(str(x).strip())
    return ' '.join(words)


# TF-IDF Vectorization
def tf_idf_vectorize():
    x_train, x_test, y_train, y_test = train_test_split(df_data['CONTENT_NEW'], df_data['SCORE'], test_size=0.1)

    x_train = x_train.apply(lambda x: preprocessing_sentence(x))
    x_test = x_test.apply(lambda x: preprocessing_sentence(x))

    tf = TfidfVectorizer()
    x_train = tf.fit_transform(x_train).toarray()
    x_test = tf.transform(x_test).toarray()
    return x_train, x_test, y_train, y_test


# top N n-grams Vectorization
def ngram_vectorize(N=100, min_n=1, max_n=3):
    x_train, x_test, y_train, y_test = train_test_split(df_data['CONTENT_NEW'], df_data['SCORE'], test_size=0.1)

    x_train = x_train.apply(lambda x: preprocessing_sentence(x))
    x_test = x_test.apply(lambda x: preprocessing_sentence(x))

    cv = CountVectorizer(ngram_range=(min_n, max_n), max_features=N)
    x_train = cv.fit_transform(x_train).toarray()
    x_test = cv.transform(x_test).toarray()
    return x_train, x_test, y_train, y_test


# LogisticRegression
def logistic_regression(vector_typ):

    if vector_typ == 'tf-idf':
        x_train, x_test, y_train, y_test = tf_idf_vectorize()
    elif vector_typ == 'ngram':
        x_train, x_test, y_train, y_test = ngram_vectorize()
    else:
        print('Invalid Vector Typ')
        return

    lr = LogisticRegression()
    lr.fit(x_train, y_train)

    train_score = lr.score(x_train, y_train)
    print("Train Accuracy:", train_score)

    y_predict = lr.predict(x_test)
    test_score = metrics.accuracy_score(y_test, y_predict)
    print("Test Accuracy:", test_score)

    confusion_mat = metrics.confusion_matrix(y_test, y_predict)
    print(confusion_mat)


if __name__ == "__main__":
    logistic_regression('tf-idf')
    logistic_regression('ngram')
