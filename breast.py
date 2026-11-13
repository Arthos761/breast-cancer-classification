import pandas as pd
import optuna
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import (train_test_split,
                                     cross_val_score,
                                     StratifiedKFold)

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (accuracy_score,
                             f1_score, recall_score,
                             precision_score, confusion_matrix,
                             ConfusionMatrixDisplay,
                             make_scorer)



# 1. Первый взгляд и разделение данных

df = pd.read_csv('data.csv')
print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum())
print(df['diagnosis'].value_counts())

df = df.drop(columns=['id', 'Unnamed: 32'])

# Разделение на группы(x/y)
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']
y = (df['diagnosis'] == 'M').astype(int)

print(X.shape)
print(X.describe().T)

print(df.groupby('diagnosis')[['radius_mean',
                         'texture_mean',
                         'perimeter_mean',
                         'area_mean']].mean())

# График признаков: какая корреляция между нимим
corr = X.corr()
plt.figure(figsize=(16, 12))
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.tight_layout()

# сохраняем график
plt.savefig('charts/correlation_heatmap.png',
            dpi=300,
            bbox_inches='tight')

high_corr = corr.abs().unstack()

high_corr = high_corr[
    (high_corr > 0.9) & (high_corr < 1.0)
]

print(high_corr.sort_values(ascending=False))

# График разделения классов про среднему радиусу
sns.histplot(
    data=df,
    x='radius_mean',
    hue='diagnosis',
    kde=True
)

plt.title('Распределение среднего радиуса по классам')
plt.xlabel('Средний радиус (radius_mean)')
plt.ylabel('Количество наблюдений')

plt.tight_layout()

# Сохранение графика
plt.savefig(
    'charts/radius_mean_distribution.png',
    dpi=300,
    bbox_inches='tight'
)


# Раздление на тренировочные и тестовые данные
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# 2. Предобработка данных
preprocessing = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(score_func=f_classif, k=20)),
])


# 3. Строим модели и обучаем
lr_base = Pipeline([
    ('preprocessing', preprocessing),
    ('lr', LogisticRegression())
])

svm_base  = Pipeline([
    ('preprocessing', preprocessing),
    ('svm', SVC(kernel='linear', ))
])

ran_base  = Pipeline([
    ('preprocessing', preprocessing),
    ('ran', RandomForestClassifier())
])



lr_base.fit(X_train, y_train)
svm_base.fit(X_train, y_train)
ran_base.fit(X_train, y_train)



# 4. Перекрестная проверка
f1_m_scorer = make_scorer(f1_score, pos_label= 1)

cv1 = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores_lr = cross_val_score(
    lr_base , X_train, y_train,
    cv=cv1,
    scoring=f1_m_scorer
)
print(f"РЕЗУЛЬТАТЫ LR С cross_val_score:\n ")
print(f"Scores: {scores_lr}")
print(f"Mean F1: {scores_lr.mean():.3f}")
print(f"Std: {scores_lr.std():.3f}\n")



scores_svm = cross_val_score(
    svm_base ,  X_train, y_train,
    cv=cv1,
    scoring=f1_m_scorer
)
print(f"РЕЗУЛЬТАТЫ SVC С cross_val_score:\n")
print(f"Scores: {scores_svm}")
print(f"Mean F1: {scores_svm.mean():.3f}")
print(f"Std: {scores_svm.std():.3f}\n")



scores_ran = cross_val_score(
    ran_base , X_train, y_train,
    cv=cv1,
    scoring=f1_m_scorer
)
print(f"РЕЗУЛЬТАТЫ RAN С cross_val_score:\n ")
print(f"Scores: {scores_ran}")
print(f"Mean F1: {scores_ran.mean():.3f}")
print(f"Std: {scores_ran.std():.3f}")

# 5. Подбор гиперпараметров с помошью Optuna

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

def objective_lr(trial):

    model = LogisticRegression(
        C=trial.suggest_float("lr_C", 1e-4, 1e3, log=True),
        solver=trial.suggest_categorical(
            "lr_solver", ["liblinear", "lbfgs", "saga"]
        ),
        class_weight=trial.suggest_categorical(
            "lr_class_weight", [None, "balanced"]
        ),
        max_iter=trial.suggest_categorical(
            "lr_max_iter", [1000, 2000, 5000]
        ),
        random_state=42
    )
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(f_classif, k=20)),
        ("model", model)
    ])
    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=f1_m_scorer,
        n_jobs=-1
    )
    return scores.mean()


def objective_svm(trial):
    kernel = trial.suggest_categorical(
        "svc_kernel", ["linear", "rbf", "poly", "sigmoid"]
    )
    params = {
        "C": trial.suggest_float("svc_C", 1e-4, 1e3, log=True),
        "kernel": kernel,
        "class_weight": trial.suggest_categorical(
        "svc_class_weight", [None, "balanced"]
        )
    }
    if kernel in ["rbf", "poly", "sigmoid"]:
        params["gamma"] = trial.suggest_categorical(
            "svc_gamma_mode", ["scale", "auto"]
        )
        # Альтернатива: исследовать числовой gamma отдельно.

    if kernel == "poly":
        params["degree"] = trial.suggest_int(
            "svc_degree", 2, 5
        )
    model = SVC(**params)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(f_classif, k=20)),
        ("model", model)
    ])
    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=f1_m_scorer,
        n_jobs=-1
    )
    return scores.mean()

def objective_ran(trial):
    model = RandomForestClassifier(
        n_estimators=trial.suggest_int(
            "ran_n_estimators", 100, 1000, step=100
        ),
        max_depth=trial.suggest_categorical(
            "ran_max_depth", [None, 3, 5, 10, 15, 20, 30]
        ),
        min_samples_split=trial.suggest_int(
            "ran_min_samples_split", 2, 20
        ),
        min_samples_leaf=trial.suggest_int(
            "ran_min_samples_leaf", 1, 10
        ),
        max_features=trial.suggest_categorical(
             "ran_max_features", ["sqrt", "log2", None]
        ),
        class_weight=trial.suggest_categorical(
            "ran_class_weight",
            [None, "balanced", "balanced_subsample"]
        ),
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(f_classif, k=20)),
        ("model", model)
    ])
    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=f1_m_scorer,
        n_jobs=-1
    )
    return scores.mean()

study_lr = optuna.create_study(direction="maximize")
study_lr.optimize(objective_lr, n_trials=100)

study_svc = optuna.create_study(direction="maximize")
study_svc.optimize(objective_svm, n_trials=100)

study_rf = optuna.create_study(direction="maximize")
study_rf.optimize(objective_ran, n_trials=100)

print('Результаты LR с параметрами:\n')
print(study_lr.best_params)
print(study_lr.best_value)

print('Результаты SVC с параметрами:\n')
print(study_svc.best_params)
print(study_svc.best_value)

print('Результаты RAN с параметрами:\n')
print(study_rf.best_params)
print(study_rf.best_value)

print('\n')
print('\n')

# 6. Вторая вариация моделей с подобранными параметрами
lr_tuned = Pipeline([
    ("scaler", StandardScaler()),
    ("selector", SelectKBest(f_classif, k=20)),
    ("model", LogisticRegression(
        C=study_lr.best_params['lr_C'],
        solver=study_lr.best_params['lr_solver'],
        max_iter=study_lr.best_params['lr_max_iter'],
        class_weight=study_lr.best_params['lr_class_weight'],
        random_state=42
    ))
])

svm_tuned = Pipeline([
    ("scaler", StandardScaler()),
    ("selector", SelectKBest(f_classif, k=20)),
    ("model", SVC(
        C=study_svc.best_params['svc_C'],
        kernel=study_svc.best_params['svc_kernel'],
        class_weight=study_svc.best_params['svc_class_weight'],
        random_state=42
    ))
])

ran_tuned = Pipeline([
    ("scaler", StandardScaler()),
    ("selector", SelectKBest(f_classif, k=20)),
    ("model", RandomForestClassifier(
        n_estimators=study_rf.best_params['ran_n_estimators'],
        max_depth=study_rf.best_params['ran_max_depth'],
        min_samples_split=study_rf.best_params['ran_min_samples_split'],
        min_samples_leaf=study_rf.best_params['ran_min_samples_leaf'],
        max_features=study_rf.best_params['ran_max_features'],
        class_weight=study_rf.best_params['ran_class_weight'],
        random_state=42,
        n_jobs=-1
    ))
])

lr_tuned.fit(X_train, y_train)
y_pred_lr = lr_tuned.predict(x_test)

svm_tuned.fit(X_train, y_train)
y_pred_svm = svm_tuned.predict(x_test)

ran_tuned.fit(X_train, y_train)
y_pred_ran = ran_tuned.predict(x_test)


# создали для просмтора признаков
selector = lr_base .named_steps['preprocessing'].named_steps['selector']
selected_features = X.columns[selector.get_support()]


# 7. Финальная проверерка на test выборке
print('\n\n')
print("ФИНАЛЬНАЯ ПРОВЕРКА!!!")
print('\n\n')

print(f'Использоваемы признаки для обучения:\n '
      f'{selected_features} \n ')

print(f'Результаты LogRegression:\n Точность: {accuracy_score(y_test,y_pred_lr)*100}%\n '
      f'Precision: {precision_score(y_test,y_pred_lr)*100}%\n '
      f'Recall: {recall_score(y_test,y_pred_lr)*100}%\n '
      f'F1-Score: {f1_score(y_test,y_pred_lr)*100}%\n ')


print(f'Результаты SVC:\n Точность: {accuracy_score(y_test,y_pred_svm)*100}%\n '
      f'Precision: {precision_score(y_test,y_pred_svm)*100}%\n '
      f'Recall: {recall_score(y_test,y_pred_svm)*100}%\n '
      f'F1-Score: {f1_score(y_test,y_pred_svm)*100}%\n ')


print(f'Результаты RandomForestClassifier:\n Точность: {accuracy_score(y_test,y_pred_ran)*100}%\n '
      f'Precision: {precision_score(y_test,y_pred_ran)*100}%\n '
      f'Recall: {recall_score(y_test,y_pred_ran)*100}%\n '
      f'F1-Score: {f1_score(y_test,y_pred_ran)*100}%\n ')



# 8. Матрица ошибок для финальных моделей

#Логистическая регрессия
cm_lr = confusion_matrix(y_test, y_pred_lr)
disp1 = ConfusionMatrixDisplay(
    confusion_matrix=cm_lr,
    display_labels=['ДоброКачественая', "ЗлоКачественная"]
)
disp1.plot(cmap='Blues', values_format='d')
plt.title('Logistic Regression — Confusion Matrix')
plt.tight_layout()
plt.savefig('charts/confusion_matrix_lr.png', dpi=300, bbox_inches='tight')


#SVC
cm_svm = confusion_matrix(y_test, y_pred_svm)
disp2 = ConfusionMatrixDisplay(
    confusion_matrix=cm_svm,
    display_labels=['ДоброКачественная', "ЗлоКачественная"]
)
disp2.plot(cmap='Blues', values_format='d')
plt.title('SVC — Confusion Matrix')
plt.tight_layout()
plt.savefig('charts/confusion_matrix_svc.png', dpi=300, bbox_inches='tight')

#RandomForestCl
cm_ran = confusion_matrix(y_test, y_pred_ran)
disp3 = ConfusionMatrixDisplay(
    confusion_matrix=cm_ran,
    display_labels=['ДоброКачественная', "ЗлоКачественная"]
)
disp3.plot(cmap='Blues', values_format='d')
plt.title('Random Forest — Confusion Matrix')
plt.tight_layout()
plt.savefig('charts/confusion_matrix_rf.png', dpi=300, bbox_inches='tight')


# Сравнение финальных моделей по F1-score

models = ['Logistic Regression', 'SVC', 'Random Forest']

f1_scores = [
    f1_score(y_test,y_pred_ran)*100,
    f1_score(y_test,y_pred_svm)*100,
    f1_score(y_test,y_pred_ran)*100
]

plt.figure(figsize=(10, 6))

plt.bar(models, f1_scores)

plt.title('Сравнение моделей по F1-score')
plt.xlabel('Модели')
plt.ylabel('F1-score')

plt.ylim(0.955, 1.0)
plt.tight_layout()
plt.savefig('charts/models_f1-score.png', dpi=300, bbox_inches='tight')