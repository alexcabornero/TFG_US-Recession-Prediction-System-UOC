"""
Modelo Random Forest

Replica la plantilla del baseline (mismo walk-forward, mismo threshold tuning,
misma ablación de rebalanceo) añadiendo optimización anidada de hiperparámetros
mediante `RandomizedSearchCV` con `TimeSeriesSplit` interno sobre el train de
cada fold externo. El test externo nunca interviene en la búsqueda.

Incluye:
- Pipeline de imblearn con StandardScaler + (SMOTE opcional) + RandomForestClassifier
- Walk-Forward CV de 4 folds reutilizando las fechas de `walk_forward_config`
- Tuning anidado: `RandomizedSearchCV(n_iter=20, cv=TimeSeriesSplit(3), scoring='average_precision')`
- Umbral óptimo mediano sobre curva PR del train de cada fold
- Ablación de las tres condiciones de rebalanceo ('none', 'balanced', 'smote')
"""

import json
import os
import warnings

os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')

from typing import Dict, List

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from baseline import cargar_dataset
from walk_forward_config import (
    CONDICIONES_REBALANCEO,
    FECHAS_CORTE,
    GAP_MESES,
    HOLDOUT_START,
)


PARAM_DISTRIBUTIONS: Dict[str, List] = {
    'classifier__n_estimators': [100, 200, 300, 500],
    'classifier__max_depth': [None, 5, 10, 15, 20],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
}


class RandomForestModel:
    """
    Random Forest con walk-forward CV externo y RandomizedSearchCV interno.
    """

    def __init__(
        self,
        random_state: int = 42,
        rebalanceo: str = 'balanced',
        n_iter: int = 20,
        n_splits_inner: int = 3,
    ):
        """
        Args:
            random_state: Semilla para reproducibilidad.
            rebalanceo: 'none' | 'balanced' | 'smote'.
            n_iter: Nº de combinaciones aleatorias en RandomizedSearchCV.
            n_splits_inner: Nº de splits del TimeSeriesSplit interno.
        """
        if rebalanceo not in set(CONDICIONES_REBALANCEO):
            raise ValueError(
                f"rebalanceo debe ser uno de {CONDICIONES_REBALANCEO}; "
                f"recibido {rebalanceo!r}"
            )
        self.random_state = random_state
        self.rebalanceo = rebalanceo
        self.n_iter = n_iter
        self.n_splits_inner = n_splits_inner
        self.modelo = None
        self.metricas: Dict = {}
        self.umbral_optimo = 0.5
        self.mejor_fold = None
        self.metricas_por_fold: List[Dict] = []

    def construir_pipeline(self) -> Pipeline:
        """
        Construye el pipeline según `self.rebalanceo`.

        Pipeline según rebalanceo. StandardScaler siempre. SMOTE lo necesita; coste despreciable para RF.
        """
        class_weight = 'balanced' if self.rebalanceo == 'balanced' else None
        clasificador = RandomForestClassifier(
            random_state=self.random_state,
            class_weight=class_weight,
        )
        pasos = [('scaler', StandardScaler())]
        if self.rebalanceo == 'smote':
            pasos.append(('smote', SMOTE(random_state=self.random_state)))
        pasos.append(('classifier', clasificador))
        return Pipeline(pasos)

    def walk_forward_cv(self, X: pd.DataFrame, y: pd.Series, n_folds: int = 4) -> Dict:
        """
        Walk-Forward CV externo + RandomizedSearchCV interno sobre el train
        de cada fold. El test externo solo se usa para reportar métricas.
        """
        fechas_corte = FECHAS_CORTE[:n_folds]
        metricas_por_fold: List[Dict] = []

        for i, config in enumerate(fechas_corte):
            mascara_train = X.index <= config['train_end']
            mascara_test = (X.index >= config['test_start']) & (X.index <= config['test_end'])
            X_train, y_train = X[mascara_train], y[mascara_train]
            X_test, y_test = X[mascara_test], y[mascara_test]
            if len(X_train) == 0 or len(X_test) == 0:
                continue

            pipeline = self.construir_pipeline()
            tscv_interno = TimeSeriesSplit(n_splits=self.n_splits_inner)
            busqueda = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=PARAM_DISTRIBUTIONS,
                n_iter=self.n_iter,
                scoring='average_precision',
                cv=tscv_interno,
                random_state=self.random_state,
                n_jobs=-1,
                refit=True,
            )
            busqueda.fit(X_train, y_train)
            mejor_pipeline = busqueda.best_estimator_

            # Umbral óptimo sobre predicciones out-of-fold del train, no sobre
            # predict_proba(X_train)
            # Umbral OOF: RF satura probabilidades in-sample → umbral no transferible al test.
            # OOF manual con TimeSeriesSplit (cross_val_predict no cubre todas las muestras).
            y_proba_oof = np.full(len(y_train), np.nan)
            for idx_entrenamiento, idx_validacion in TimeSeriesSplit(
                n_splits=self.n_splits_inner
            ).split(X_train):
                pipeline_fold = clone(mejor_pipeline)
                pipeline_fold.fit(
                    X_train.iloc[idx_entrenamiento], y_train.iloc[idx_entrenamiento]
                )
                y_proba_oof[idx_validacion] = pipeline_fold.predict_proba(
                    X_train.iloc[idx_validacion]
                )[:, 1]
            mascara_oof = ~np.isnan(y_proba_oof)
            precisiones_tr, recalls_tr, umbrales_tr = precision_recall_curve(
                y_train.values[mascara_oof], y_proba_oof[mascara_oof]
            )
            f1_tr = np.where(
                (precisiones_tr[:-1] + recalls_tr[:-1]) > 0,
                2 * precisiones_tr[:-1] * recalls_tr[:-1]
                / (precisiones_tr[:-1] + recalls_tr[:-1]),
                0.0,
            )
            idx_mejor = int(np.argmax(f1_tr))
            umbral_optimo_fold = float(umbrales_tr[idx_mejor])

            y_pred_proba = mejor_pipeline.predict_proba(X_test)[:, 1]
            auc_roc = (
                roc_auc_score(y_test, y_pred_proba)
                if len(np.unique(y_test)) > 1 else 0.5
            )
            pr_auc = (
                average_precision_score(y_test, y_pred_proba)
                if len(np.unique(y_test)) > 1 else 0.0
            )

            metricas_por_fold.append({
                'fold': i + 1,
                'nombre': config['nombre'],
                'train_start': str(X_train.index[0].date()),
                'train_end': str(X_train.index[-1].date()),
                'test_start': str(X_test.index[0].date()),
                'test_end': str(X_test.index[-1].date()),
                'train_size': len(X_train),
                'test_size': len(X_test),
                'umbral_optimo': umbral_optimo_fold,
                'auc_roc': auc_roc,
                'pr_auc': pr_auc,
                'n_recesiones_test': int(y_test.sum()),
                'best_params': {k: v for k, v in busqueda.best_params_.items()},
                'cv_best_score': float(busqueda.best_score_),
                'y_test': y_test.values,
                'y_pred_proba': y_pred_proba,
                'precisiones_tr': precisiones_tr,
                'recalls_tr': recalls_tr,
                'umbrales_tr': umbrales_tr,
                'pipeline': mejor_pipeline,
            })

        self.metricas_por_fold = metricas_por_fold
        if not metricas_por_fold:
            self.metricas = {}
            return self.metricas

        self.umbral_optimo = float(np.median([f['umbral_optimo'] for f in metricas_por_fold]))

        todas_predicciones, todos_valores_reales, todas_probabilidades = [], [], []
        for f in metricas_por_fold:
            y_pred = (f['y_pred_proba'] >= self.umbral_optimo).astype(int)
            f['recall'] = float(recall_score(f['y_test'], y_pred, zero_division=0))
            f['f1_score'] = float(f1_score(f['y_test'], y_pred, zero_division=0))
            f['n_recesiones_pred'] = int(y_pred.sum())
            todas_predicciones.extend(y_pred)
            todos_valores_reales.extend(f['y_test'])
            todas_probabilidades.extend(f['y_pred_proba'])

        self.mejor_fold = max(metricas_por_fold, key=lambda x: x['f1_score'])

        campos_descartados = {
            'y_test', 'y_pred_proba', 'precisiones_tr', 'recalls_tr',
            'umbrales_tr', 'pipeline',
        }
        self.metricas = {
            'modelo': 'RandomForestClassifier',
            'rebalanceo': self.rebalanceo,
            'n_folds': len(metricas_por_fold),
            'umbral_optimo': self.umbral_optimo,
            'n_iter_random_search': self.n_iter,
            'n_splits_inner': self.n_splits_inner,
            'folds': [
                {k: v for k, v in f.items() if k not in campos_descartados}
                for f in metricas_por_fold
            ],
            'aggregated': {
                'recall_mean': float(np.mean([f['recall'] for f in metricas_por_fold])),
                'recall_std': float(np.std([f['recall'] for f in metricas_por_fold])),
                'f1_mean': float(np.mean([f['f1_score'] for f in metricas_por_fold])),
                'f1_std': float(np.std([f['f1_score'] for f in metricas_por_fold])),
                'auc_roc_mean': float(np.mean([f['auc_roc'] for f in metricas_por_fold])),
                'auc_roc_std': float(np.std([f['auc_roc'] for f in metricas_por_fold])),
                'pr_auc_mean': float(np.mean([f['pr_auc'] for f in metricas_por_fold])),
                'pr_auc_std': float(np.std([f['pr_auc'] for f in metricas_por_fold])),
            },
            'overall': {
                'recall': float(recall_score(todos_valores_reales, todas_predicciones, zero_division=0)),
                'f1_score': float(f1_score(todos_valores_reales, todas_predicciones, zero_division=0)),
                'auc_roc': float(roc_auc_score(todos_valores_reales, todas_probabilidades)),
                'pr_auc': float(average_precision_score(todos_valores_reales, todas_probabilidades)),
                'confusion_matrix': confusion_matrix(todos_valores_reales, todas_predicciones).tolist(),
            },
        }
        return self.metricas

    def visualizar_pr_train_por_fold(self, guardar_ruta: str = None) -> None:
        """Curva PR del train por fold con el umbral seleccionado marcado."""
        if not self.metricas_por_fold:
            return
        folds = self.metricas_por_fold
        fig, axes = plt.subplots(1, len(folds), figsize=(5 * len(folds), 5), squeeze=False)
        for ax, fold in zip(axes[0], folds):
            p, r, u = fold['precisiones_tr'], fold['recalls_tr'], fold['umbrales_tr']
            umbral = fold['umbral_optimo']
            idx = int(np.argmin(np.abs(u - umbral)))
            pu, ru = float(p[idx]), float(r[idx])
            f1u = 2 * pu * ru / (pu + ru) if (pu + ru) > 0 else 0.0
            ax.plot(r, p, color='steelblue', linewidth=2, label='Curva PR (train)')
            ax.scatter([ru], [pu], color='red', s=80, zorder=5,
                       label=f'Umbral={umbral:.3f}\nP={pu:.2f}  R={ru:.2f}\nF1={f1u:.2f}')
            ax.set_xlabel('Recall', fontsize=11)
            ax.set_ylabel('Precision', fontsize=11)
            ax.set_title(f"{fold['nombre']} — Train PR", fontsize=12, fontweight='bold')
            ax.set_xlim(0, 1.02)
            ax.set_ylim(0, 1.02)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower left', fontsize=9)
        plt.tight_layout()
        if guardar_ruta:
            os.makedirs(os.path.dirname(guardar_ruta), exist_ok=True)
            plt.savefig(guardar_ruta, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def visualizar_curvas_roc_pr(self, guardar_ruta: str = None) -> None:
        """Curvas ROC y PR del mejor fold (por F1) sobre el test externo."""
        if self.mejor_fold is None:
            return
        y_test = self.mejor_fold['y_test']
        y_proba = self.mejor_fold['y_pred_proba']
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc_val = auc(fpr, tpr)
        precision, recall_curve_, _ = precision_recall_curve(y_test, y_proba)
        pr_auc_val = auc(recall_curve_, precision)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax1.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc_val:.3f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Baseline')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate', fontsize=11)
        ax1.set_ylabel('True Positive Rate (Recall)', fontsize=11)
        ax1.set_title(f'Curva ROC\n{self.mejor_fold["nombre"]}', fontsize=12, fontweight='bold')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)

        ax2.plot(recall_curve_, precision, color='darkgreen', lw=2,
                 label=f'PR curve (AUC = {pr_auc_val:.3f})')
        ax2.axhline(y_test.mean(), color='navy', lw=1, linestyle='--',
                    label=f'Baseline ({y_test.mean():.3f})')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall', fontsize=11)
        ax2.set_ylabel('Precision', fontsize=11)
        ax2.set_title(f'Curva Precision-Recall\n{self.mejor_fold["nombre"]}',
                      fontsize=12, fontweight='bold')
        ax2.legend(loc='lower left')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        if guardar_ruta:
            os.makedirs(os.path.dirname(guardar_ruta), exist_ok=True)
            plt.savefig(guardar_ruta, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def guardar_metricas(self, ruta: str) -> str:
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        with open(ruta, 'w') as f:
            json.dump(self.metricas, f, indent=2)
        return ruta

    def entrenar_final(self, X: pd.DataFrame, y: pd.Series, best_params: Dict) -> Pipeline:
        """
        Entrena el pipeline final sobre todo el tramo pre-hold-out con los
        mejores hiperparámetros del fold ganador.
        """
        pipeline = self.construir_pipeline()
        pipeline.set_params(**best_params)
        pipeline.fit(X, y)
        self.modelo = pipeline
        return pipeline

    def guardar_modelo(self, ruta: str) -> str:
        if self.modelo is None:
            raise RuntimeError("No hay modelo entrenado. Ejecuta entrenar_final primero.")
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        joblib.dump(self.modelo, ruta)
        return ruta


def _ejecutar_target(
    etiqueta: str,
    X: pd.DataFrame,
    y: pd.Series,
    ruta_metricas: str,
    ruta_roc_pr: str,
    ruta_pr_train: str,
) -> RandomForestModel:
    """Ejecuta el CV, guarda artefactos y muestra el resumen compacto."""
    modelo = RandomForestModel(random_state=42, rebalanceo='balanced')
    metricas = modelo.walk_forward_cv(X, y, n_folds=4)
    modelo.guardar_metricas(ruta_metricas)
    modelo.visualizar_curvas_roc_pr(guardar_ruta=ruta_roc_pr)
    modelo.visualizar_pr_train_por_fold(guardar_ruta=ruta_pr_train)

    print(f"\n{etiqueta}")
    for fold in metricas['folds']:
        print(
            f"{fold['nombre']}  "
            f"Train: {fold['train_start'][:7]} \u2192 {fold['train_end'][:7]}  |  "
            f"Test: {fold['test_start'][:7]} \u2192 {fold['test_end'][:7]}  |  "
            f"AUC-ROC: {fold['auc_roc']:.3f}  PR-AUC: {fold['pr_auc']:.3f}"
        )
    ag = metricas['aggregated']
    print(
        f"Umbral mediano: {metricas['umbral_optimo']:.3f}  |  "
        f"Recall: {ag['recall_mean']:.3f}  F1: {ag['f1_mean']:.3f}  "
        f"AUC-ROC: {ag['auc_roc_mean']:.3f}  PR-AUC: {ag['pr_auc_mean']:.3f}"
    )
    return modelo


def _ablacion_target(etiqueta: str, X: pd.DataFrame, y: pd.Series) -> Dict:
    """Ejecuta las 3 condiciones de rebalanceo sobre los mismos folds."""
    resultados: Dict[str, Dict[str, float]] = {}
    for cond in CONDICIONES_REBALANCEO:
        modelo = RandomForestModel(random_state=42, rebalanceo=cond)
        m = modelo.walk_forward_cv(X, y, n_folds=4)
        ag = m['aggregated']
        resultados[cond] = {
            'pr_auc': ag['pr_auc_mean'],
            'auc_roc': ag['auc_roc_mean'],
            'recall': ag['recall_mean'],
            'f1_score': ag['f1_mean'],
            'umbral_mediano': m['umbral_optimo'],
        }

    print(f"\nABLACIÓN — REBALANCEO ({etiqueta})")
    print(f"{'Condición':<12} {'PR-AUC':>7}  {'AUC-ROC':>7}  {'Recall':>7}  {'F1':>7}")
    for cond in CONDICIONES_REBALANCEO:
        r = resultados[cond]
        print(
            f"{cond:<12} "
            f"{r['pr_auc']:>7.3f}  "
            f"{r['auc_roc']:>7.3f}  "
            f"{r['recall']:>7.3f}  "
            f"{r['f1_score']:>7.3f}"
        )
    return resultados


class BalancedRandomForestModel(RandomForestModel):
    """
    Variante que usa `BalancedRandomForestClassifier` de imblearn, que integra
    el submuestreo de la clase mayoritaria dentro del bootstrap de cada árbol.
    No requiere ablación de rebalanceo: el balanceo está embebido en el
    clasificador. Mantiene el resto de la pipeline y protocolo del RF estándar.
    """

    def __init__(
        self,
        random_state: int = 42,
        n_iter: int = 20,
        n_splits_inner: int = 3,
    ):
        super().__init__(
            random_state=random_state,
            rebalanceo='none',
            n_iter=n_iter,
            n_splits_inner=n_splits_inner,
        )

    def construir_pipeline(self) -> Pipeline:
        """StandardScaler -> BalancedRandomForestClassifier (sin SMOTE, sin class_weight)."""
        clasificador = BalancedRandomForestClassifier(
            random_state=self.random_state,
            sampling_strategy='all',
            replacement=True,
            bootstrap=False,
        )
        return Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', clasificador),
        ])


def _ejecutar_balanced_rf(X: pd.DataFrame, y_12m: pd.Series) -> None:
    """
    Entrena BalancedRandomForestClassifier sobre target_12m con el mismo
    protocolo (walk-forward, tuning anidado, umbral OOF) que el RF estándar.
    Una sola condición. Si supera PR-AUC 0.615 del baseline, serializa.
    """
    modelo = BalancedRandomForestModel(random_state=42)
    metricas = modelo.walk_forward_cv(X, y_12m, n_folds=4)
    modelo.guardar_metricas("models/random_forest_balanced_metrics_12m.json")
    modelo.visualizar_curvas_roc_pr(
        guardar_ruta="docs/figures/random_forest_balanced_12m_roc_pr.png"
    )
    modelo.visualizar_pr_train_por_fold(
        guardar_ruta="docs/figures/random_forest_balanced_12m_pr_train_por_fold.png"
    )

    print("\n" + "=" * 64)
    print("BALANCED RANDOM FOREST (TARGET 12M)")
    print("=" * 64)
    for fold in metricas['folds']:
        print(
            f"{fold['nombre']}  "
            f"Train: {fold['train_start'][:7]} \u2192 {fold['train_end'][:7]}  |  "
            f"Test: {fold['test_start'][:7]} \u2192 {fold['test_end'][:7]}  |  "
            f"AUC-ROC: {fold['auc_roc']:.3f}  PR-AUC: {fold['pr_auc']:.3f}"
        )
    ag = metricas['aggregated']
    print(
        f"Umbral mediano: {metricas['umbral_optimo']:.3f}  |  "
        f"Recall: {ag['recall_mean']:.3f}  F1: {ag['f1_mean']:.3f}  "
        f"AUC-ROC: {ag['auc_roc_mean']:.3f}  PR-AUC: {ag['pr_auc_mean']:.3f}"
    )

    with open("models/random_forest_metrics_12m.json", 'r') as f:
        rf_std = json.load(f)
    ag_std = rf_std['aggregated']
    print("\nComparativa target_12m (PR-AUC | AUC-ROC | Recall | F1):")
    print(
        f"  Baseline LR    {0.615:.3f} | {0.783:.3f} | {0.667:.3f} | {0.273:.3f}"
    )
    print(
        f"  RF balanced    {ag_std['pr_auc_mean']:.3f} | "
        f"{ag_std['auc_roc_mean']:.3f} | {ag_std['recall_mean']:.3f} | "
        f"{ag_std['f1_mean']:.3f}"
    )
    print(
        f"  BalancedRF     {ag['pr_auc_mean']:.3f} | "
        f"{ag['auc_roc_mean']:.3f} | {ag['recall_mean']:.3f} | "
        f"{ag['f1_mean']:.3f}"
    )

    if ag['pr_auc_mean'] > 0.615:
        mejores_params = modelo.mejor_fold['best_params']
        mascara = X.index < HOLDOUT_START
        modelo.entrenar_final(X[mascara], y_12m[mascara], best_params=mejores_params)
        modelo.guardar_modelo("models/random_forest_balanced.pkl")
        print(
            f"\n[OK] PR-AUC {ag['pr_auc_mean']:.3f} > 0.615 (baseline). "
            f"Modelo serializado en models/random_forest_balanced.pkl "
            f"con hiperparámetros {mejores_params}"
        )
    else:
        print(
            f"\n[X] PR-AUC {ag['pr_auc_mean']:.3f} <= 0.615 (baseline). "
            f"No se serializa."
        )


if __name__ == "__main__":
    X, y_6m, y_12m = cargar_dataset()
    os.makedirs("docs/figures", exist_ok=True)

    print("=" * 64)
    print("RANDOM FOREST")
    print("=" * 64)

    _ejecutar_target(
        etiqueta="TARGET 6M",
        X=X,
        y=y_6m,
        ruta_metricas="models/random_forest_metrics_6m.json",
        ruta_roc_pr="docs/figures/random_forest_6m_roc_pr.png",
        ruta_pr_train="docs/figures/random_forest_6m_pr_train_por_fold.png",
    )

    _ejecutar_target(
        etiqueta="TARGET 12M",
        X=X,
        y=y_12m,
        ruta_metricas="models/random_forest_metrics_12m.json",
        ruta_roc_pr="docs/figures/random_forest_12m_roc_pr.png",
        ruta_pr_train="docs/figures/random_forest_12m_pr_train_por_fold.png",
    )

    ablacion = {
        'target_6m': _ablacion_target('target_6m', X, y_6m),
        'target_12m': _ablacion_target('target_12m', X, y_12m),
    }
    os.makedirs("models", exist_ok=True)
    with open("models/random_forest_ablacion.json", 'w') as f:
        json.dump(ablacion, f, indent=2, default=str)

    _ejecutar_balanced_rf(X, y_12m)
