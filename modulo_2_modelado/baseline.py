"""
Módulo de modelo baseline: Regresión Logística

Implementa el modelo más sencillo como punto de referencia (baseline)
para comparar con modelos más complejos (Random Forest, XGBoost, LightGBM).

Incluye:
- Pipeline de imblearn con preprocesamiento + LogisticRegression
- Walk-Forward Cross-Validation para series temporales
- Métricas: Recall, F1-Score, AUC-ROC, matriz de confusión
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    recall_score, f1_score, roc_auc_score, 
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, auc, average_precision_score
)
from sklearn.model_selection import TimeSeriesSplit
import json
import os
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from walk_forward_config import (
    FECHAS_CORTE, HOLDOUT_START, HOLDOUT_END,
    HOLDOUT_EXPANSION_END, HOLDOUT_COVID_START,
    HOLDOUT_COVID_END, GAP_MESES
)


class BaselineModel:
    """
    Modelo baseline de Regresión Logística para predicción de recesiones.
    
    Este modelo sirve como punto de referencia mínimo viable. Cualquier
    modelo más complejo debe superar significativamente este baseline.
    """
    
    def __init__(self, random_state: int = 42, rebalanceo: str = 'balanced'):
        """
        Inicializa el modelo baseline.

        Args:
            random_state: Semilla para reproducibilidad
            rebalanceo: Estrategia de rebalanceo de clases. Valores admitidos:
                - 'none': sin rebalanceo
                - 'balanced': class_weight='balanced' en la LogisticRegression
                - 'smote': inyectar SMOTE tras el StandardScaler en el pipeline
        """
        if rebalanceo not in {'none', 'balanced', 'smote'}:
            raise ValueError(
                f"rebalanceo debe ser 'none', 'balanced' o 'smote'; recibido {rebalanceo!r}"
            )
        self.random_state = random_state
        self.rebalanceo = rebalanceo
        self.modelo = None
        self.metricas = {}
        self.umbral_optimo = 0.5
        self.mejor_fold = None

    def construir_pipeline(self) -> Pipeline:
        """
        Construye el pipeline de scikit-learn / imblearn según `self.rebalanceo`.

        Estrategias:
        - 'none': StandardScaler -> LogisticRegression (sin rebalanceo)
        - 'balanced': StandardScaler -> LogisticRegression(class_weight='balanced')
        - 'smote': StandardScaler -> SMOTE -> LogisticRegression

        SMOTE solo actúa dentro del pipeline durante el `fit`; imblearn se asegura
        de que no se aplique a predict/predict_proba, evitando leakage hacia el
        test y el hold-out.

        Returns:
            Pipeline configurado
        """
        class_weight = 'balanced' if self.rebalanceo == 'balanced' else None
        clasificador = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            class_weight=class_weight,
            C=1.0,
            solver='lbfgs',
        )

        pasos = [('scaler', StandardScaler())]
        if self.rebalanceo == 'smote':
            pasos.append(('smote', SMOTE(random_state=self.random_state)))
        pasos.append(('classifier', clasificador))

        return Pipeline(pasos)
    
    def walk_forward_cv(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        n_folds: int = 4,
        gap_meses: int = GAP_MESES
    ) -> Dict:
        """
        Implementa Walk-Forward Cross-Validation para series temporales.

        Estrategia de validación (4 folds de 80 meses de test con gap de 12 meses):
        - Fold 1: Train 1967-01 → 1980-08, Test 1981-09 → 1988-04
        - Fold 2: Train 1967-01 → 1988-04, Test 1989-05 → 1995-12
        - Fold 3: Train 1967-01 → 1995-12, Test 1997-01 → 2003-08
        - Fold 4: Train 1967-01 → 2003-08, Test 2004-09 → 2011-04
        - Hold-out (intocable): 2011-05 → 2025-01

        Args:
            X: Features (DataFrame)
            y: Target (Series)
            n_folds: Número de splits para validación (default: 4)
            gap_meses: Gap en meses entre train y test para evitar leakage temporal

        Returns:
            Diccionario con métricas agregadas y por fold
        """
        # Fechas de corte cargadas desde walk_forward_config (fuente de verdad única).
        # El hold-out se evalúa por separado en `evaluar_hold_out`, solo tras la
        # selección del modelo ganador final.
        fechas_corte = FECHAS_CORTE[:n_folds]
        
        metricas_por_fold = []
        
        for i, config in enumerate(fechas_corte):
            # Usar train_end directamente (el gap ya está implícito en las fechas manuales)
            mascara_train = X.index <= config['train_end']
            mascara_test = (X.index >= config['test_start']) & (X.index <= config['test_end'])
            
            X_train = X[mascara_train]
            y_train = y[mascara_train]
            X_test = X[mascara_test]
            y_test = y[mascara_test]
            
            if len(X_train) == 0 or len(X_test) == 0:
                continue

            pipeline = self.construir_pipeline()
            pipeline.fit(X_train, y_train)

            # Umbral óptimo del fold: máximo F1 sobre la curva PR del TRAIN
            y_proba_train = pipeline.predict_proba(X_train)[:, 1]
            precisiones_tr, recalls_tr, umbrales_tr = precision_recall_curve(y_train, y_proba_train)
            f1_tr = np.where(
                (precisiones_tr[:-1] + recalls_tr[:-1]) > 0,
                2 * precisiones_tr[:-1] * recalls_tr[:-1] / (precisiones_tr[:-1] + recalls_tr[:-1]),
                0.0
            )
            idx_mejor = int(np.argmax(f1_tr))
            umbral_optimo_fold = float(umbrales_tr[idx_mejor])

            # Probabilidades sobre el test (métricas con umbral se calculan tras la mediana)
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
            auc_roc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5
            pr_auc = average_precision_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0

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
                'y_test': y_test.values,
                'y_pred_proba': y_pred_proba,
                'precisiones_tr': precisiones_tr,
                'recalls_tr': recalls_tr,
                'umbrales_tr': umbrales_tr,
            })

        self.metricas_por_fold = metricas_por_fold

        if not metricas_por_fold:
            self.metricas = {}
            return self.metricas

        # Umbral mediano entre folds: estabiliza la decisión final
        self.umbral_optimo = float(np.median([f['umbral_optimo'] for f in metricas_por_fold]))

        # Recalcular métricas dependientes del umbral con el umbral mediano
        todas_predicciones = []
        todos_valores_reales = []
        todas_probabilidades = []
        for f in metricas_por_fold:
            y_pred = (f['y_pred_proba'] >= self.umbral_optimo).astype(int)
            f['recall'] = float(recall_score(f['y_test'], y_pred, zero_division=0))
            f['f1_score'] = float(f1_score(f['y_test'], y_pred, zero_division=0))
            f['n_recesiones_pred'] = int(y_pred.sum())
            todas_predicciones.extend(y_pred)
            todos_valores_reales.extend(f['y_test'])
            todas_probabilidades.extend(f['y_pred_proba'])

        self.mejor_fold = max(metricas_por_fold, key=lambda x: x['f1_score'])

        self.metricas = {
            'modelo': 'LogisticRegression',
            'n_folds': len(metricas_por_fold),
            'umbral_optimo': self.umbral_optimo,
            'folds': [
                {k: v for k, v in f.items()
                 if k not in ['y_test', 'y_pred_proba', 'precisiones_tr', 'recalls_tr', 'umbrales_tr']}
                for f in metricas_por_fold
            ],
            'aggregated': {
                'recall_mean': np.mean([f['recall'] for f in metricas_por_fold]),
                'recall_std': np.std([f['recall'] for f in metricas_por_fold]),
                'f1_mean': np.mean([f['f1_score'] for f in metricas_por_fold]),
                'f1_std': np.std([f['f1_score'] for f in metricas_por_fold]),
                'auc_roc_mean': np.mean([f['auc_roc'] for f in metricas_por_fold]),
                'auc_roc_std': np.std([f['auc_roc'] for f in metricas_por_fold]),
                'pr_auc_mean': np.mean([f['pr_auc'] for f in metricas_por_fold]),
                'pr_auc_std': np.std([f['pr_auc'] for f in metricas_por_fold]),
            },
            'overall': {
                'recall': recall_score(todos_valores_reales, todas_predicciones, zero_division=0),
                'f1_score': f1_score(todos_valores_reales, todas_predicciones, zero_division=0),
                'auc_roc': roc_auc_score(todos_valores_reales, todas_probabilidades),
                'pr_auc': average_precision_score(todos_valores_reales, todas_probabilidades),
                'confusion_matrix': confusion_matrix(todos_valores_reales, todas_predicciones).tolist(),
            },
        }

        return self.metricas
    
    def visualizar_pr_train_por_fold(self, guardar_ruta: str = None) -> None:
        """
        Visualiza la curva Precision-Recall del TRAIN para cada fold,
        marcando el umbral seleccionado (máximo F1).

        Args:
            guardar_ruta: Ruta opcional para guardar el gráfico
        """
        if not getattr(self, 'metricas_por_fold', None):
            print("No hay datos de folds. Ejecuta walk_forward_cv primero.")
            return

        folds = self.metricas_por_fold
        n_folds = len(folds)
        fig, axes = plt.subplots(1, n_folds, figsize=(5 * n_folds, 5), squeeze=False)

        for ax, fold in zip(axes[0], folds):
            precisiones = fold['precisiones_tr']
            recalls = fold['recalls_tr']
            umbrales = fold['umbrales_tr']
            umbral = fold['umbral_optimo']
            idx = int(np.argmin(np.abs(umbrales - umbral)))
            p_umbral = float(precisiones[idx])
            r_umbral = float(recalls[idx])
            f1_umbral = (
                2 * p_umbral * r_umbral / (p_umbral + r_umbral)
                if (p_umbral + r_umbral) > 0 else 0.0
            )

            ax.plot(recalls, precisiones, color='steelblue', linewidth=2,
                    label='Curva PR (train)')
            ax.scatter([r_umbral], [p_umbral], color='red', s=80, zorder=5,
                       label=(f'Umbral={umbral:.3f}\n'
                              f'P={p_umbral:.2f}  R={r_umbral:.2f}\n'
                              f'F1={f1_umbral:.2f}'))

            ax.set_xlabel('Recall', fontsize=11)
            ax.set_ylabel('Precision', fontsize=11)
            ax.set_title(f"{fold['nombre']} — Train PR",
                         fontsize=12, fontweight='bold')
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
        """
        Visualiza curvas ROC y Precision-Recall del mejor fold.
        
        Args:
            guardar_ruta: Ruta opcional para guardar el gráfico
        """
        if self.mejor_fold is None:
            print("No hay datos de mejor fold. Ejecuta walk_forward_cv primero.")
            return
        
        y_test = self.mejor_fold['y_test']
        y_proba = self.mejor_fold['y_pred_proba']
        
        # Calcular curvas
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        precision, recall_curve, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall_curve, precision)
        
        # Crear subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Curva ROC
        ax1.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Baseline')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate', fontsize=11)
        ax1.set_ylabel('True Positive Rate (Recall)', fontsize=11)
        ax1.set_title(f'Curva ROC\n{self.mejor_fold["nombre"]}', fontsize=12, fontweight='bold')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        # Curva Precision-Recall
        ax2.plot(recall_curve, precision, color='darkgreen', lw=2,
                 label=f'PR curve (AUC = {pr_auc:.3f})')
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
            plt.savefig(guardar_ruta, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def obtener_coeficientes(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Obtiene los coeficientes del modelo ordenados por importancia absoluta.
        
        Args:
            X: DataFrame con las features (para obtener nombres)
            
        Returns:
            DataFrame con coeficientes ordenados
        """
        if self.modelo is None:
            print("El modelo no ha sido entrenado. Ejecuta entrenar_final primero.")
            return pd.DataFrame()
        
        # Extraer coeficientes del modelo
        coeficientes = self.modelo.named_steps['classifier'].coef_[0]
        nombres = X.columns.tolist()
        
        # Crear DataFrame
        df_coef = pd.DataFrame({
            'feature': nombres,
            'coeficiente': coeficientes,
            'abs_coeficiente': np.abs(coeficientes)
        }).sort_values('abs_coeficiente', ascending=False)
        
        return df_coef
    
    def _calcular_metricas_ventana(
        self,
        X_ventana: pd.DataFrame,
        y_ventana: pd.Series,
    ) -> Dict:
        """
        Calcula métricas del modelo ya entrenado sobre una ventana temporal.

        Args:
            X_ventana: Features de la ventana
            y_ventana: Target real de la ventana

        Returns:
            Diccionario con métricas (recall, f1, auc_roc, pr_auc, matriz de
            confusión, false alarm rate, umbral usado, fechas y conteos).
        """
        y_pred_proba = self.modelo.predict_proba(X_ventana)[:, 1]
        y_pred = (y_pred_proba >= self.umbral_optimo).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_ventana, y_pred, labels=[0, 1]
        ).ravel()
        negativos = tn + fp
        false_alarm_rate = float(fp / negativos) if negativos > 0 else 0.0

        return {
            'fecha_inicio': str(X_ventana.index[0].date()),
            'fecha_fin': str(X_ventana.index[-1].date()),
            'n_observaciones': len(X_ventana),
            'n_recesiones_real': int(y_ventana.sum()),
            'n_recesiones_pred': int(y_pred.sum()),
            'recall': float(recall_score(y_ventana, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_ventana, y_pred, zero_division=0)),
            'auc_roc': float(roc_auc_score(y_ventana, y_pred_proba))
                       if len(np.unique(y_ventana)) > 1 else 0.5,
            'pr_auc': float(average_precision_score(y_ventana, y_pred_proba)),
            'false_alarm_rate': false_alarm_rate,
            'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]],
            'umbral_usado': float(self.umbral_optimo),
        }

    def evaluar_hold_out(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        fecha_inicio: pd.Timestamp = HOLDOUT_START,
        fecha_inicio_covid: pd.Timestamp = HOLDOUT_COVID_START,
        fecha_fin_covid: pd.Timestamp = HOLDOUT_COVID_END,
    ) -> Dict:
        """
        Evalúa el modelo sobre el hold-out intocable, segmentado en sub-ventanas.

        Estructura del hold-out:
        - Expansión pura: may 2011 — feb 2020. Mide especificidad vía
          `false_alarm_rate`. Principal indicador de fiabilidad fuera de recesión.
        - COVID: mar-abr 2020. Reportado como caso de estudio separado porque
          es un shock exógeno (pandemia) no anticipable por indicadores macro
          clásicos (yield spread, credit spread) a 12 meses.
        - Global: may 2011 — fin del hold-out. Se mantiene solo como referencia;
          no debe promediarse sin segmentar dado el fuerte desbalance (~98% neg.).

        Solo debe ejecutarse una vez, tras la selección del modelo ganador.

        Args:
            X: Features completas del dataset.
            y: Target completo del dataset.
            fecha_inicio: Inicio del hold-out (incluido).
            fecha_inicio_covid: Inicio de la sub-ventana COVID (incluido).
            fecha_fin_covid: Fin de la sub-ventana COVID (incluido).

        Returns:
            Diccionario con claves 'expansion', 'covid' y 'global', cada una con
            sus métricas; además de la nota explicativa sobre el COVID.
        """
        print(f"\n{'='*70}")
        print("EVALUACIÓN HOLD-OUT (segmentado)")
        print(f"{'='*70}")
        print("⚠️  Este conjunto solo debe usarse para la evaluación final del "
              "modelo ganador.")

        if self.modelo is None:
            print("⚠️  No hay modelo entrenado. Ejecuta entrenar_final primero.")
            return {}

        inicio = pd.Timestamp(fecha_inicio)
        inicio_covid = pd.Timestamp(fecha_inicio_covid)
        fin_covid = pd.Timestamp(fecha_fin_covid)
        # Permite pasar strings o Timestamps indistintamente.

        mascara_hold = X.index >= inicio
        X_hold, y_hold = X[mascara_hold], y[mascara_hold]
        if len(X_hold) == 0:
            print("No hay datos en el hold-out.")
            return {}

        mascara_expansion = (X.index >= inicio) & (X.index < inicio_covid)
        mascara_covid = (X.index >= inicio_covid) & (X.index <= fin_covid)

        nota_covid = (
            "La recesión COVID (mar-abr 2020) es un shock exógeno (pandemia) "
            "no anticipable por indicadores macroeconómicos clásicos (yield "
            "spread, credit spread) a 12 meses de antelación. Se reporta como "
            "caso de estudio separado y no debe promediarse con la sub-ventana "
            "expansiva."
        )

        resultado: Dict = {'umbral_usado': float(self.umbral_optimo)}

        if mascara_expansion.sum() > 0:
            resultado['expansion'] = self._calcular_metricas_ventana(
                X[mascara_expansion], y[mascara_expansion]
            )
            exp = resultado['expansion']
            print(f"\nSub-ventana EXPANSIÓN ({exp['fecha_inicio']} a "
                  f"{exp['fecha_fin']}, {exp['n_observaciones']} obs)")
            print(f"  False alarm rate: {exp['false_alarm_rate']:.3f} "
                  f"(FP={exp['confusion_matrix'][0][1]} / "
                  f"negativos={exp['confusion_matrix'][0][0] + exp['confusion_matrix'][0][1]})")
            print(f"  Recall: {exp['recall']:.3f}, F1: {exp['f1_score']:.3f}, "
                  f"AUC-ROC: {exp['auc_roc']:.3f}, PR-AUC: {exp['pr_auc']:.3f}")
        else:
            resultado['expansion'] = {}
            print("\nSub-ventana EXPANSIÓN: sin observaciones.")

        if mascara_covid.sum() > 0:
            resultado['covid'] = self._calcular_metricas_ventana(
                X[mascara_covid], y[mascara_covid]
            )
            resultado['covid']['nota'] = nota_covid
            cov = resultado['covid']
            print(f"\nSub-ventana COVID ({cov['fecha_inicio']} a "
                  f"{cov['fecha_fin']}, {cov['n_observaciones']} obs)")
            print(f"  Recesiones reales/predichas: {cov['n_recesiones_real']}"
                  f"/{cov['n_recesiones_pred']}")
            print(f"  Recall: {cov['recall']:.3f}, F1: {cov['f1_score']:.3f}")
            print(f"  Nota: {nota_covid}")
        else:
            resultado['covid'] = {'nota': nota_covid}
            print("\nSub-ventana COVID: sin observaciones.")

        resultado['global'] = self._calcular_metricas_ventana(X_hold, y_hold)
        glb = resultado['global']
        print(f"\nHold-out GLOBAL ({glb['fecha_inicio']} a {glb['fecha_fin']}, "
              f"{glb['n_observaciones']} obs) — solo referencia")
        print(f"  Recall: {glb['recall']:.3f}, F1: {glb['f1_score']:.3f}, "
              f"AUC-ROC: {glb['auc_roc']:.3f}, PR-AUC: {glb['pr_auc']:.3f}")

        return resultado
    
    def guardar_metricas(self, ruta: str = "models/baseline_metrics.json") -> str:
        """
        Guarda las métricas del modelo en formato JSON.
        
        Args:
            ruta: Ruta donde guardar el archivo JSON
            
        Returns:
            Ruta del archivo guardado
        """
        # Asegurar que existe el directorio
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        
        with open(ruta, 'w') as f:
            json.dump(self.metricas, f, indent=2)
        
        return ruta
    
    def entrenar_final(self, X: pd.DataFrame, y: pd.Series) -> Pipeline:
        """
        Entrena el modelo final con todos los datos disponibles.
        
        Este modelo se usará para hacer predicciones en producción.
        
        Args:
            X: Features completos
            y: Target completo
            
        Returns:
            Pipeline entrenado
        """
        print(f"\nEntrenando modelo final con {len(X)} observaciones...")
        print(f"Período: {X.index[0].date()} a {X.index[-1].date()}")
        
        pipeline = self.construir_pipeline()
        pipeline.fit(X, y)
        
        self.modelo = pipeline
        print("Modelo final entrenado exitosamente.")
        
        return pipeline
    
    def guardar_modelo(self, ruta: str = "models/baseline_model.pkl") -> str:
        """
        Guarda el modelo entrenado en disco.
        
        Args:
            ruta: Ruta donde guardar el modelo
            
        Returns:
            Ruta del modelo guardado
        """
        import joblib
        
        if self.modelo is None:
            raise ValueError("El modelo no ha sido entrenado. Llama a entrenar_final() primero.")
        
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        joblib.dump(self.modelo, ruta)
        
        print(f"Modelo guardado en: {ruta}")
        return ruta


def cargar_dataset(ruta: str = "data/processed/dataset_final.csv") -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Carga el dataset final y separa features y targets.
    
    Args:
        ruta: Ruta al dataset_final.csv
        
    Returns:
        Tupla (X, y_6m, y_12m) donde:
        - X: Features (sin columnas target)
        - y_6m: Target a 6 meses
        - y_12m: Target a 12 meses
    """
    df = pd.read_csv(ruta, index_col=0, parse_dates=True)
    # NOTA: usrec se excluye de features porque es la variable fuente (leakage directo)
    X = df.drop(columns=['target_6m', 'target_12m', 'usrec'])
    y_6m = df['target_6m']
    y_12m = df['target_12m']

    return X, y_6m, y_12m


CONDICIONES_REBALANCEO: List[str] = ['none', 'balanced', 'smote']


def _ablacion_target(etiqueta: str, X: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Ejecuta las tres condiciones de rebalanceo sobre los mismos folds del
    walk-forward y devuelve las métricas agregadas de cada una.

    Args:
        etiqueta: Nombre del target mostrado en la tabla (ej. 'target_12m').
        X: Features completas.
        y: Target (serie binaria).

    Returns:
        Diccionario {condicion: {pr_auc, auc_roc, recall, f1_score, umbral_mediano}}.
    """
    resultados: Dict[str, Dict[str, float]] = {}
    for cond in CONDICIONES_REBALANCEO:
        modelo = BaselineModel(random_state=42, rebalanceo=cond)
        m = modelo.walk_forward_cv(X, y, n_folds=4, gap_meses=GAP_MESES)
        ag = m['aggregated']
        resultados[cond] = {
            'pr_auc': float(ag['pr_auc_mean']),
            'auc_roc': float(ag['auc_roc_mean']),
            'recall': float(ag['recall_mean']),
            'f1_score': float(ag['f1_mean']),
            'umbral_mediano': float(m['umbral_optimo']),
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


def _ejecutar_target(
    etiqueta: str,
    X: pd.DataFrame,
    y: pd.Series,
    ruta_metricas: str,
    ruta_roc_pr: str,
    ruta_pr_train: str,
) -> None:
    """Ejecuta el CV, guarda artefactos y muestra el resumen compacto del target."""
    modelo = BaselineModel(random_state=42)
    metricas = modelo.walk_forward_cv(X, y, n_folds=4, gap_meses=GAP_MESES)
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


if __name__ == "__main__":
    X, y_6m, y_12m = cargar_dataset()
    os.makedirs("docs/figures", exist_ok=True)

    print("=" * 64)
    print("BASELINE \u2014 REGRESI\u00d3N LOG\u00cdSTICA")
    print("=" * 64)

    _ejecutar_target(
        etiqueta="TARGET 6M",
        X=X,
        y=y_6m,
        ruta_metricas="models/baseline_metrics_6m.json",
        ruta_roc_pr="docs/figures/baseline_6m_roc_pr.png",
        ruta_pr_train="docs/figures/baseline_6m_pr_train_por_fold.png",
    )

    _ejecutar_target(
        etiqueta="TARGET 12M",
        X=X,
        y=y_12m,
        ruta_metricas="models/baseline_metrics_12m.json",
        ruta_roc_pr="docs/figures/baseline_12m_roc_pr.png",
        ruta_pr_train="docs/figures/baseline_12m_pr_train_por_fold.png",
    )

    ablacion = {
        'target_6m': _ablacion_target('target_6m', X, y_6m),
        'target_12m': _ablacion_target('target_12m', X, y_12m),
    }
    os.makedirs("models", exist_ok=True)
    with open("models/ablacion_rebalanceo.json", 'w') as f:
        json.dump(ablacion, f, indent=2)
