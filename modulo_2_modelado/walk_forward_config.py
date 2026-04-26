"""
Configuración compartida del Walk-Forward Cross-Validation y hold-out.

Fuente de verdad única para las fechas de corte de folds y sub-ventanas del
hold-out utilizadas por todos los scripts de modelado del Hito 2. Cualquier
cambio en la configuración del CV debe hacerse aquí y repercute automáticamente
en baseline.py, random_forest.py, xgboost_model.py, lightgbm_model.py, etc.

Diseño:
- 4 folds expanding-window con test de 80 meses, gap de 12 meses entre
  train_end y test_start para eliminar el leakage temporal del shift(-12).
- Cada fold de test cubre exactamente una recesión NBER distinta.
- Hold-out intocable segmentado en ventana de periodo en expansión (may 2011 - feb 2020)
  y ventana COVID (mar-abr 2020).
"""

import pandas as pd


FECHAS_CORTE = [
    {
        'nombre': 'Fold 1',
        'train_end':  pd.Timestamp('1980-08-31'),
        'test_start': pd.Timestamp('1981-09-30'),
        'test_end':   pd.Timestamp('1988-04-30')
    },
    {
        'nombre': 'Fold 2',
        'train_end':  pd.Timestamp('1988-04-30'),
        'test_start': pd.Timestamp('1989-05-31'),
        'test_end':   pd.Timestamp('1995-12-31')
    },
    {
        'nombre': 'Fold 3',
        'train_end':  pd.Timestamp('1995-12-31'),
        'test_start': pd.Timestamp('1997-01-31'),
        'test_end':   pd.Timestamp('2003-08-31')
    },
    {
        'nombre': 'Fold 4',
        'train_end':  pd.Timestamp('2003-08-31'),
        'test_start': pd.Timestamp('2004-09-30'),
        'test_end':   pd.Timestamp('2011-04-30')
    },
]

HOLDOUT_START          = pd.Timestamp('2011-05-31')
HOLDOUT_END            = pd.Timestamp('2025-01-31')
HOLDOUT_EXPANSION_END  = pd.Timestamp('2020-02-29')
HOLDOUT_COVID_START    = pd.Timestamp('2020-03-31')
HOLDOUT_COVID_END      = pd.Timestamp('2020-04-30')

GAP_MESES = 12

CONDICIONES_REBALANCEO = ['none', 'balanced', 'smote']


if __name__ == "__main__":
    import json
    import os

    from baseline import cargar_dataset

    X, y_6m, y_12m = cargar_dataset()
    train_start = X.index[0]

    def _conteos(y_ventana: pd.Series) -> dict:
        """Devuelve {pos, neg} de una serie binaria."""
        return {
            "pos": int(y_ventana.sum()),
            "neg": int(len(y_ventana) - y_ventana.sum()),
        }

    folds_json = []
    for f in FECHAS_CORTE:
        mascara_train = X.index <= f["train_end"]
        mascara_test = (X.index >= f["test_start"]) & (X.index <= f["test_end"])
        folds_json.append({
            "nombre": f["nombre"],
            "train_start": str(train_start.date()),
            "train_end": str(f["train_end"].date()),
            "test_start": str(f["test_start"].date()),
            "test_end": str(f["test_end"].date()),
            "train_size": int(mascara_train.sum()),
            "test_size": int(mascara_test.sum()),
            "target_6m": {
                "train_pos": _conteos(y_6m[mascara_train])["pos"],
                "train_neg": _conteos(y_6m[mascara_train])["neg"],
                "test_pos":  _conteos(y_6m[mascara_test])["pos"],
                "test_neg":  _conteos(y_6m[mascara_test])["neg"],
            },
            "target_12m": {
                "train_pos": _conteos(y_12m[mascara_train])["pos"],
                "train_neg": _conteos(y_12m[mascara_train])["neg"],
                "test_pos":  _conteos(y_12m[mascara_test])["pos"],
                "test_neg":  _conteos(y_12m[mascara_test])["neg"],
            },
        })

    mascara_holdout = (X.index >= HOLDOUT_START) & (X.index <= HOLDOUT_END)
    holdout_json = {
        "start": str(HOLDOUT_START.date()),
        "end": str(HOLDOUT_END.date()),
        "size": int(mascara_holdout.sum()),
        "target_6m": _conteos(y_6m[mascara_holdout]),
        "target_12m": _conteos(y_12m[mascara_holdout]),
    }

    mascara_expansion = (X.index >= HOLDOUT_START) & (X.index <= HOLDOUT_EXPANSION_END)
    holdout_expansion_json = {
        "start": str(HOLDOUT_START.date()),
        "end": str(HOLDOUT_EXPANSION_END.date()),
        "size": int(mascara_expansion.sum()),
        "target_6m": _conteos(y_6m[mascara_expansion]),
        "target_12m": _conteos(y_12m[mascara_expansion]),
    }

    mascara_covid = (X.index >= HOLDOUT_COVID_START) & (X.index <= HOLDOUT_COVID_END)
    holdout_covid_json = {
        "start": str(HOLDOUT_COVID_START.date()),
        "end": str(HOLDOUT_COVID_END.date()),
        "size": int(mascara_covid.sum()),
        "target_6m": _conteos(y_6m[mascara_covid]),
        "target_12m": _conteos(y_12m[mascara_covid]),
    }

    splits = {
        "gap_meses": GAP_MESES,
        "folds": folds_json,
        "holdout": holdout_json,
        "holdout_expansion": holdout_expansion_json,
        "holdout_covid": holdout_covid_json,
    }

    os.makedirs("models", exist_ok=True)
    ruta_salida = "models/walk_forward_splits.json"
    with open(ruta_salida, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Splits exportados a {ruta_salida}\n")
    print(json.dumps(splits, indent=2))
