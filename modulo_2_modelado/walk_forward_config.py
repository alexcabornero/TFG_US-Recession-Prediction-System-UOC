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
- Hold-out intocable segmentado en sub-ventana expansiva (may 2011 - feb 2020)
  y sub-ventana COVID (mar-abr 2020).
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
