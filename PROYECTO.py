import flet as ft
import asyncio
import time
import pandas as pd
import matplotlib.pyplot as plt
from flet.matplotlib_chart import MatplotlibChart
from flet.plotly_chart import PlotlyChart
import numpy as np
import gc
from datetime import datetime
import matplotlib
import re
import joblib
import warnings
matplotlib.use("Agg")
# Matplotlib font configuration to avoid missing glyphs in SVG (e.g., Arial)
import matplotlib as mpl
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans", "DejaVu Serif"]
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["axes.unicode_minus"] = False
import os
import colorsys
import unicodedata
from typing import Optional, Tuple, Dict, Any, List
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # Needed for 3D projections
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# --- PDF reportlab imports ---
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak,
    ListFlowable,
    ListItem,
    Flowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import tempfile
import shutil

APP_VERSION = "v1.0.0"

# Ruta por defecto al modelo de machine learning entrenado para diagnóstico.
MODEL_PATH = "C:\\Users\\QTMP0104\\Downloads\\Proyecto\\Proyecto\\Proyecto\\IA\\modelo_vibraciones.pkl"

# Columnas esperadas por el modelo al momento de generar predicciones.
ML_FEATURE_COLUMNS = [
    "rms_acc_ms2",
    "peak_acc_ms2",
    "rms_vel_mm_s",
    "dom_freq_hz",
    "dom_amp_mm_s",
    "r2x",
    "r3x",
    "energy_low",
    "energy_mid",
    "energy_high",
]

_ML_MODEL = None
_ML_MODEL_ERROR = None
_ML_MODEL_VERSION = None


def _load_ml_model():
    """Carga el modelo de machine learning entrenado desde disco (una única vez)."""

    global _ML_MODEL, _ML_MODEL_ERROR, _ML_MODEL_VERSION
    if _ML_MODEL is not None or _ML_MODEL_ERROR is not None:
        return _ML_MODEL
    try:
        _ML_MODEL = joblib.load(MODEL_PATH)
        _ensure_model_feature_names(_ML_MODEL)
        try:
            _ML_MODEL_VERSION = getattr(_ML_MODEL, "__version__", None)
            if not _ML_MODEL_VERSION:
                _ML_MODEL_VERSION = getattr(_ML_MODEL, "version", None)
            if not _ML_MODEL_VERSION and hasattr(_ML_MODEL, "steps"):
                for _, step in getattr(_ML_MODEL, "steps"):
                    _ML_MODEL_VERSION = getattr(step, "__version__", None) or getattr(step, "version", None)
                    if _ML_MODEL_VERSION:
                        break
        except Exception:
            _ML_MODEL_VERSION = None
    except FileNotFoundError as exc:
        _ML_MODEL_ERROR = f"Modelo no encontrado en {MODEL_PATH}: {exc}"
    except Exception as exc:  # pragma: no cover - defensivo
        _ML_MODEL_ERROR = f"No se pudo cargar el modelo de ML: {exc}"
    return _ML_MODEL


def _ensure_model_feature_names(model: Any) -> None:
    """Asigna nombres de features esperados cuando el modelo carece de ellos."""

    try:
        if hasattr(model, "feature_names_in_"):
            names = getattr(model, "feature_names_in_")
            if names is None or len(names) == 0:
                setattr(model, "feature_names_in_", np.array(ML_FEATURE_COLUMNS, dtype=object))
            return
        if hasattr(model, "estimators_") and isinstance(getattr(model, "estimators_"), list):
            setattr(model, "feature_names_in_", np.array(ML_FEATURE_COLUMNS, dtype=object))
            return
        if hasattr(model, "steps") and isinstance(getattr(model, "steps"), list):
            for _, step in getattr(model, "steps"):
                _ensure_model_feature_names(step)
            if not hasattr(model, "feature_names_in_"):
                setattr(model, "feature_names_in_", np.array(ML_FEATURE_COLUMNS, dtype=object))
    except Exception:
        # Si no es posible asignar los nombres, dejamos el modelo tal como está.
        pass


def _run_ml_diagnosis(feature_row: Dict[str, float]) -> Dict[str, Any]:
    """Ejecuta el modelo de ML sobre un conjunto de features y devuelve el resultado."""

    model = _load_ml_model()
    if model is None:
        return {
            "status": "unavailable",
            "message": _ML_MODEL_ERROR or "Modelo no disponible",
        }

    try:
        df_features = pd.DataFrame([feature_row])
        df_features = df_features.reindex(columns=ML_FEATURE_COLUMNS)
        df_features = df_features.fillna(0.0)
        try:
            df_features = df_features.astype(float)
        except Exception:
            df_features = df_features.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        expected_names: List[str] = []
        if hasattr(model, "feature_names_in_"):
            try:
                raw_names = getattr(model, "feature_names_in_")
                if raw_names is not None:
                    expected_names = [str(name) for name in list(raw_names) if name is not None]
            except Exception:
                expected_names = []

        target_columns = expected_names if expected_names else ML_FEATURE_COLUMNS
        df_features = df_features.reindex(columns=target_columns).fillna(0.0)
        features_for_model: Any = df_features
    except Exception as exc:
        return {
            "status": "error",
            "message": f"No se pudieron preparar las features para el modelo: {exc}",
        }

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            raw_pred = model.predict(features_for_model)[0]
        try:
            pred_value = raw_pred.item()  # type: ignore[attr-defined]
        except Exception:
            pred_value = raw_pred
        probabilities = None
        if hasattr(model, "predict_proba"):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    proba = model.predict_proba(features_for_model)
                if proba is not None:
                    probabilities = proba[0].tolist()
                    classes = list(getattr(model, "classes_", []))
                else:
                    classes = []
            except Exception:
                probabilities = None
                classes = []
        else:
            classes = []
        return {
            "status": "ok",
            "label": str(pred_value),
            "raw_prediction": pred_value,
            "probabilities": probabilities,
            "classes": classes,
            "model_version": _ML_MODEL_VERSION,
        }
    except Exception as exc:  # pragma: no cover - robustez
        return {
            "status": "error",
            "message": f"Error al ejecutar el modelo: {exc}",
        }

# Conjunto de fallas consideradas en la Tabla de Charlotte para motores eléctricos.
# Cada entrada incluye un identificador, el nombre de la falla y una descripción breve
# para contextualizar al usuario durante la interpretación del diagnóstico automático.
CHARLOTTE_MOTOR_FAULTS: List[Dict[str, str]] = [
    {
        "code": "EM01",
        "name": "Desbalanceo del rotor",
        "description": "Vibración 1X dominante en dirección radial; aumenta con las RPM.",
    },
    {
        "code": "EM02",
        "name": "Desalineación angular",
        "description": "Elevación de 2X y 3X, con componentes axiales pronunciados.",
    },
    {
        "code": "EM03",
        "name": "Desalineación paralela",
        "description": "Componentes 1X y 2X en dirección radial y axial con desfase entre soportes.",
    },
    {
        "code": "EM04",
        "name": "Holgura mecánica",
        "description": "Multiplicidad de armónicos de 1X y presencia de impactos o subarmónicos.",
    },
    {
        "code": "EM05",
        "name": "Holgura estructural / base suelta",
        "description": "Respuesta amplia entre 1X y 3X acompañada de modulación irregular.",
    },
    {
        "code": "EM06",
        "name": "Resonancia estructural",
        "description": "Picos muy agudos con factor Q alto y sensibilidad extrema a pequeños cambios.",
    },
    {
        "code": "EM07",
        "name": "Soft foot (pata coja)",
        "description": "Variaciones de fase/1X durante el apriete de pernos y fuerte componente axial.",
    },
    {
        "code": "EM08",
        "name": "Eje doblado",
        "description": "1X dominante con fuertes componentes axiales y segundo armónico moderado.",
    },
    {
        "code": "EM09",
        "name": "Rotor excéntrico",
        "description": "1X radial elevado acompañado de bandas laterales sincronizadas con RPM.",
    },
    {
        "code": "EM10",
        "name": "Barras de rotor rotas",
        "description": "Bandas laterales alrededor de 1X y 2X, modulación a frecuencia de resbalamiento.",
    },
    {
        "code": "EM11",
        "name": "Rotor flojo / roce de rotor",
        "description": "Vibración subsíncrona, impactos y crecimiento de armónicos impares.",
    },
    {
        "code": "EM12",
        "name": "Problemas eléctricos del estator",
        "description": "Componentes a 2X línea y picos a 1X línea +/- 1X mecánico.",
    },
    {
        "code": "EM13",
        "name": "Desequilibrio de tensión / armónicos de línea",
        "description": "Elevación persistente de 2X y 3X de línea y modulación armónica.",
    },
    {
        "code": "EM14",
        "name": "Rodamiento - pista externa",
        "description": "Frecuencias BPFO y sus armónicos con posibles bandas laterales a 1X.",
    },
    {
        "code": "EM15",
        "name": "Rodamiento - pista interna",
        "description": "Frecuencias BPFI dominantes y modulación con 1X o frecuencia de rotación.",
    },
    {
        "code": "EM16",
        "name": "Rodamiento - elemento rodante",
        "description": "BSF y sus armónicos con envolvente rica en alta frecuencia.",
    },
    {
        "code": "EM17",
        "name": "Rodamiento - jaula / separador",
        "description": "FTF y subarmónicos, a menudo acompañados de impulsos repetitivos.",
    },
    {
        "code": "EM18",
        "name": "Lubricación deficiente o contaminación",
        "description": "Crecimiento amplio en alta frecuencia y elevación del ruido de fondo.",
    },
    {
        "code": "EM19",
        "name": "Problemas en acoplamiento",
        "description": "Combinación de armónicos 1X-3X y variaciones según la carga transmitida.",
    },
    {
        "code": "EM20",
        "name": "Ventilador o elementos auxiliares",
        "description": "Picos a frecuencias de aspas/paletas y subarmónicos modulados.",
    },
]


def _resolve_fft_window(window_name: Optional[str]) -> str:
    """Normaliza el nombre de ventana FFT a un identificador soportado."""

    try:
        name = str(window_name or "").strip().lower()
    except Exception:
        name = ""
    if name in {"flat-top", "flat_top", "flat top", "flattop"}:
        return "flattop"
    return "hann"


def _build_fft_window(window_name: Optional[str], n: int) -> Tuple[np.ndarray, float]:
    """Genera la ventana solicitada y su ganancia coherente (media)."""

    if n <= 0:
        return np.zeros(0, dtype=float), 1.0
    if n == 1:
        return np.ones(1, dtype=float), 1.0
    name = _resolve_fft_window(window_name)
    if name == "flattop":
        k = np.arange(n, dtype=float)
        denom = float(n - 1) if n > 1 else 1.0
        angle = 2.0 * np.pi * k / denom
        window = (
            1.0
            - 1.93 * np.cos(angle)
            + 1.29 * np.cos(2.0 * angle)
            - 0.388 * np.cos(3.0 * angle)
            + 0.0322 * np.cos(4.0 * angle)
        )
    else:
        window = np.hanning(n)
    window = np.asarray(window, dtype=float)
    cg = float(np.mean(window)) if np.any(window) else 1.0
    if not np.isfinite(cg) or abs(cg) < 1e-12:
        cg = 1.0
    return window, cg


def _charlotte_faults_lines() -> List[str]:
    """Devuelve las descripciones formateadas de las fallas Charlotte para motores."""
    lines: List[str] = []
    for entry in CHARLOTTE_MOTOR_FAULTS:
        code = entry.get("code", "-")
        name = entry.get("name", "Falla")
        desc = entry.get("description", "")
        formatted = f"• {code} – {name}: {desc}"
        lines.append(formatted)
    return lines


def _build_charlotte_reference_table(
    entries: Optional[List[Dict[str, str]]],
    styles,
    accent_color,
):
    """Genera una tabla con estilo para la referencia de Charlotte que respete los anchos."""

    if not entries:
        return None

    style_map = getattr(styles, "byName", {})

    if "CharlotteHeader" not in style_map:
        styles.add(
            ParagraphStyle(
                "CharlotteHeader",
                parent=styles["Heading4"],
                textColor=colors.white,
                alignment=1,
                fontSize=10,
                leading=12,
                spaceAfter=0,
            )
        )
    if "CharlotteCode" not in style_map:
        styles.add(
            ParagraphStyle(
                "CharlotteCode",
                parent=styles["Normal"],
                alignment=1,
                textColor=accent_color,
                fontSize=9,
                leading=11,
                fontName="Helvetica-Bold",
                spaceAfter=0,
            )
        )
    if "CharlotteName" not in style_map:
        styles.add(
            ParagraphStyle(
                "CharlotteName",
                parent=styles["Normal"],
                fontSize=9.5,
                leading=11,
                textColor=colors.HexColor("#2c3e50"),
                spaceAfter=1,
            )
        )
    if "CharlotteDescription" not in style_map:
        styles.add(
            ParagraphStyle(
                "CharlotteDescription",
                parent=styles["Normal"],
                fontSize=8.5,
                leading=10.5,
                textColor=colors.HexColor("#4d5b6a"),
                spaceAfter=3,
            )
        )

    header = [
        Paragraph("<b>Código</b>", styles["CharlotteHeader"]),
        Paragraph("<b>Falla</b>", styles["CharlotteHeader"]),
        Paragraph("<b>Descripción</b>", styles["CharlotteHeader"]),
    ]

    rows: List[List[Any]] = [header]
    for entry in entries:
        code = str(entry.get("code", "-"))
        name = str(entry.get("name", ""))
        desc = str(entry.get("description", ""))
        rows.append(
            [
                Paragraph(code, styles["CharlotteCode"]),
                Paragraph(name, styles["CharlotteName"]),
                Paragraph(desc, styles["CharlotteDescription"]),
            ]
        )

    table = Table(rows, colWidths=[70, 160, 220])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2f2f2")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d7d7d7")),
            ]
        )
    )
    return table


class ProbabilityBar(Flowable):
    """Pequeño progress-bar para visualizar probabilidades en el PDF."""

    def __init__(
        self,
        value: float,
        width: float = 140.0,
        height: float = 8.0,
        fill_color: str = "#3498db",
        back_color: str = "#ecf0f1",
    ) -> None:
        super().__init__()
        self.value = float(np.clip(value, 0.0, 1.0)) if np.isfinite(value) else 0.0
        self.width = float(width)
        self.height = float(height)
        self.fill_color = colors.HexColor(fill_color)
        self.back_color = colors.HexColor(back_color)

    def draw(self) -> None:  # pragma: no cover - dibujo directo
        self.canv.setFillColor(self.back_color)
        radius = self.height / 2.0
        self.canv.roundRect(0, 0, self.width, self.height, radius, stroke=0, fill=1)
        if self.value <= 0:
            return
        self.canv.setFillColor(self.fill_color)
        self.canv.roundRect(0, 0, self.width * self.value, self.height, radius, stroke=0, fill=1)


def _pdf_metric_grid(rows: List[Tuple[str, str]], accent_color) -> Table:
    """Genera una tabla estilizada de métricas clave para el PDF."""

    sample = getSampleStyleSheet()
    label_style = ParagraphStyle(
        name="metric_label_temp",
        parent=sample["Normal"],
        textColor=colors.HexColor("#2c3e50"),
    )
    value_style = ParagraphStyle(
        name="metric_value_temp",
        parent=sample["Normal"],
        alignment=2,
        textColor=colors.HexColor("#2c3e50"),
    )
    data: List[List[Any]] = []
    for label, value in rows:
        data.append([
            Paragraph(f"<b>{label}</b>", label_style),
            Paragraph(value, value_style),
        ])
    tbl = Table(data, colWidths=[260, 180])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#2c3e50")),
                ("LINEABOVE", (0, 0), (-1, 0), 0.3, colors.HexColor("#b0b0b0")),
                ("LINEBELOW", (0, -1), (-1, -1), 0.3, colors.HexColor("#b0b0b0")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d7d7d7")),
            ]
        )
    )
    return tbl


def _pdf_card(contents: List[Any], accent_color, background: str = "#ffffff") -> Table:
    """Envuelve contenidos en una tarjeta estilo app."""

    card = Table([[c] for c in contents], colWidths=[440])
    card.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(background)),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#bcbcbc")),
                ("INNERGRID", (0, 0), (-1, -1), 0.0, colors.white),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return card


def _split_diagnosis(findings: Optional[List[str]]) -> Tuple[Optional[str], List[str]]:
    """Separa la entrada de severidad ISO del resto de hallazgos."""
    severity: Optional[str] = None
    core: List[str] = []
    if not findings:
        return severity, core
    for item in findings:
        text = str(item)
        if severity is None and text.lower().startswith("severidad iso"):
            severity = text
            continue
        core.append(text)
    return severity, core

# =========================
#   Utilidades de rodamientos (frecuencias teóricas)
# =========================
def bearing_freqs_from_geometry(
    rpm: Optional[float],
    n_elements: Optional[int],
    d_mm: Optional[float],
    D_mm: Optional[float],
    theta_deg: Optional[float] = 0.0,
) -> Dict[str, Optional[float]]:
    """
    Calcula FTF/BPFO/BPFI/BSF a partir de la geometría del rodamiento y RPM.
    Parámetros:
      - rpm: velocidad del eje [rev/min]
      - n_elements: número de elementos rodantes
      - d_mm: diámetro del elemento [mm]
      - D_mm: diámetro de paso (pitch) [mm]
      - theta_deg: ángulo de contacto [grados]

    Devuelve dict con claves: ftf, bpfo, bpfi, bsf (en Hz). None si faltan datos.
    Fórmulas estándar (frecuencia del eje f1 = rpm/60):
      FTF  = 0.5 * f1 * (1 - (d/D) * cosθ)
      BPFO = 0.5 * n * f1 * (1 - (d/D) * cosθ)
      BPFI = 0.5 * n * f1 * (1 + (d/D) * cosθ)
      BSF  = (D/d) * 0.5 * f1 * (1 - ((d/D) * cosθ)**2)
    """
    try:
        if rpm is None or n_elements is None or d_mm is None or D_mm is None:
            return {"ftf": None, "bpfo": None, "bpfi": None, "bsf": None}
        f1 = float(rpm) / 60.0
        if f1 <= 0 or n_elements <= 0 or d_mm <= 0 or D_mm <= 0:
            return {"ftf": None, "bpfo": None, "bpfi": None, "bsf": None}
        ratio = float(d_mm) / float(D_mm)
        # Físicamente 0 < d/D < 1; limitar para evitar valores no realistas
        if not np.isfinite(ratio):
            return {"ftf": None, "bpfo": None, "bpfi": None, "bsf": None}
        ratio = float(min(0.999, max(1e-9, ratio)))
        th = float(theta_deg or 0.0)
        cth = float(np.cos(np.deg2rad(th)))
        ftf = 0.5 * f1 * (1.0 - ratio * cth)
        bpfo = 0.5 * float(n_elements) * f1 * (1.0 - ratio * cth)
        bpfi = 0.5 * float(n_elements) * f1 * (1.0 + ratio * cth)
        # Evitar división por cero en BSF si d_mm ~ 0
        bsf = (float(D_mm) / float(d_mm)) * 0.5 * f1 * (1.0 - ( (ratio * cth) ** 2 )) if d_mm > 0 else None
        return {"ftf": ftf, "bpfo": bpfo, "bpfi": bpfi, "bsf": bsf}
    except Exception:
        return {"ftf": None, "bpfo": None, "bpfi": None, "bsf": None}

# =========================
#   Filtros anti-alias y decimación
# =========================
def _kaiser_beta(atten_db: float) -> float:
    A = float(max(0.0, atten_db))
    if A > 50.0:
        return 0.1102 * (A - 8.7)
    if A >= 21.0:
        return 0.5842 * (A - 21.0) ** 0.4 + 0.07886 * (A - 21.0)
    return 0.0

def design_kaiser_lowpass(fs_hz: float, f_pass_hz: float, f_stop_hz: float, atten_db: float = 80.0) -> np.ndarray:
    """
    Diseña un FIR pasa‑bajas de fase lineal (ventana Kaiser) para anti‑aliasing.
    - fs_hz: frecuencia de muestreo actual
    - f_pass_hz: borde de banda de paso (se mantiene plano)
    - f_stop_hz: inicio de banda de rechazo (>= Nyquist de la señal decimada deseada)
    - atten_db: atenuación en banda de rechazo objetivo (dB)

    Devuelve coeficientes (taps) normalizados a ganancia DC = 1.
    """
    fs = float(fs_hz)
    f_pass = float(f_pass_hz)
    f_stop = float(f_stop_hz)
    if not (fs > 0 and 0 < f_pass < f_stop < fs * 0.5):
        raise ValueError("Parámetros inválidos para diseño del FIR (revisa fs, f_pass, f_stop).")

    # Parámetros Kaiser
    beta = _kaiser_beta(atten_db)
    delta_f = max(1e-9, f_stop - f_pass)
    # Ancho de transición en rad/muestra (0..pi)
    d_omega = 2.0 * np.pi * (delta_f / fs)
    # Longitud aproximada (Oppenheim/Schafer). A mayor atenuación o menor transición -> más taps
    numtaps = int(np.ceil((max(atten_db, 0.0) - 8.0) / (2.285 * d_omega))) + 1
    numtaps = max(11, numtaps)
    if numtaps % 2 == 0:
        numtaps += 1  # tipo I, fase lineal y retardo entero

    # Corte en mitad de la banda de transición
    f_c = 0.5 * (f_pass + f_stop)
    f_c_n = f_c / fs  # ciclos por muestra
    n = np.arange(numtaps)
    m = (numtaps - 1) / 2.0
    # Kernel ideal (sinc) y ventana Kaiser
    h = 2.0 * f_c_n * np.sinc(2.0 * f_c_n * (n - m))
    w = np.kaiser(numtaps, beta)
    h *= w
    # Normalizar ganancia DC a 1
    h /= np.sum(h)
    return h.astype(float)

def anti_alias_and_decimate(time_s: np.ndarray,
                            x: np.ndarray,
                            f_max_hz: float,
                            margin: float = 2.8,
                            atten_db: float = 80.0) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
    """
    Filtra con FIR anti‑alias y decima para evitar problemas por muestrear de más.
    - f_max_hz: frecuencia máxima de análisis que se desea conservar
    - margin: factor >= 2.5–3 sobre f_max para fs_out (regla práctica)
    - atten_db: atenuación objetivo del anti‑alias (dB)

    Devuelve: (t_dec, x_dec, fs_out, info)
    """
    t = np.asarray(time_s).astype(float).ravel()
    y = np.asarray(x).astype(float).ravel()
    if t.size < 3 or y.size != t.size:
        raise ValueError("Se requieren series t y x del mismo tamaño (>=3).")
    dt = float(np.median(np.diff(t)))
    if not (dt > 0):
        raise ValueError("No se pudo estimar dt > 0 para decimación.")
    fs_in = 1.0 / dt

    if not (f_max_hz and f_max_hz > 0):
        # Nada que hacer si no hay banda objetivo
        return t, y, fs_in, {"M": 1, "fs_in": fs_in, "fs_out": fs_in, "note": "sin decimación"}

    # Factor entero de decimación buscando fs_out >= margin * f_max
    M = int(np.floor(fs_in / (margin * f_max_hz)))
    if M < 2:
        return t, y, fs_in, {"M": 1, "fs_in": fs_in, "fs_out": fs_in, "note": "fs ya adecuada"}

    fs_out = fs_in / M
    nyq_out = 0.5 * fs_out
    # Banda del FIR en fs_in: paso hasta f_max, rechazo a 0.9*Nyquist de fs_out (colchón)
    f_pass = min(f_max_hz, 0.9 * nyq_out)
    f_stop = 0.95 * nyq_out

    delta_f = max(1e-9, f_stop - f_pass)
    d_omega = 2.0 * np.pi * (delta_f / fs_in)
    est_numtaps = int(np.ceil((max(atten_db, 0.0) - 8.0) / (2.285 * d_omega))) + 1
    if est_numtaps % 2 == 0:
        est_numtaps += 1
    MAX_FIR_TAPS = 4095
    if est_numtaps > MAX_FIR_TAPS or est_numtaps > y.size:
        idx = np.arange(0, y.size, M)
        if idx.size <= 1:
            return t[::M], y[::M], fs_out, {
                "M": int(M),
                "numtaps": 1,
                "fs_in": float(fs_in),
                "fs_out": float(fs_out),
                "f_pass_hz": float(f_pass),
                "f_stop_hz": float(f_stop),
                "atten_db": float(atten_db),
                "note": "decimación directa (FIR omitido por límite de taps)",
            }
        seg_lengths = np.minimum(M, y.size - idx).astype(float)
        sum_y = np.add.reduceat(y, idx)
        sum_t = np.add.reduceat(t, idx)
        x_dec = (sum_y / seg_lengths).astype(float)
        t_dec = (sum_t / seg_lengths).astype(float)
        info = {
            "M": int(M),
            "numtaps": int(min(est_numtaps, MAX_FIR_TAPS)),
            "fs_in": float(fs_in),
            "fs_out": float(fs_out),
            "f_pass_hz": float(f_pass),
            "f_stop_hz": float(f_stop),
            "atten_db": float(atten_db),
            "note": "decimación simplificada (promedio por bloques)",
        }
        return t_dec, x_dec, fs_out, info

    h = design_kaiser_lowpass(fs_in, f_pass, f_stop, atten_db=atten_db)
    gd = (len(h) - 1) // 2

    # Convolución y recorte de transitorios en los extremos
    y_f = np.convolve(y, h, mode="full")
    # Centrar (compensar retardo de grupo) y quitar bordes con transitorio
    y_lin = y_f[gd:gd + y.size]
    start = gd
    stop = y_lin.size - gd
    if stop <= start:
        # Si la señal es demasiado corta comparada con el FIR, no recortamos extremos
        start, stop = 0, y_lin.size
    yy = y_lin[start:stop]
    tt = t[start:stop]

    # Decimación: tomar una de cada M muestras
    t_dec = tt[::M]
    x_dec = yy[::M]
    fs_out = fs_in / M

    info = {
        "M": int(M),
        "numtaps": int(len(h)),
        "fs_in": float(fs_in),
        "fs_out": float(fs_out),
        "f_pass_hz": float(f_pass),
        "f_stop_hz": float(f_stop),
        "atten_db": float(atten_db),
    }
    return t_dec, x_dec, fs_out, info

# =========================
#   Analizador independiente
# =========================
def analyze_vibration(
    time_s: np.ndarray,
    acc_ms2: np.ndarray,
    rpm: Optional[float] = None,
    line_freq_hz: Optional[float] = None,
    bpfo_hz: Optional[float] = None,
    bpfi_hz: Optional[float] = None,
    bsf_hz: Optional[float] = None,
    ftf_hz: Optional[float] = None,
    gear_teeth: Optional[int] = None,
    segment: Optional[Tuple[float, float]] = None,
    pre_decimate_to_fmax_hz: Optional[float] = None,
    pre_decimate_margin: float = 2.8,
    pre_decimate_atten_db: float = 80.0,
    env_bp_lo_hz: Optional[float] = None,
    env_bp_hi_hz: Optional[float] = None,
    tol_frac: float = 0.02,
    min_bins: int = 2,
    min_snr_db: float = 6.0,
    top_k_peaks: int = 5,
    fft_window: str = "hann",
) -> Dict[str, Any]:
    analysis_start = time.perf_counter()

    def _highpass_first_order(y: np.ndarray, dt: float, cutoff_hz: float = 0.5) -> np.ndarray:
        y = np.asarray(y, dtype=float).ravel()
        if y.size < 2 or not np.isfinite(dt) or dt <= 0 or not np.isfinite(cutoff_hz) or cutoff_hz <= 0:
            return y
        rc = 1.0 / (2.0 * np.pi * cutoff_hz)
        alpha = rc / (rc + dt)
        out = np.empty_like(y)
        prev_x = float(y[0])
        prev_y = 0.0
        out[0] = 0.0
        for i in range(1, y.size):
            x_i = float(y[i])
            prev_y = alpha * (prev_y + x_i - prev_x)
            out[i] = prev_y
            prev_x = x_i
        return out
    # ======= FENCE: aislar entradas y normalizar tiempo =======
    # Copias duras (evita vistas / referencias compartidas entre corridas)
       # ============================================================
    # 🚨 Aislar sesión de análisis y limpiar caché numérico
    # ============================================================
    np.seterr(all="ignore")
    gc.collect()

    time_s = np.array(time_s, dtype=float, copy=True)
    acc_ms2 = np.array(acc_ms2, dtype=float, copy=True)

    # 🔧 Limpiar datos corruptos o fuera de orden
    m = np.isfinite(time_s) & np.isfinite(acc_ms2)
    time_s, acc_ms2 = time_s[m], acc_ms2[m]
    if time_s.size < 2:
        raise ValueError("Datos insuficientes.")
    if not np.all(np.diff(time_s) >= 0):
        idx = np.argsort(time_s)
        time_s, acc_ms2 = time_s[idx], acc_ms2[idx]
    time_s -= float(time_s[0])

    # 🔧 Quitar componente DC y tendencia lineal antes de integrar
    acc_ms2 -= np.mean(acc_ms2)
    if acc_ms2.size >= 2:
        try:
            p = np.polyfit(np.arange(acc_ms2.size), acc_ms2, 1)
            acc_ms2 -= (p[0] * np.arange(acc_ms2.size) + p[1])
        except Exception:
            pass

    def _to_1d(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x).astype(float).ravel()
        return x
    def _clean_pair(t: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        m = np.isfinite(t) & np.isfinite(y)
        t2, y2 = t[m], y[m]
        if len(t2) > 1 and not np.all(np.diff(t2) >= 0):
            idx = np.argsort(t2)
            t2, y2 = t2[idx], y2[idx]
        return t2, y2
    def _segment(t: np.ndarray, y: np.ndarray, seg: Optional[Tuple[float,float]]):
        if seg is None or len(t) < 2:
            return t, y
        t0, t1 = seg
        if t0 > t1:
            t0, t1 = t1, t0
        m = (t >= t0) & (t <= t1)
        tt, yy = t[m], y[m]
        return (tt, yy) if len(tt) >= 2 else (t, y)
    def _fs_from_time(t: np.ndarray) -> Tuple[float, float]:
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
        fs = 1.0 / dt if dt > 0 else 0.0
        return fs, dt
    window_type = _resolve_fft_window(fft_window)

    def _acc_fft_to_vel_mm_s(y: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Convierte aceleración (m/s²) en velocidad (mm/s) mediante integración FFT
        con detrending para evitar deriva y mantener el RMS estable entre análisis.
        Devuelve: (frecuencias, magnitud de aceleración, magnitud de velocidad en mm/s, RMS temporal de velocidad en mm/s).
        """

        N = len(y)
        if N < 2 or dt <= 0:
            return np.array([]), np.array([]), np.array([]), 0.0

        # Eje de tiempo para detrend estable
        t_axis = np.arange(N, dtype=float) * dt

        # ---- PRE: quitar DC y tendencia a la aceleración ----
        y_proc = np.asarray(y, dtype=float) - float(np.mean(y))
        if y_proc.size >= 2:
            try:
                p = np.polyfit(t_axis, y_proc, 1)
                y_proc = y_proc - (p[0] * t_axis + p[1])
            except Exception:
                pass

        # ---- RMS temporal de velocidad (integración sin deriva) ----
        vel_time = np.cumsum(y_proc) * dt
        if vel_time.size:
            vel_time -= float(np.mean(vel_time))
        rms_vel_time_mm = 1000.0 * float(np.sqrt(np.mean(vel_time**2))) if vel_time.size else 0.0

        # ---- FFT de aceleración y conversión a velocidad ----
        yf = np.fft.rfft(y_proc)
        xf = np.fft.rfftfreq(N, dt)
        w = 2.0 * np.pi * xf
        if w.size:
            w[0] = np.inf  # evita división por cero en DC

        mag_acc = (2.0 / N) * np.abs(yf)
        mag_vel_mm = 1000.0 * (mag_acc / w)
        if mag_vel_mm.size:
            mag_vel_mm[0] = 0.0  # DC no tiene sentido físico en velocidad
            try:
                # Atenuar frecuencias muy bajas para evitar explosiones por integración
                min_vel_freq = 0.5  # Hz
                low_idx = xf < min_vel_freq
                mag_vel_mm[low_idx] = 0.0
            except Exception:
                pass

        return xf, mag_acc, mag_vel_mm, rms_vel_time_mm
    def _analytic_signal(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        N = len(y)
        if N < 2:
            return y.astype(complex)
        Y = np.fft.fft(y)
        h = np.zeros(N)
        if N % 2 == 0:
            h[0] = 1
            h[N // 2] = 1
            h[1:N // 2] = 2
        else:
            h[0] = 1
            h[1:(N + 1) // 2] = 2
        Z = np.fft.ifft(Y * h)
        return Z
    def _envelope_spectrum(y: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        N = len(y)
        if N < 2 or dt <= 0:
            return np.array([]), np.array([])
        z = _analytic_signal(y)
        env = np.abs(z)
        env = env - float(np.mean(env))
        Ef = np.fft.fft(env)
        xf = np.fft.fftfreq(N, dt)[: N // 2]
        mag = 2.0 / N * np.abs(Ef[: N // 2])
        return xf, mag
    def _bandpass_fft(y: np.ndarray, dt: float, f_lo: float, f_hi: float) -> np.ndarray:
        try:
            if dt <= 0:
                return y
            fs = 1.0 / dt
            if not (f_lo and f_hi) or not (0.0 < f_lo < f_hi < 0.5 * fs):
                return y
            N = len(y)
            Y = np.fft.fft(y)
            f = np.fft.fftfreq(N, dt)
            mask = (np.abs(f) >= float(f_lo)) & (np.abs(f) <= float(f_hi))
            Yf = np.where(mask, Y, 0.0)
            yf = np.fft.ifft(Yf).real
            return yf.astype(float)
        except Exception:
            return y
    def _amp_near(xf: np.ndarray, spec: np.ndarray, f: Optional[float], df: float) -> float:
        if xf is None or spec is None or len(xf) == 0 or len(spec) == 0:
            return 0.0
        if f is None or not np.isfinite(f) or f <= 0:
            return 0.0
        bw = max(tol_frac * f, min_bins * df)
        idx = (xf >= (f - bw)) & (xf <= (f + bw))
        return float(np.max(spec[idx])) if np.any(idx) else 0.0
    def _find_top_peaks(xf: np.ndarray, y: np.ndarray, k: int, min_freq: float = 0.5, snr_db: float = 6.0) -> List[Dict[str, float]]:
        if len(xf) == 0 or len(y) == 0:
            return []
        mask = xf >= min_freq
        xv = xf[mask]
        yv = y[mask]
        if len(yv) == 0:
            return []
        ref = float(np.median(yv) + 1e-12)
        snr = 20.0 * np.log10(np.maximum(yv, 1e-12) / ref)
        cand = np.where(snr >= snr_db)[0]
        if len(cand) == 0:
            idx = np.argsort(yv)[-k:][::-1]
        else:
            idx = cand[np.argsort(yv[cand])[-k:]][::-1]
        peaks = []
        for i in idx[:k]:
            peaks.append({"f_hz": float(xv[i]), "amp": float(yv[i]), "snr_db": float(snr[i])})
        return peaks
    def _severity_iso_mm_s(rms_mm_s: float) -> Tuple[str, str]:
        if rms_mm_s <= 2.8:
            return "Buena (Aceptable)", "#2ecc71"
        elif rms_mm_s <= 4.5:
            return "Satisfactoria (Vigilancia)", "#f1c40f"
        elif rms_mm_s <= 7.1:
            return "Insatisfactoria (Crítica)", "#e67e22"
        else:
            return "Inaceptable (Riesgo de daño)", "#e74c3c"
    def _get_1x(dom_freq_guess: Optional[float], rpm_opt: Optional[float]) -> float:
        try:
            if rpm_opt and np.isfinite(rpm_opt) and rpm_opt > 0:
                return float(rpm_opt) / 60.0
        except Exception:
            pass
        try:
            if dom_freq_guess and np.isfinite(dom_freq_guess) and dom_freq_guess > 0:
                return float(dom_freq_guess)
        except Exception:
            pass
        return 0.0
    t = _to_1d(time_s)
    a = _to_1d(acc_ms2)
    t, a = _clean_pair(t, a)
    if len(t) < 2:
        raise ValueError("Datos insuficientes.")
    t, a = _segment(t, a, segment)
    predec_info = None
    if pre_decimate_to_fmax_hz is not None:
        try:
            t, a, fs_try, info = anti_alias_and_decimate(
                t, a, f_max_hz=float(pre_decimate_to_fmax_hz), margin=float(pre_decimate_margin), atten_db=float(pre_decimate_atten_db)
            )
            predec_info = info
        except Exception:
            predec_info = {"error": "falló pre-decimación"}
            # continuar con datos originales
            pass
    fs, dt = _fs_from_time(t)

    # Preprocesado: quitar DC/tendencia para evitar pico LF en FFT/integración
    # Esto mejora la visualización y la estimación de dominante.
    try:
        if len(t) >= 2:
            x0 = t - float(t[0])
            p = np.polyfit(x0, a, 1)
            trend = p[0] * x0 + p[1]
        else:
            trend = np.full_like(a, float(np.mean(a)))
    except Exception:
        trend = np.full_like(a, float(np.mean(a)))
    a_proc = a - trend
    df = fs / len(a) if fs > 0 and len(a) > 0 else 0.0
    a_hp = _highpass_first_order(a_proc, dt, cutoff_hz=0.5)
    xf, mag_acc, mag_vel_mm, rms_vel_time_mm = _acc_fft_to_vel_mm_s(a_hp, dt)
    # RMS de aceleración sin DC/tendencia
    rms_time_acc = float(np.sqrt(np.mean(a_hp**2))) if len(a_hp) else 0.0
    peak_acc = float(np.max(np.abs(a))) if len(a) else 0.0
    pp_acc = float(np.ptp(a)) if len(a) else 0.0
    if len(mag_vel_mm) > 0:
        # Dominante ignorando muy baja frecuencia para evitar sesgos por DC
        dom_min_hz = 0.5
        mag_for_dom = mag_vel_mm.copy()
        try:
            if len(xf) == len(mag_for_dom):
                mag_for_dom[xf < dom_min_hz] = 0.0
        except Exception:
            pass
        idx_dom = int(np.argmax(mag_for_dom)) if len(mag_for_dom) else 0
        dom_freq = float(xf[idx_dom]) if len(xf) > idx_dom else 0.0
        dom_amp = float(mag_vel_mm[idx_dom]) if len(mag_vel_mm) > idx_dom else 0.0
        if (dom_freq <= 0.0) and np.any(xf >= dom_min_hz):
            try:
                candidates = np.where(xf >= dom_min_hz)[0]
                if candidates.size:
                    best_idx = candidates[int(np.argmax(mag_vel_mm[candidates]))]
                    dom_freq = float(xf[best_idx])
                    dom_amp = float(mag_vel_mm[best_idx])
                    idx_dom = int(best_idx)
            except Exception:
                pass
    else:
        dom_freq, dom_amp = 0.0, 0.0
    # Severidad basada en RMS de velocidad temporal (mm/s)
    rms_vel_spec_mm = rms_vel_time_mm
    f1 = _get_1x(dom_freq, rpm)
    rpm_est = None
    if rpm and np.isfinite(rpm) and rpm > 0:
        rpm_est = float(rpm)
    elif f1 and f1 > 0:
        rpm_est = float(f1 * 60.0)
    elif dom_freq and dom_freq > 0:
        rpm_est = float(dom_freq * 60.0)
    r2x = _amp_near(xf, mag_vel_mm, 2.0 * f1 if f1 > 0 else 0.0, df) / (dom_amp + 1e-12)
    r3x = _amp_near(xf, mag_vel_mm, 3.0 * f1 if f1 > 0 else 0.0, df) / (dom_amp + 1e-12)
    if len(xf) > 1:
        spec_energy = mag_vel_mm**2
        try:
            e_total = float(np.trapz(spec_energy, xf))
        except Exception:
            e_total = float(np.sum(spec_energy))
        if e_total <= 0:
            e_total = 1e-12
        def _band_energy_int(lo: float, hi: float) -> float:
            mask = (xf >= lo) & (xf < hi)
            if not np.any(mask):
                return 0.0
            try:
                return float(np.trapz(spec_energy[mask], xf[mask]))
            except Exception:
                return float(np.sum(spec_energy[mask]))
        e_low = _band_energy_int(0.0, 30.0)
        e_mid = _band_energy_int(30.0, 120.0)
        e_high = _band_energy_int(120.0, float(np.max(xf) if np.any(xf) else 120.0))
    else:
        e_total = 1e-12
        e_low = e_mid = e_high = 0.0
    energy_low_frac = float(e_low / e_total) if e_total > 0 else 0.0
    energy_mid_frac = float(e_mid / e_total) if e_total > 0 else 0.0
    energy_high_frac = float(e_high / e_total) if e_total > 0 else 0.0
    peaks_fft = _find_top_peaks(xf, mag_vel_mm, k=top_k_peaks, min_freq=0.5, snr_db=min_snr_db)
    # Envolvente: opcionalmente aplicar band-pass previo
    a_env_src = a_proc
    try:
        _lo = float(env_bp_lo_hz) if env_bp_lo_hz is not None else None
    except Exception:
        _lo = None
    try:
        _hi = float(env_bp_hi_hz) if env_bp_hi_hz is not None else None
    except Exception:
        _hi = None
    if (_lo is not None) and (_hi is not None):
        a_env_src = _bandpass_fft(a_proc, dt, _lo, _hi)
    xf_env, env_spec = _envelope_spectrum(a_env_src, dt)
    peaks_env = _find_top_peaks(xf_env, env_spec, k=top_k_peaks, min_freq=1.0, snr_db=min_snr_db)
    # Resolución de la envolvente para tolerancias correctas
    df_env = 0.0
    try:
        if xf_env is not None and len(xf_env) > 1:
            df_env = float(np.median(np.diff(xf_env)))
    except Exception:
        df_env = 0.0
    sev_label, sev_color = _severity_iso_mm_s(rms_vel_spec_mm)
    if rms_vel_spec_mm > 7.1 and "insatisfact" not in sev_label.lower() and "inacept" not in sev_label.lower():
        sev_label = "Insatisfactoria (Crítica)"
        sev_color = "#e67e22"
    findings: List[str] = []
    findings.append(f"Severidad ISO: {sev_label} (RMS={rms_vel_spec_mm:.3f} mm/s)")
    ml_features = {
        "rms_acc_ms2": rms_time_acc,
        "peak_acc_ms2": peak_acc,
        "pp_acc_ms2": pp_acc,
        "rms_vel_mm_s": rms_vel_spec_mm,
        "dom_freq_hz": dom_freq,
        "dom_amp_mm_s": dom_amp,
        "r2x": r2x,
        "r3x": r3x,
        "energy_low": energy_low_frac,
        "energy_mid": energy_mid_frac,
        "energy_high": energy_high_frac,
    }
    order_summary: Dict[str, Any] = {
        "rpm_est": rpm_est,
        "f1_hz": f1 if f1 else 0.0,
    }
    if rpm_est and rpm_est > 0:
        base_f = rpm_est / 60.0
        order_summary.update(
            {
                "amp_1x": _amp_near(xf, mag_vel_mm, base_f, df),
                "amp_2x": _amp_near(xf, mag_vel_mm, 2.0 * base_f, df),
                "amp_3x": _amp_near(xf, mag_vel_mm, 3.0 * base_f, df),
            }
        )
        order_peaks: List[Dict[str, float]] = []
        for peak in peaks_fft or []:
            freq = float(peak.get("f_hz", 0.0))
            if freq <= 0:
                continue
            order_val = freq / base_f if base_f > 0 else 0.0
            order_peaks.append(
                {
                    "f_hz": freq,
                    "amp": float(peak.get("amp", 0.0)),
                    "order": float(order_val),
                }
            )
        order_summary["peaks"] = order_peaks
    ml_result = _run_ml_diagnosis(ml_features)
    if ml_result.get("status") == "ok" and ml_result.get("label"):
        findings.append(f"Diagnóstico ML: {ml_result['label']}")
    elif ml_result.get("status") == "error" and ml_result.get("message"):
        findings.append(f"Diagnóstico ML no disponible: {ml_result['message']}")
    elif ml_result.get("status") == "unavailable" and ml_result.get("message"):
        findings.append(f"Modelo ML no disponible: {ml_result['message']}")
    def _severity_rank(text: str) -> int:
        txt = (text or "").lower()
        if "inacept" in txt or "riesgo" in txt:
            return 3
        if "insatisf" in txt or "crít" in txt:
            return 2
        if "satisf" in txt or "vigil" in txt or "warning" in txt:
            return 1
        return 0
    def _ml_rank_from_label(label: str) -> Optional[int]:
        if not label:
            return None
        txt = label.lower()
        good_kw = ["normal", "aceptable", "ok", "sin falla", "healthy", "buena"]
        warn_kw = ["vigil", "alert", "moderada", "warning", "incipiente"]
        bad_kw = ["criti", "falla", "fault", "severa", "severe", "critical", "grave", "danger"]
        if any(k in txt for k in bad_kw):
            return 3
        if any(k in txt for k in warn_kw):
            return 2
        if any(k in txt for k in good_kw):
            return 0
        # Si etiqueta coincide con catálogo Charlotte (código EMxx) asumir condición anómala
        if re.search(r"em\d{2}", txt):
            return 2
        return None
    iso_rank = _severity_rank(sev_label)
    conflict_note = None
    ml_rank = None
    if ml_result.get("status") == "ok":
        ml_rank = _ml_rank_from_label(str(ml_result.get("label", "")))
        if ml_rank is not None and abs(ml_rank - iso_rank) >= 2:
            if ml_rank < iso_rank:
                conflict_note = "ML indica condición más benigna que la severidad ISO. Revisar datos."
            else:
                conflict_note = "ML indica condición más severa que ISO. Validar diagnóstico."
    if conflict_note:
        findings.append(f"⚠️ Conflicto ISO vs ML: {conflict_note}")
    if f1 > 0 and dom_freq > 0:
        if (abs(dom_freq - f1) <= max(tol_frac * f1, min_bins * df)) and (r2x < 0.5) and (r3x < 0.4) and (e_low / e_total > 0.5):
            findings.append("Desbalanceo probable: 1X dominante, 2X/3X bajos, energía en baja frecuencia.")
    if r2x >= 0.6 or r3x >= 0.4:
        findings.append("Desalineación probable: armónicos 2X/3X elevados respecto a 1X.")
    if gear_teeth and gear_teeth > 0 and f1 > 0:
        fmesh = gear_teeth * f1
        a_mesh = _amp_near(xf, mag_vel_mm, fmesh, df)
        if a_mesh > 0.2 * (dom_amp + 1e-12):
            findings.append(f"Engranes: componente en malla ~{fmesh:.1f} Hz.")
    bearing_hits = []
    bearing_matches: Dict[str, Dict[str, float]] = {}
    for name, freq in (("BPFO", bpfo_hz), ("BPFI", bpfi_hz), ("BSF", bsf_hz), ("FTF", ftf_hz)):
        if freq and freq > 0:
            tol_env = df_env if df_env > 0 else (df if df > 0 else (1.0/len(a) if len(a)>0 else 0.1))
            a_env = _amp_near(xf_env, env_spec, freq, tol_env)
            if a_env > 0:
                # Sidebands en envolvente ±k*f1 (k=1..2) si f1 válido
                has_sb = False
                if f1 and f1 > 0:
                    sb_amps = []
                    for k in (1, 2):
                        sb_amps.append(_amp_near(xf_env, env_spec, freq - k * f1, tol_env))
                        sb_amps.append(_amp_near(xf_env, env_spec, freq + k * f1, tol_env))
                    try:
                        sb_vals = [float(s) for s in sb_amps if s is not None]
                        sb_avg = float(np.mean(sb_vals)) if sb_vals else 0.0
                    except Exception:
                        sb_avg = 0.0
                    if sb_avg >= 0.2 * a_env:
                        has_sb = True
                bearing_hits.append(name + (" (SB)" if has_sb else ""))
            if xf_env is not None and env_spec is not None and len(xf_env) and len(env_spec):
                bw = max(tol_frac * freq, tol_env if tol_env > 0 else df if df > 0 else 0.5)
                idx = np.where(np.abs(xf_env - freq) <= (bw if bw > 0 else 0.5))[0]
                if idx.size:
                    peak_idx = idx[np.argmax(env_spec[idx])]
                    bearing_matches[name] = {
                        "target_hz": float(freq),
                        "match_hz": float(xf_env[peak_idx]),
                        "amp": float(env_spec[peak_idx]),
                        "delta_hz": float(xf_env[peak_idx] - freq),
                    }
    if bearing_hits:
        findings.append("Rodamientos: evidencia en envolvente para " + ", ".join(bearing_hits))
    else:
        # Modo automático parcial: sugerir posible defecto de rodamiento si NO hay BPFO/BPFI/BSF/FTF
        # y se observan picos destacados en el espectro de envolvente fuera de armónicos conocidos.
        try:
            if not any([(bpfo_hz and bpfo_hz > 0), (bpfi_hz and bpfi_hz > 0), (bsf_hz and bsf_hz > 0), (ftf_hz and ftf_hz > 0)]):
                # Construir lista de frecuencias conocidas a ignorar (1X..6X, línea y malla si aplica)
                known_env = []
                if f1 and f1 > 0:
                    for k in range(1, 7):
                        known_env.append(k * f1)
                if line_freq_hz and line_freq_hz > 0:
                    known_env.extend([line_freq_hz, 2.0 * line_freq_hz])
                if gear_teeth and gear_teeth > 0 and f1 and f1 > 0:
                    known_env.append(gear_teeth * f1)

                def _near_known_env(f):
                    for fk in known_env:
                        bw = max(tol_frac * max(f, fk), max(2, min_bins) * (df if df > 0 else 0.0))
                        if abs(f - fk) <= (bw if bw > 0 else 1.0):
                            return True
                    return False

                # Elegir picos de la envolvente significativos fuera de las conocidas
                cand = []
                for p in (peaks_env or []):
                    f0 = float(p.get("f_hz", 0.0))
                    a0 = float(p.get("amp", 0.0))
                    if f0 <= 1.0 or a0 <= 0:
                        continue
                    if _near_known_env(f0):
                        continue
                    cand.append((f0, a0))
                # Requiere al menos 2 picos relevantes para sugerir
                if len(cand) >= 2:
                    cand.sort(key=lambda x: x[1], reverse=True)
                    top_fs = ", ".join(f"{f:.1f} Hz" for f, _ in cand[:3])
                    findings.append(f"Rodamientos (modo automático parcial): picos en envolvente ~ {top_fs}.")
        except Exception:
            pass
    if line_freq_hz and line_freq_hz > 0:
        a_line = _amp_near(xf, mag_vel_mm, line_freq_hz, df)
        a_2line = _amp_near(xf, mag_vel_mm, 2.0 * line_freq_hz, df)
        if (a_line > 0.2 * (dom_amp + 1e-12)) or (a_2line > 0.2 * (dom_amp + 1e-12)):
            findings.append(f"Eléctrico: componentes en {line_freq_hz:.0f} Hz y/o {2*line_freq_hz:.0f} Hz.")
    # Resonancias estructurales: picos agudos no armonicos con Q alto
    try:
        if len(peaks_fft) > 0 and len(xf) > 3:
            # Conjunto de frecuencias conocidas a evitar
            known = []
            if f1 and f1 > 0:
                for k in range(1, 9):
                    known.append(k * f1)
            if line_freq_hz and line_freq_hz > 0:
                known.extend([line_freq_hz, 2.0 * line_freq_hz])
            if gear_teeth and gear_teeth > 0 and f1 and f1 > 0:
                known.append(gear_teeth * f1)
            for name, freq in (("BPFO", bpfo_hz), ("BPFI", bpfi_hz), ("BSF", bsf_hz), ("FTF", ftf_hz)):
                if freq and freq > 0:
                    known.append(freq)
            def _near_any(f):
                for fk in known:
                    bw = max(tol_frac * max(f, fk), max(2, min_bins) * (df if df > 0 else 0.0))
                    if abs(f - fk) <= (bw if bw > 0 else 1.0):
                        return True
                return False
            resonances = []
            for p in peaks_fft:
                f0 = float(p.get("f_hz", 0.0))
                a0 = float(p.get("amp", 0.0))
                if f0 <= 0 or a0 <= 0:
                    continue
                if _near_any(f0):
                    continue
                # Estimar Q con ancho a -3 dB (~0.707*A)
                thr = a0 / np.sqrt(2.0)
                idx0 = int(np.argmin(np.abs(xf - f0)))
                # Buscar izquierda
                iL = idx0
                while iL > 0 and mag_vel_mm[iL] > thr:
                    iL -= 1
                # Buscar derecha
                iR = idx0
                max_idx = len(mag_vel_mm) - 1
                while iR < max_idx and mag_vel_mm[iR] > thr:
                    iR += 1
                if iR > iL and (iR - iL) >= 2:
                    fL = float(xf[max(iL, 0)])
                    fR = float(xf[min(iR, max_idx)])
                    bw = max(fR - fL, df if df > 0 else 1e-6)
                    Q = float(f0 / bw) if bw > 0 else 0.0
                    # Umbrales: pico relevante y Q alto
                    if Q >= 8.0 and a0 >= max(0.2 * (dom_amp + 1e-12), 0.3):
                        resonances.append((f0, Q, a0))
            # Reportar hasta 2 resonancias principales
            resonances.sort(key=lambda x: x[2], reverse=True)
            for f0, Q, a0 in resonances[:2]:
                findings.append(f"Resonancia estructural probable: pico agudo ~{f0:.1f} Hz (Q~{Q:.1f}).")
    except Exception:
        pass
    if len(findings) == 1:
        findings.append("Sin anomalías evidentes según reglas actuales.")
    severity_summary, core_findings = _split_diagnosis(findings)
    runtime_s = float(time.perf_counter() - analysis_start)
    analyzed_at = datetime.utcnow().isoformat()
    return {
        "segment_used": (float(t[0]), float(t[-1])),
        "fs_hz": fs,
        "dt_s": dt,
        "df_hz": df,
        "n": int(len(a)),
        "pre_decimation": predec_info,
        "time": {
            "t_s": t,
            "acc_ms2": a,
            "rms_acc_ms2": rms_time_acc,
            "peak_acc_ms2": peak_acc,
            "pp_acc_ms2": pp_acc,
        },
        "fft": {
            "f_hz": xf,
            "acc_spec_ms2": mag_acc,
            "vel_spec_mm_s": mag_vel_mm,
            "peaks": peaks_fft,
            "dom_freq_hz": dom_freq,
            "dom_amp_mm_s": dom_amp,
            "rms_vel_mm_s": rms_vel_spec_mm,
            "r2x": r2x,
            "r3x": r3x,
            "energy": {"low": e_low, "mid": e_mid, "high": e_high, "total": e_total},
            "window": window_type,
        },
        "envelope": {
            "f_hz": xf_env,
            "amp": env_spec,
            "peaks": peaks_env,
        },
        "rpm": rpm,
        "f1_hz": f1,
        "severity": {"label": sev_label, "color": sev_color, "rms_mm_s": rms_vel_spec_mm},
        "diagnosis": findings,
        "diagnosis_summary": severity_summary,
        "diagnosis_findings": core_findings,
        "ml": {
            "features": ml_features,
            "result": ml_result,
        },
        "iso_ml_conflict": {
            "conflict": bool(conflict_note),
            "note": conflict_note,
            "iso_rank": iso_rank,
            "ml_rank": ml_rank,
        },
        "orders": order_summary,
        "bearing_matches": bearing_matches,
        "charlotte_catalog": [dict(entry) for entry in CHARLOTTE_MOTOR_FAULTS],
        "charlotte_lines": _charlotte_faults_lines(),
        "metadata": {
            "analyzer_version": APP_VERSION,
            "model_version": ml_result.get("model_version"),
            "analyzed_at": analyzed_at,
            "runtime_s": runtime_s,
        },
    }



# =========================

#   Botón de Menú Mejorado

# =========================

class MenuButton(ft.Container):

    def __init__(self, icon_name, tooltip, on_click_handler, data=None, is_dark=False):

        self.is_dark = is_dark

        self.icon = ft.Icon(

            name=icon_name,

            color="#e0e0e0" if is_dark else "#2c3e50",

            size=28

        )

        super().__init__(

            width=65,

            height=65,

            border_radius=15,

            tooltip=tooltip,

            content=self.icon,

            ink=True,

            on_hover=self._on_hover,

            on_click=on_click_handler,

            data=data,

            padding=10,

            animate=ft.Animation(200, "easeOut"),

        )

        self.is_active = False



    def _on_hover(self, e: ft.HoverEvent):

        if not self.is_active:

            if e.data == "true":
                accent = getattr(self, "accent", "#3498db")
                self.bgcolor = ft.Colors.with_opacity(0.15, accent)

                self.scale = 1.05

            else:

                self.bgcolor = "transparent"

                self.scale = 1.0

            if self.page:

                self.update()



    def set_active(self, active: bool, safe=False):

        self.is_active = active

        if active:
            self.bgcolor = getattr(self, "accent", "#3498db")
            
            self.icon.color = "white"

            self.scale = 1.05

        else:

            self.bgcolor = "transparent"

            self.icon.color = "#e0e0e0" if self.is_dark else "#2c3e50"

            self.scale = 1.0

        if not safe and self.page:

            self.update()



    def update_theme(self, is_dark: bool):

        self.is_dark = is_dark

        if not self.is_active:

            self.icon.color = "#e0e0e0" if is_dark else "#2c3e50"

            self.bgcolor = "transparent"

        if self.page:

            self.update()





# =========================

#   Aplicación Principal

# =========================

class MainApp:

    def __init__(self, page: ft.Page):

        print("MainApp inicializada")

        self.page = page

        self.page.title = "Sistema de Análisis de Vibraciones Mecánicas"

        self.page.padding = 0

        

        # Configuración de ventana

        self.page.window.width = 1400

        self.page.window.height = 850

        self.page.window.min_width = 1000

        self.page.window.min_height = 700

        # Capacidades de la plataforma
        self._interactive_notice_logged = False
        self.interactive_charts_enabled = self._detect_interactive_chart_support()

        # Estado con valores por defecto

        self.clock_24h = self._get_bool_storage("clock_24h", True)



        stored_accent = self.page.client_storage.get("accent")
        self.accent = stored_accent if stored_accent is not None else "#3498db"
        self.time_plot_color = self._get_color_pref("time_plot_color", "#00bcd4")
        self.fft_plot_color = self._get_color_pref("fft_plot_color", self._accent_ui())
        self.combine_signals_enabled = self._get_bool_storage("combine_signals_enabled", False)
        self._last_combined_sources: List[str] = []
        try:
            stored_fft_window = self.page.client_storage.get("fft_window_type")
        except Exception:
            stored_fft_window = None
        self.fft_window_type = _resolve_fft_window(stored_fft_window)
        try:
            stored_input_unit = self.page.client_storage.get("input_signal_unit")
        except Exception:
            stored_input_unit = None
        valid_input_units = {
            "acc_ms2",
            "acc_g",
            "vel_ms",
            "vel_mm",
            "vel_ips",
            "disp_m",
            "disp_mm",
            "disp_um",
        }
        self.input_signal_unit = stored_input_unit if stored_input_unit in valid_input_units else "acc_ms2"



        self.is_dark_mode = self._get_bool_storage("is_dark_mode", True)



        self.is_panel_expanded = True  # Add this line after other state variables

        self.is_menu_expanded = True  # Add this line



        self._apply_theme()

        self.last_view = "welcome"

        self.uploaded_files = []

        self.current_df = None
        self._raw_current_df: Optional[pd.DataFrame] = None

        self.file_data_storage = {}  # Almacenar datos de archivos
        self.signal_unit_map: Dict[str, str] = {}
        self._last_axis_severity: List[Dict[str, Any]] = []
        self._last_primary_severity: Optional[Dict[str, Any]] = None
        self._fft_zoom_range: Optional[Tuple[float, float]] = None
        self._fft_full_range: Optional[Tuple[float, float]] = None
        self._fft_zoom_syncing = False
        self._fft_display_scale: float = 1.0
        self._fft_display_unit: str = "Hz"

        # Estado de análisis/rodamientos
        self.analysis_mode = 'auto'            # 'auto' o 'assist'
        self.selected_bearing_model = ''       # modelo seleccionado para preselección en análisis

        # Base de rodamientos (opcional)
        self.bearing_db_items: List[Dict[str, Any]] = []
        self._load_bearing_db()
        # Favoritos de reportes
        self.report_favorites = self._load_report_favorites()
        self.report_show_favs_only = self._get_bool_storage("report_favs_only", False)
        # Favoritos de rodamientos
        self.bearing_favorites = self._load_bearing_favorites()
        self.bearing_show_favs_only = self._get_bool_storage("bearing_favs_only", False)
        # Favoritos de archivos de datos
        self.data_favorites = self._load_data_favorites()
        self.data_show_favs_only = self._get_bool_storage("data_favs_only", False)
        # Preferencias de análisis avanzados
        self.runup_3d_enabled = self._get_bool_storage("runup_3d_enabled", False)
        self.orbit_plot_enabled = self._get_bool_storage("orbit_plot_enabled", False)
        stored_orbit_x = self.page.client_storage.get("orbit_axis_x")
        stored_orbit_y = self.page.client_storage.get("orbit_axis_y")
        self.orbit_axis_x_pref = stored_orbit_x if isinstance(stored_orbit_x, str) else None
        self.orbit_axis_y_pref = stored_orbit_y if isinstance(stored_orbit_y, str) else None
        try:
            stored_orbit_period = self.page.client_storage.get("orbit_period_seconds")
        except Exception:
            stored_orbit_period = None
        self.orbit_period_seconds: Optional[float] = None
        try:
            if stored_orbit_period not in (None, "", "None"):
                candidate = float(stored_orbit_period)
                if np.isfinite(candidate) and candidate > 0:
                    self.orbit_period_seconds = float(candidate)
        except Exception:
            self.orbit_period_seconds = None



        self.file_picker = ft.FilePicker(on_result=self._handle_file_pick_result)
        self.page.overlay.append(self.file_picker)
        # File picker para CSV de rodamientos
        self.bearing_file_picker = ft.FilePicker(on_result=self._bearing_csv_pick_result)
        self.page.overlay.append(self.bearing_file_picker)

        self._pdf_export_running = False
        self.pdf_progress_text = ft.Text(
            "Generando reporte PDF...",
            size=14,
            weight="w500",
            color="white",
        )
        self.pdf_progress_overlay = ft.Container(
            visible=False,
            expand=True,
            alignment=ft.alignment.center,
            bgcolor=ft.Colors.with_opacity(0.45, "black"),
            content=ft.Card(
                elevation=8,
                content=ft.Container(
                    padding=ft.padding.all(20),
                    content=ft.Column(
                        [
                            ft.ProgressRing(color=self._accent_ui(), width=60, height=60),
                            self.pdf_progress_text,
                        ],
                        alignment="center",
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        spacing=14,
                    ),
                ),
            ),
        )
        self.page.overlay.append(self.pdf_progress_overlay)



        self.clock_text = ft.Text(self._get_current_time(), size=14, weight="w500")

        self.files_list_view = ft.ListView(expand=1, spacing=8, auto_scroll=False, padding=10)



        self.menu_buttons = {}

        self.menu = self._build_menu()

        self.control_panel = self._build_control_panel()

        self.main_content_area = ft.Container(

            expand=True, 

            padding=25,

            border_radius=20,

            bgcolor=ft.Colors.with_opacity(0.03, "white" if self.is_dark_mode else "black"),

            margin=10,

            alignment=ft.alignment.center

        )

        

        # Layout principal con diseño mejorado

        self.content = ft.Row(

            expand=True,

            spacing=0,

            controls=[

                self.menu, 

                ft.Container(expand=True, content=self.main_content_area, padding=0),

                self.control_panel

            ],

        )

        

        self.main_content_area.content = self._build_welcome_view()

        self.page.run_task(self._start_clock_timer)



    def _get_current_time(self):

        fmt = "%H:%M:%S" if self.clock_24h else "%I:%M:%S %p"

        return time.strftime(fmt)



    def _apply_theme(self):

        self.page.theme_mode = ft.ThemeMode.DARK if self.is_dark_mode else ft.ThemeMode.LIGHT

        self.page.bgcolor = "#1a1a2e" if self.is_dark_mode else "#f5f5f5"

        self.page.update()

    # ===== Rodamientos: DB + asistentes =====
    def _load_bearing_db(self):
        """Intenta cargar 'bearing_db.csv' del directorio actual.
        Formato esperado de columnas: model,brand?,n,d_mm,D_mm,theta_deg
        """
        try:
            path = os.path.join(os.getcwd(), "bearing_db.csv")
            if os.path.exists(path):
                df = pd.read_csv(path)
                items: List[Dict[str, Any]] = []
                for _, r in df.iterrows():
                    items.append({
                        "model": str(r.get("model", "")).strip(),
                        "brand": (str(r.get("brand", "")).strip() if "brand" in df.columns else None),
                        "n": int(r.get("n", 0)) if pd.notna(r.get("n", None)) else None,
                        "d_mm": float(r.get("d_mm", 0.0)) if pd.notna(r.get("d_mm", None)) else None,
                        "D_mm": float(r.get("D_mm", 0.0)) if pd.notna(r.get("D_mm", None)) else None,
                        "theta_deg": float(r.get("theta_deg", 0.0)) if pd.notna(r.get("theta_deg", None)) else 0.0,
                    })
                # Filtrar modelos no vacíos
                self.bearing_db_items = [it for it in items if it.get("model")] 
            else:
                self.bearing_db_items = []
        except Exception:
            self.bearing_db_items = []

    def _bearing_db_model_options(self) -> List[ft.dropdown.Option]:
        try:
            return [ft.dropdown.Option(it.get("model", "")) for it in (self.bearing_db_items or []) if it.get("model")]
        except Exception:
            return []

    def _on_bearing_model_change(self, e=None):
        try:
            model = getattr(self, "bearing_model_dd", None).value if getattr(self, "bearing_model_dd", None) else None
        except Exception:
            model = None
        if not model:
            return
        try:
            for it in (self.bearing_db_items or []):
                if it.get("model") == model:
                    # Rellenar campos de geometría
                    if getattr(self, "br_n_field", None):
                        self.br_n_field.value = str(it.get("n") or "")
                    if getattr(self, "br_d_mm_field", None):
                        self.br_d_mm_field.value = str(it.get("d_mm") or "")
                    if getattr(self, "br_D_mm_field", None):
                        self.br_D_mm_field.value = str(it.get("D_mm") or "")
                    if getattr(self, "br_theta_deg_field", None):
                        self.br_theta_deg_field.value = str(it.get("theta_deg") or "0")
                    if self.page:
                        try:
                            self.br_n_field.update(); self.br_d_mm_field.update(); self.br_D_mm_field.update(); self.br_theta_deg_field.update()
                        except Exception:
                            pass
                    break
        except Exception:
            pass

    def _on_mode_change(self, e=None):
        mode = None
        try:
            mode = getattr(self, "analysis_mode_dd", None).value if getattr(self, "analysis_mode_dd", None) else None
        except Exception:
            mode = None
        # Persistir estado seleccionado
        try:
            if mode in ("auto", "assist"):
                self.analysis_mode = mode
        except Exception:
            pass
        try:
            if getattr(self, "assisted_box", None) is not None:
                self.assisted_box.visible = (mode == "assist")
                self.assisted_box.update() if self.assisted_box.page else None
        except Exception:
            pass
        # En modo automático, limpiar campos de BPFO/BPFI/BSF/FTF para no condicionar el diagnóstico
        try:
            if mode == "auto":
                for fld_name in ("bpfo_field", "bpfi_field", "bsf_field", "ftf_field"):
                    fld = getattr(self, fld_name, None)
                    if fld:
                        fld.value = ""
                        fld.update() if fld.page else None
        except Exception:
            pass
        # Refrescar análisis
        try:
            self._update_analysis()
        except Exception:
            pass

    def _on_config_tab_change(self, e: ft.ControlEvent):
        key = None
        try:
            tabs_control = e.control
            selected_index = getattr(tabs_control, "selected_index", 0) or 0
            keys = getattr(self, "config_tab_keys", []) or []
            if 0 <= selected_index < len(keys):
                key = keys[selected_index]
        except Exception:
            key = None
        if not key:
            return
        try:
            new_view = (self.config_tab_views or {}).get(key)
        except Exception:
            new_view = None
        if not new_view:
            return
        try:
            self.active_config_tab = key
        except Exception:
            pass
        try:
            self.config_tab_body.content = new_view
            if self.config_tab_body.page:
                self.config_tab_body.update()
        except Exception:
            pass

    def _compute_bearing_freqs_click(self, e=None):
        try:
            rpm_val = float(self.rpm_hint_field.value) if getattr(self, "rpm_hint_field", None) and getattr(self.rpm_hint_field, "value", "") else None
        except Exception:
            rpm_val = None
        def _tf_float(tf):
            try:
                return float(tf.value) if tf and getattr(tf, "value", "") else None
            except Exception:
                return None
        n_val = None
        try:
            n_val = int(float(self.br_n_field.value)) if getattr(self, "br_n_field", None) and getattr(self, "br_n_field").value not in (None, "") else None
        except Exception:
            n_val = None
        d_val = _tf_float(getattr(self, "br_d_mm_field", None))
        D_val = _tf_float(getattr(self, "br_D_mm_field", None))
        th_val = _tf_float(getattr(self, "br_theta_deg_field", None)) or 0.0
        freqs = bearing_freqs_from_geometry(rpm_val, n_val, d_val, D_val, th_val)
        # Rellenar los campos BPFO/BPFI/BSF/FTF si hay valores
        try:
            if freqs.get("bpfo"):
                self.bpfo_field.value = f"{freqs['bpfo']:.3f}"
            if freqs.get("bpfi"):
                self.bpfi_field.value = f"{freqs['bpfi']:.3f}"
            if freqs.get("bsf"):
                self.bsf_field.value = f"{freqs['bsf']:.3f}"
            if freqs.get("ftf"):
                self.ftf_field.value = f"{freqs['ftf']:.3f}"
            for fld in (self.bpfo_field, self.bpfi_field, self.bsf_field, self.ftf_field):
                try:
                    fld.update()
                except Exception:
                    pass
        except Exception:
            pass
        # Refrescar análisis automáticamente
        try:
            self._update_analysis()
        except Exception:
            pass

    def _save_bearing_db(self):
        try:
            path = os.path.join(os.getcwd(), "bearing_db.csv")
            import csv
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["model", "brand", "n", "d_mm", "D_mm", "theta_deg"])
                for it in (self.bearing_db_items or []):
                    writer.writerow([
                        it.get("model", ""),
                        it.get("brand", ""),
                        it.get("n", ""),
                        it.get("d_mm", ""),
                        it.get("D_mm", ""),
                        it.get("theta_deg", 0.0),
                    ])
        except Exception:
            pass

    def _refresh_bearing_list_ui(self):
        try:
            if not getattr(self, "bearing_list_view", None):
                return
            self.bearing_list_view.controls.clear()
            q = ""
            try:
                q = str(getattr(self, 'bearing_search', None).value or "").strip().lower()
            except Exception:
                q = ""
            items = list(self.bearing_db_items or [])
            # Filtrar por marca según pestaña seleccionada
            try:
                sel_brand = None
                if getattr(self, 'bearing_tabs', None) and getattr(self.bearing_tabs, 'tabs', None):
                    idx = int(getattr(self.bearing_tabs, 'selected_index', 0) or 0)
                    tabs = self.bearing_tabs.tabs
                    if 0 <= idx < len(tabs):
                        sel_brand = getattr(tabs[idx], 'text', None)
                if sel_brand and sel_brand not in ("Todos", "Todas", "All"):
                    def _get_brand(it):
                        b = it.get('brand') if isinstance(it, dict) else None
                        if b:
                            return str(b)
                        # intenta inferir de 'model' (prefijo alfabético)
                        m = str(it.get('model',''))
                        for cut in (" ", "-", "_"):
                            if cut in m:
                                return m.split(cut)[0]
                        # letras iniciales
                        prefix = ''.join([ch for ch in m if ch.isalpha()])
                        return prefix or "Otros"
                    items = [it for it in items if _get_brand(it) == sel_brand]
            except Exception:
                pass
            if q:
                def _match(it):
                    try:
                        return (q in str(it.get('model','')).lower()) or (q in str(it.get('n','')).lower())
                    except Exception:
                        return False
                items = [it for it in items if _match(it)]
            # Filtrar por favoritos si está activo
            try:
                if getattr(self, 'bearing_show_favs_only', False):
                    favs = getattr(self, 'bearing_favorites', {}) or {}
                    items = [it for it in items if bool(favs.get(str(it.get('model','')), False))]
            except Exception:
                pass
            for _, it in enumerate(items):
                model = str(it.get("model", ""))
                subtitle = f"n={it.get('n','?')}  d={it.get('d_mm','?')}mm  D={it.get('D_mm','?')}mm  θ={it.get('theta_deg',0)}°"
                # Star favorite icon
                is_fav = False
                try:
                    is_fav = bool(getattr(self, 'bearing_favorites', {}).get(model, False))
                except Exception:
                    is_fav = False
                star_icon = ft.Icons.STAR if is_fav else ft.Icons.STAR_BORDER_ROUNDED
                star_color = "#f1c40f" if is_fav else "#bdc3c7"
                tile = ft.ListTile(
                    leading=ft.IconButton(icon=star_icon, icon_color=star_color, tooltip="Favorito", on_click=lambda e, m=model: self._toggle_bearing_favorite(m)),
                    title=ft.Text(model),
                    subtitle=ft.Text(subtitle),
                    on_click=lambda e, m=model: self._select_bearing_by_model(m),
                    trailing=ft.IconButton(icon=ft.Icons.DELETE_FOREVER_ROUNDED, tooltip="Eliminar", on_click=lambda e, m=model: self._bearing_delete_model(m)),
                    dense=True,
                )
                self.bearing_list_view.controls.append(tile)
            if self.bearing_list_view.page:
                self.bearing_list_view.update()
        except Exception:
            pass

    def _bearing_brand_names(self) -> List[str]:
        try:
            brands = []
            for it in (self.bearing_db_items or []):
                b = None
                try:
                    b = it.get('brand')
                except Exception:
                    b = None
                if not b:
                    m = str(it.get('model',''))
                    for cut in (" ", "-", "_"):
                        if cut in m:
                            b = m.split(cut)[0]
                            break
                    if not b:
                        pref = ''.join([ch for ch in m if ch.isalpha()])
                        b = pref or "Otros"
                brands.append(str(b))
            uniq = sorted({b for b in brands if b})
            return ["Todos"] + uniq
        except Exception:
            return ["Todos"]

    def _rebuild_bearing_tabs(self):
        try:
            names = self._bearing_brand_names()
            if not getattr(self, 'bearing_tabs', None):
                self.bearing_tabs = ft.Tabs(tabs=[ft.Tab(text=n) for n in names], selected_index=0, on_change=self._on_bearing_tab_change)
            else:
                self.bearing_tabs.tabs = [ft.Tab(text=n) for n in names]
                self.bearing_tabs.selected_index = min(getattr(self.bearing_tabs, 'selected_index', 0) or 0, len(names)-1)
                if self.bearing_tabs.page:
                    self.bearing_tabs.update()
        except Exception:
            pass

    def _on_bearing_tab_change(self, e=None):
        try:
            self._refresh_bearing_list_ui()
        except Exception:
            pass

    def _toggle_bearing_favs_filter(self):
        try:
            self.bearing_show_favs_only = bool(getattr(self, 'bearing_favs_only_cb', None).value) if getattr(self, 'bearing_favs_only_cb', None) else False
        except Exception:
            self.bearing_show_favs_only = False
        try:
            self.page.client_storage.set("bearing_favs_only", self.bearing_show_favs_only)
        except Exception:
            pass
        self._refresh_bearing_list_ui()

    # Favoritos de rodamientos
    def _bearing_favorites_path(self) -> str:
        try:
            return os.path.join(os.getcwd(), "bearing_favorites.json")
        except Exception:
            return "bearing_favorites.json"

    def _load_bearing_favorites(self) -> Dict[str, bool]:
        try:
            import json
            path = self._bearing_favorites_path()
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return {str(k): bool(v) for k, v in data.items()}
        except Exception:
            pass
        return {}

    def _save_bearing_favorites(self):
        try:
            import json
            path = self._bearing_favorites_path()
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.bearing_favorites or {}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _toggle_bearing_favorite(self, model: str):
        try:
            m = str(model or "")
            if not m:
                return
            cur = bool((self.bearing_favorites or {}).get(m, False))
            self.bearing_favorites[m] = not cur
            self._save_bearing_favorites()
        except Exception:
            pass
        try:
            self._refresh_bearing_list_ui()
        except Exception:
            pass

    def _select_bearing_from_list(self, idx: int):
        try:
            if idx < 0 or idx >= len(self.bearing_db_items):
                return
            it = self.bearing_db_items[idx]
            self._bearing_sel_index = idx
            # Rellenar panel detalle del diálogo
            if getattr(self, "br_model_field_dlg", None):
                self.br_model_field_dlg.value = str(it.get("model", ""))
            if getattr(self, "br_n_field_dlg", None):
                self.br_n_field_dlg.value = str(it.get("n", ""))
            if getattr(self, "br_d_mm_field_dlg", None):
                self.br_d_mm_field_dlg.value = str(it.get("d_mm", ""))
            if getattr(self, "br_D_mm_field_dlg", None):
                self.br_D_mm_field_dlg.value = str(it.get("D_mm", ""))
            if getattr(self, "br_theta_deg_field_dlg", None):
                self.br_theta_deg_field_dlg.value = str(it.get("theta_deg", 0))
            try:
                self.br_model_field_dlg.update(); self.br_n_field_dlg.update(); self.br_d_mm_field_dlg.update(); self.br_D_mm_field_dlg.update(); self.br_theta_deg_field_dlg.update()
            except Exception:
                pass
        except Exception:
            pass

    def _select_bearing_by_model(self, model: str):
        try:
            target = str(model or "")
            if not target:
                return
            idx = None
            for i, it in enumerate(self.bearing_db_items or []):
                if str(it.get('model','')) == target:
                    idx = i
                    break
            if idx is None:
                return
            self._select_bearing_from_list(idx)
        except Exception:
            pass

    def _bearing_new_click(self, e=None):
        try:
            self._bearing_sel_index = None
            self.br_model_field_dlg.value = ""
            self.br_n_field_dlg.value = ""
            self.br_d_mm_field_dlg.value = ""
            self.br_D_mm_field_dlg.value = ""
            self.br_theta_deg_field_dlg.value = "0"
            for fld in (self.br_model_field_dlg, self.br_n_field_dlg, self.br_d_mm_field_dlg, self.br_D_mm_field_dlg, self.br_theta_deg_field_dlg):
                try:
                    fld.update()
                except Exception:
                    pass
        except Exception:
            pass

    def _bearing_save_click(self, e=None):
        try:
            model = str(getattr(self, "br_model_field_dlg", None).value or "").strip()
            if not model:
                return
            def _to_float(tf):
                try:
                    return float(tf.value) if tf and getattr(tf, 'value', '') else None
                except Exception:
                    return None
            def _to_int(tf):
                try:
                    return int(float(tf.value)) if tf and getattr(tf, 'value', '') else None
                except Exception:
                    return None
            item = {
                "model": model,
                "n": _to_int(getattr(self, "br_n_field_dlg", None)),
                "d_mm": _to_float(getattr(self, "br_d_mm_field_dlg", None)),
                "D_mm": _to_float(getattr(self, "br_D_mm_field_dlg", None)),
                "theta_deg": _to_float(getattr(self, "br_theta_deg_field_dlg", None)) or 0.0,
            }
            # actualizar si el modelo ya existe, si no agregar
            updated = False
            for i, it in enumerate(self.bearing_db_items or []):
                if str(it.get("model", "")) == model:
                    self.bearing_db_items[i] = item
                    updated = True
                    break
            if not updated:
                self.bearing_db_items.append(item)
            # guardar y refrescar UI
            self._save_bearing_db()
            # refrescar options del dropdown
            try:
                self.bearing_model_dd.options = self._bearing_db_model_options()
                self.bearing_model_dd.update()
            except Exception:
                pass
            self._refresh_bearing_list_ui()
        except Exception:
            pass

    def _bearing_use_click(self, e=None):
        try:
            model = str(getattr(self, "br_model_field_dlg", None).value or "").strip()
            if model:
                try:
                    # Guardar modelo seleccionado en estado
                    self.selected_bearing_model = model
                    self.bearing_model_dd.value = model
                    self.bearing_model_dd.update()
                except Exception:
                    pass
            # Transferir a los campos principales de geometría
            for src, dst_name in (
                (getattr(self, "br_n_field_dlg", None), "br_n_field"),
                (getattr(self, "br_d_mm_field_dlg", None), "br_d_mm_field"),
                (getattr(self, "br_D_mm_field_dlg", None), "br_D_mm_field"),
                (getattr(self, "br_theta_deg_field_dlg", None), "br_theta_deg_field"),
            ):
                try:
                    getattr(self, dst_name).value = getattr(src, 'value', '')
                    getattr(self, dst_name).update()
                except Exception:
                    pass
            # cerrar diálogo
            if getattr(self, 'bearing_picker_dlg', None):
                self.bearing_picker_dlg.open = False
                if self.page:
                    self.page.update()
        except Exception:
            pass

    def _bearing_use_and_go(self, e=None):
        try:
            self._bearing_use_click()
            # Navegar a análisis
            try:
                if getattr(self, 'analysis_mode_dd', None):
                    self.analysis_mode_dd.value = 'assist'
                    try:
                        self.analysis_mode_dd.update()
                    except Exception:
                        pass
                    self._on_mode_change()
            except Exception:
                pass
            self._select_menu('analysis', force_rebuild=True)
        except Exception:
            try:
                self._select_menu('analysis', force_rebuild=True)
            except Exception:
                pass

    def _bearing_close_click(self, e=None):
        try:
            if getattr(self, 'bearing_picker_dlg', None):
                self.bearing_picker_dlg.open = False
                if self.page:
                    self.page.update()
        except Exception:
            pass

    def _bearing_delete_click(self, e=None):
        try:
            model = str(getattr(self, "br_model_field_dlg", None).value or "").strip()
            if not model:
                return
            before = len(self.bearing_db_items or [])
            self.bearing_db_items = [it for it in (self.bearing_db_items or []) if str(it.get('model','')) != model]
            after = len(self.bearing_db_items)
            if after < before:
                self._save_bearing_db()
                # Limpiar selección
                for fld in (self.br_model_field_dlg, self.br_n_field_dlg, self.br_d_mm_field_dlg, self.br_D_mm_field_dlg, self.br_theta_deg_field_dlg):
                    try:
                        fld.value = "" if fld is not self.br_theta_deg_field_dlg else "0"
                        fld.update()
                    except Exception:
                        pass
                # Refrescar UI
                try:
                    self.bearing_model_dd.options = self._bearing_db_model_options()
                    self.bearing_model_dd.update()
                except Exception:
                    pass
                self._refresh_bearing_list_ui()
        except Exception:
            pass

    def _bearing_delete_model(self, model: str):
        try:
            model = str(model or "").strip()
            if not model:
                return
            self.bearing_db_items = [it for it in (self.bearing_db_items or []) if str(it.get('model','')) != model]
            self._save_bearing_db()
            # refrescar UI y dropdown
            try:
                self.bearing_model_dd.options = self._bearing_db_model_options()
                self.bearing_model_dd.update()
            except Exception:
                pass
            self._refresh_bearing_list_ui()
        except Exception:
            pass

    # ==== Favoritos de reportes ====
    def _favorites_path(self) -> str:
        try:
            reports_dir = os.path.join(os.getcwd(), "reports")
            os.makedirs(reports_dir, exist_ok=True)
            return os.path.join(reports_dir, "favorites.json")
        except Exception:
            return "favorites.json"

    def _load_report_favorites(self) -> Dict[str, bool]:
        try:
            import json
            path = self._favorites_path()
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return {str(k): bool(v) for k, v in data.items()}
        except Exception:
            pass
        return {}

    def _save_report_favorites(self):
        try:
            import json
            path = self._favorites_path()
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.report_favorites or {}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _toggle_report_favorite(self, path: str):
        try:
            cur = bool((self.report_favorites or {}).get(path, False))
            self.report_favorites[path] = not cur
            self._save_report_favorites()
        except Exception:
            pass
        # Refrescar listado para actualizar icono
        try:
            self._refresh_report_list_scandir()
        except Exception:
            pass

    def _toggle_reports_fav_filter(self):
        try:
            self.report_show_favs_only = bool(getattr(self, 'report_favs_only_cb', None).value)
        except Exception:
            self.report_show_favs_only = False
        try:
            self.page.client_storage.set("report_favs_only", self.report_show_favs_only)
        except Exception:
            pass
        self._refresh_report_list_scandir()

    def _open_bearing_picker(self, e=None):
        try:
            # Crear controles del diálogo si no existen
            if not getattr(self, 'bearing_list_view', None):
                self.bearing_list_view = ft.ListView(expand=True, spacing=4, padding=4, height=400)
            # Panel detalle
            if not getattr(self, 'br_model_field_dlg', None):
                self.br_model_field_dlg = ft.TextField(label="Modelo", width=220)
                self.br_n_field_dlg = ft.TextField(label="# Elementos (n)", width=150)
                self.br_d_mm_field_dlg = ft.TextField(label="d (mm)", width=120)
                self.br_D_mm_field_dlg = ft.TextField(label="D (mm)", width=120)
                self.br_theta_deg_field_dlg = ft.TextField(label="Ángulo (°)", width=120, value="0")
            detail_col = ft.Column([
                ft.Text("Detalle del rodamiento", size=14, weight="bold"),
                self.br_model_field_dlg,
                ft.Row([self.br_n_field_dlg, self.br_d_mm_field_dlg], spacing=10),
                ft.Row([self.br_D_mm_field_dlg, self.br_theta_deg_field_dlg], spacing=10),
                ft.Row([
                    ft.OutlinedButton("Nuevo", icon=ft.Icons.ADD_ROUNDED, on_click=self._bearing_new_click),
                    ft.ElevatedButton("Guardar", icon=ft.Icons.SAVE_ROUNDED, on_click=self._bearing_save_click),
                    ft.OutlinedButton("Eliminar", icon=ft.Icons.DELETE_FOREVER_ROUNDED, on_click=self._bearing_delete_click),
                    ft.ElevatedButton("Usar", icon=ft.Icons.CHECK_CIRCLE_ROUNDED, on_click=self._bearing_use_click),
                ], spacing=10)
            ], spacing=8, width=400)
            list_col = ft.Column([
                ft.Text("Listado de rodamientos", size=14, weight="bold"),
                self.bearing_list_view,
            ], spacing=8, width=350)
            content = ft.Container(
                content=ft.Row([list_col, detail_col], spacing=20),
                padding=10,
                width=800,
            )
            self.bearing_picker_dlg = ft.AlertDialog(
                modal=True,
                title=ft.Text("Rodamientos comunes"),
                content=content,
                actions=[ft.TextButton("Cerrar", on_click=self._bearing_close_click)],
                actions_alignment=ft.MainAxisAlignment.END,
            )
            # Popular listado
            self._refresh_bearing_list_ui()
            # Abrir
            self.page.dialog = self.bearing_picker_dlg
            self.bearing_picker_dlg.open = True
            self.page.update()
        except Exception:
            pass

    # === Importar CSV de rodamientos ===
    def _bearing_open_csv_picker(self, e=None):
        try:
            # Preferir CSV
            try:
                self.bearing_file_picker.pick_files(allow_multiple=False, allowed_extensions=['csv'])
            except Exception:
                self.bearing_file_picker.pick_files(allow_multiple=False)
        except Exception:
            pass

    def _bearing_csv_pick_result(self, e):
        try:
            files = getattr(e, 'files', None)
            if not files:
                return
            path = getattr(files[0], 'path', None)
            if not path or not os.path.exists(path):
                return
            # Leer CSV
            try:
                df = pd.read_csv(path)
            except Exception:
                try:
                    df = pd.read_csv(path, encoding='latin-1')
                except Exception:
                    return
            # Normalizar columnas esperadas
            cols = {c.lower().strip(): c for c in df.columns}
            need = ['model','n','d_mm','D_mm','theta_deg']
            # admitir variantes comunes
            def _getcol(name):
                for k,v in cols.items():
                    if k == name.lower():
                        return v
                return None
            c_model = _getcol('model')
            c_n = _getcol('n')
            c_d = _getcol('d_mm') or _getcol('d')
            c_D = _getcol('D_mm') or _getcol('D')
            c_theta = _getcol('theta_deg') or _getcol('theta') or _getcol('angle')
            c_brand = _getcol('brand')
            if not c_model:
                return
            items = []
            for _, r in df.iterrows():
                try:
                    it = {
                        'model': str(r.get(c_model, '')).strip(),
                        'brand': (str(r.get(c_brand)).strip() if c_brand and pd.notna(r.get(c_brand)) else None),
                        'n': int(r.get(c_n)) if c_n and pd.notna(r.get(c_n)) else None,
                        'd_mm': float(r.get(c_d)) if c_d and pd.notna(r.get(c_d)) else None,
                        'D_mm': float(r.get(c_D)) if c_D and pd.notna(r.get(c_D)) else None,
                        'theta_deg': float(r.get(c_theta)) if c_theta and pd.notna(r.get(c_theta)) else 0.0,
                    }
                except Exception:
                    continue
                if it['model']:
                    items.append(it)
            if not items:
                return
            # Merge por modelo
            by_model = {it['model']: it for it in (self.bearing_db_items or [])}
            for it in items:
                by_model[it['model']] = it
            self.bearing_db_items = list(by_model.values())
            # Guardar y refrescar UI y dropdowns
            self._save_bearing_db()
            try:
                self.bearing_model_dd.options = self._bearing_db_model_options()
                self.bearing_model_dd.update()
            except Exception:
                pass
            # Recrear pestañas por marca
            try:
                self._rebuild_bearing_tabs()
            except Exception:
                pass
            self._refresh_bearing_list_ui()
            # Preseleccionar el primero para mostrar detalle
            try:
                self._select_bearing_by_model(items[0].get('model'))
            except Exception:
                pass
        except Exception:
            pass

    def _bearing_analyze_click(self, e=None):
        try:
            # Copiar al panel asistido principal
            self._bearing_use_click()
            # Calcular frecuencias teóricas
            self._compute_bearing_freqs_click()
            # Ir a Análisis
            self._select_menu('analysis', force_rebuild=True)
        except Exception:
            try:
                self._select_menu('analysis', force_rebuild=True)
            except Exception:
                pass

    # Helpers de configuración/acento
    def _get_bool_storage(self, key: str, default: bool) -> bool:
        try:
            v = self.page.client_storage.get(key)
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(v)
            if isinstance(v, str):
                return v.strip().lower() in ("true", "1", "yes", "on")
        except Exception:
            pass
        return default

    def _sanitize_color(self, value: str | None, fallback: str) -> str:
        try:
            val = str(value or "").strip()
            if not val:
                raise ValueError
            if val.startswith("#"):
                if len(val) == 4:
                    val = "#" + ''.join(ch * 2 for ch in val[1:])
                if len(val) >= 7:
                    return val[:7].lower()
                raise ValueError
            basic = {"black", "white", "red", "green", "blue", "yellow", "cyan", "magenta", "orange", "purple", "grey", "gray"}
            if val.lower() in basic:
                return val.lower()
        except Exception:
            pass
        return fallback

    def _get_color_pref(self, key: str, default: str) -> str:
        try:
            stored = self.page.client_storage.get(key)
        except Exception:
            stored = None
        return self._sanitize_color(stored, default)

    def _detect_interactive_chart_support(self) -> bool:
        """Determina si la plataforma actual puede renderizar gráficas interactivas."""

        try:
            if getattr(self.page, "web", False):
                return True
        except Exception:
            pass

        platform_name = ""
        try:
            platform = getattr(self.page, "platform", None)
            if platform is not None:
                if hasattr(platform, "name"):
                    platform_name = str(platform.name).lower()
                else:
                    platform_name = str(platform).lower()
        except Exception:
            platform_name = ""

        if platform_name in {"android", "ios", "fuchsia", "web"}:
            return True
        return False

    def _build_chart_notice(self) -> ft.Container:
        accent = self._accent_ui()
        text_color = "#f5f5f5" if self.is_dark_mode else "#1f2933"
        background = ft.Colors.with_opacity(0.08, accent)
        return ft.Container(
            content=ft.Row(
                controls=[
                    ft.Icon(ft.Icons.INFO_OUTLINED, color=accent, size=20),
                    ft.Text(
                        "Las gráficas interactivas no están disponibles en esta plataforma. "
                        "Se mostrarán versiones estáticas en su lugar.",
                        size=13,
                        color=text_color,
                        expand=True,
                        selectable=False,
                    ),
                ],
                spacing=10,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=ft.padding.symmetric(horizontal=14, vertical=10),
            border_radius=10,
            bgcolor=background,
        )

    def _wrap_chart_with_notice(self, chart: ft.Control | None) -> ft.Control:
        if chart is None:
            return ft.Text("No fue posible generar la gráfica.")
        if self.interactive_charts_enabled:
            return chart
        return ft.Column(
            controls=[self._build_chart_notice(), chart],
            spacing=12,
            expand=True,
        )

    def _remember_orbit_axis(self, axis: str, value: Optional[str]):
        key = "orbit_axis_x" if str(axis).lower().startswith("x") else "orbit_axis_y"
        if str(axis).lower().startswith("x"):
            self.orbit_axis_x_pref = value
        else:
            self.orbit_axis_y_pref = value
        try:
            storage = getattr(self.page, "client_storage", None)
            if not storage:
                return
            if value:
                storage.set(key, value)
            else:
                remover = getattr(storage, "remove", None)
                if callable(remover):
                    remover(key)
                else:
                    storage.set(key, "")
        except Exception:
            pass

    def _collect_selected_signals(self) -> List[str]:
        try:
            return [
                cb.label
                for cb in getattr(self, "signal_checkboxes", [])
                if getattr(cb, "value", False)
            ]
        except Exception:
            return []

    def _build_combined_signal(self, columns: List[str]) -> Optional[np.ndarray]:
        try:
            if not columns or self.current_df is None:
                return None
            data = self.current_df[columns].apply(pd.to_numeric, errors="coerce")
            arr = np.nan_to_num(data.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
            if arr.ndim != 2 or arr.shape[1] < 2:
                return None
            return np.sqrt(np.sum(np.square(arr), axis=1))
        except Exception:
            return None

    def _build_severity_traffic_light(self, rms_mm: float) -> ft.Container:
        """Crea un semáforo visual según la severidad ISO."""

        try:
            rms_val = float(rms_mm)
        except Exception:
            rms_val = 0.0
        if rms_val <= 2.8:
            active_idx = 0
        elif rms_val <= 4.5:
            active_idx = 1
        else:
            active_idx = 2
        colors = ["#2ecc71", "#f1c40f", "#e74c3c"]
        labels = ["Zona A (Buena)", "Zona B (Vigilancia)", "Zona C/D (Alarma)"]

        lights: List[ft.Control] = []
        for idx, (color, label) in enumerate(zip(colors, labels)):
            is_active = idx == active_idx
            lights.append(
                ft.Column(
                    [
                        ft.Container(
                            width=26,
                            height=26,
                            bgcolor=color if is_active else ft.Colors.with_opacity(0.18, color),
                            border=ft.border.all(2, color if is_active else ft.Colors.with_opacity(0.35, color)),
                            border_radius=30,
                        ),
                        ft.Text(label, size=11, text_align=ft.TextAlign.CENTER, width=90),
                    ],
                    spacing=4,
                    horizontal_alignment="center",
                )
            )
        return ft.Container(
            content=ft.Column(
                [
                    ft.Text("Semáforo ISO", weight="bold", size=13),
                    ft.Row(lights, alignment="spaceAround", expand=True),
                ],
                spacing=8,
            ),
            padding=ft.padding.all(10),
            border_radius=10,
            bgcolor=ft.Colors.with_opacity(0.05, self._accent_ui()),
        )

    def _build_spectral_balance_widget(self, frac_low: float, frac_mid: float, frac_high: float) -> ft.Container:
        """Visualiza la energía baja/media/alta en una barra apilada."""

        safe_total = max(frac_low + frac_mid + frac_high, 1e-9)
        portions = [
            (max(frac_low / safe_total, 0.0), "Baja (0-30 Hz)", "#3498db"),
            (max(frac_mid / safe_total, 0.0), "Media (30-120 Hz)", "#9b59b6"),
            (max(frac_high / safe_total, 0.0), "Alta (>120 Hz)", "#e67e22"),
        ]
        bar_width = 260
        segments: List[ft.Control] = []
        for frac, _, color in portions:
            width_px = max(2.0 if frac > 0 else 0.0, bar_width * frac)
            segments.append(
                ft.Container(
                    width=width_px,
                    height=14,
                    bgcolor=color,
                )
            )
        legend_items: List[ft.Control] = []
        for frac, label, color in portions:
            percent = max(min(frac * 100.0, 100.0), 0.0)
            legend_items.append(
                ft.Row(
                    [
                        ft.Container(width=10, height=10, bgcolor=color, border_radius=3),
                        ft.Text(f"{label}: {percent:.1f}%", size=11),
                    ],
                    spacing=6,
                    alignment="start",
                )
            )
        return ft.Container(
            content=ft.Column(
                [
                    ft.Text("Balance espectral", weight="bold", size=13),
                    ft.Container(
                        content=ft.Row(segments, spacing=0, alignment="start"),
                        width=bar_width,
                        height=18,
                        border_radius=6,
                        bgcolor=ft.Colors.with_opacity(0.08, "#ffffff" if self.is_dark_mode else "#000000"),
                        padding=ft.padding.symmetric(horizontal=2, vertical=2),
                    ),
                    ft.Column(legend_items, spacing=4, tight=True),
                ],
                spacing=8,
            ),
            padding=ft.padding.all(10),
            border_radius=10,
            bgcolor=ft.Colors.with_opacity(0.04, self._accent_ui()),
        )

    def _axis_letter_from_name(self, column: Any) -> Optional[str]:
        name = str(column)
        cl = name.lower()
        if any(token in cl for token in ["ch1", "channel1", "axis x", " eje x"]):
            return "X"
        if any(token in cl for token in ["ch2", "channel2", "axis y", " eje y"]):
            return "Y"
        if any(token in cl for token in ["ch3", "channel3", "axis z", " eje z"]):
            return "Z"
        if ("acc" in cl) or ("acel" in cl):
            if "x" in cl and "y" not in cl and "z" not in cl:
                return "X"
            if "y" in cl and "x" not in cl and "z" not in cl:
                return "Y"
            if "z" in cl and "x" not in cl and "y" not in cl:
                return "Z"
        if re.search(r"[_\-\s]x(?![a-z0-9])", cl) or cl.endswith("x"):
            return "X"
        if re.search(r"[_\-\s]y(?![a-z0-9])", cl) or cl.endswith("y"):
            return "Y"
        if re.search(r"[_\-\s]z(?![a-z0-9])", cl) or cl.endswith("z"):
            return "Z"
        return None

    def _axis_display_name(self, column: Any) -> str:
        letter = self._axis_letter_from_name(column)
        return f"Eje {letter}" if letter else str(column)

    def _get_axis_columns(
        self,
        time_col: str,
        df: Optional[pd.DataFrame] = None,
        unit_map: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        data = df if df is not None else self.current_df
        if data is None:
            return []
        axis_cols: List[str] = []
        units = unit_map if unit_map is not None else (self.signal_unit_map or {})
        for col in data.columns:
            if str(col) == str(time_col):
                continue
            unit = units.get(col) if isinstance(units, dict) else None
            if unit == "acc_ms2":
                axis_cols.append(col)
                continue
            name = str(col).lower()
            if ("acc" in name) or ("acel" in name) or self._axis_letter_from_name(col):
                axis_cols.append(col)
        return axis_cols

    def _compute_axis_severity(
        self,
        time_col: str,
        mask: np.ndarray,
        rpm_val: Optional[float],
        line_val: Optional[float],
        teeth_val: Optional[int],
        pre_decimate_hz: Optional[float],
        bpfo_val: Optional[float],
        bpfi_val: Optional[float],
        bsf_val: Optional[float],
        ftf_val: Optional[float],
        env_lo: Optional[float],
        env_hi: Optional[float],
        df: Optional[pd.DataFrame] = None,
        unit_map: Optional[Dict[str, str]] = None,
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        df_source = df if df is not None else getattr(self, "_raw_current_df", None)
        if df_source is None or getattr(df_source, "empty", True) or (time_col not in getattr(df_source, "columns", [])):
            df_source = self.current_df
        if df_source is None or getattr(df_source, "empty", True):
            self._last_axis_severity = []
            self._last_primary_severity = None
            return [], None
        try:
            t_full = pd.to_numeric(df_source[time_col], errors="coerce").to_numpy(dtype=float)
        except Exception:
            self._last_axis_severity = []
            self._last_primary_severity = None
            return [], None
        if mask.size != t_full.size:
            mask = mask[: t_full.size]
        t_segment_raw = t_full[mask]
        axis_cols = self._get_axis_columns(time_col, df_source, unit_map)
        try:
            print(f"[DEBUG] Unit map utilizado: {unit_map}")
        except Exception:
            pass
        axis_summaries: List[Dict[str, Any]] = []
        rms_values: List[float] = []
        for col in axis_cols:
            try:
                series = pd.to_numeric(df_source[col], errors="coerce")
                values = series.to_numpy(dtype=float)
            except Exception:
                values = np.asarray(df_source[col], dtype=float)
            if values.size != mask.size:
                if values.size:
                    values = values[: mask.size]
                else:
                    continue
            seg_raw = values[mask]
            try:
                t_axis, acc_axis, _, _ = self._prepare_segment_for_analysis(t_segment_raw, seg_raw, col)
            except Exception:
                t_axis, acc_axis = np.asarray([]), np.asarray([])
            if acc_axis.size < 2 or t_axis.size < 2:
                entry = {
                    "name": self._axis_display_name(col),
                    "column": col,
                    "rms_mm_s": 0.0,
                    "iso_label": "Sin datos",
                    "emoji_label": "Sin datos",
                    "color": "#7f8c8d",
                    "is_global": False,
                }
                axis_summaries.append(entry)
                continue
            try:
                res_axis = analyze_vibration(
                    t_axis,
                    acc_axis,
                    rpm=rpm_val,
                    line_freq_hz=line_val,
                    bpfo_hz=bpfo_val,
                    bpfi_hz=bpfi_val,
                    bsf_hz=bsf_val,
                    ftf_hz=ftf_val,
                    gear_teeth=teeth_val,
                    pre_decimate_to_fmax_hz=pre_decimate_hz,
                    env_bp_lo_hz=env_lo,
                    env_bp_hi_hz=env_hi,
                    fft_window=self.fft_window_type,
                )
                axis_rms = float(res_axis.get("severity", {}).get("rms_mm_s", 0.0))
                iso_label = res_axis.get("severity", {}).get("label", "N/D")
                color = res_axis.get("severity", {}).get("color", "#7f8c8d")
            except Exception:
                axis_rms = 0.0
                iso_label = "N/D"
                color = "#7f8c8d"
            emoji_label = self._classify_severity(axis_rms)
            debug_axis_msg = (
                f"[DEBUG] RMS por eje {col}: {axis_rms:.6f} mm/s "
                f"(segmento rows={int(mask.sum())})"
            )
            print(debug_axis_msg)
            entry = {
                "name": self._axis_display_name(col),
                "column": col,
                "rms_mm_s": axis_rms,
                "iso_label": iso_label,
                "emoji_label": emoji_label,
                "color": color,
                "is_global": False,
            }
            axis_summaries.append(entry)
            rms_values.append(axis_rms)
        if rms_values:
            # Calcular el RMS global como media cuadrática de los ejes evaluados
            # evitando que ejecuciones previas influyan en el resultado.
            global_rms = float(np.sqrt(np.mean(np.square(rms_values))))
            global_label, global_color = self._classify_severity_ms(global_rms / 1000.0)
            global_entry = {
                "name": "Global",
                "column": "global",
                "rms_mm_s": global_rms,
                "iso_label": global_label,
                "emoji_label": self._classify_severity(global_rms),
                "color": global_color,
                "is_global": True,
            }
            axis_summaries.insert(0, global_entry)
        primary = None
        if axis_summaries:
            primary = next((entry for entry in axis_summaries if entry.get("is_global")), axis_summaries[0])
        self._last_axis_severity = axis_summaries
        self._last_primary_severity = primary
        return axis_summaries, primary

    # Helpers de lectura segura de campos
    def _fldf(self, fld):
        try:
            return float(fld.value) if fld and getattr(fld, 'value', '') else None
        except Exception:
            return None

    def _tfv(self, tf) -> str:
        try:
            return str(tf.value).strip() if tf and getattr(tf, 'value', '') != '' else ''
        except Exception:
            return ''

    def _accent_ui(self) -> str:
        try:
            a = str(self.accent).strip()
            return a if a else "#3498db"
        except Exception:
            return "#3498db"

    def _accent_hex(self) -> str:
        try:
            a = str(self.accent).strip()
            if a.startswith("#") and (len(a) == 7 or len(a) == 9):
                return a[:7]
        except Exception:
            pass
        return "#3498db"

    def _set_accent(self, hex_color: str):
        try:
            val = str(hex_color or "").strip()
            if not val:
                return
            # Normalizar a #RRGGBB
            if val.startswith("#") and len(val) in (4, 7):
                if len(val) == 4:  # #RGB -> #RRGGBB
                    r, g, b = val[1], val[2], val[3]
                    val = f"#{r}{r}{g}{g}{b}{b}"
            self.accent = val
            self.page.client_storage.set("accent", self.accent)
            # Aplicar cambios
            self._apply_theme()
            self._update_theme_for_all_components()
        except Exception:
            pass

    def _on_accent_swatch_click(self, e):
        try:
            hex_color = getattr(e.control, "data", None)
            if hex_color:
                self._set_accent(hex_color)
        except Exception:
            pass

    def _build_accent_palette(self):
        # Generar paleta: columnas = diferentes H (0..330, paso 30), filas = diferentes V
        cols = 12
        rows = 6
        hues = [i * (360 // cols) for i in range(cols)]
        vals = [1.0, 0.85, 0.7, 0.55, 0.4, 0.25]

        swatches: List[ft.Control] = []
        for v in vals:
            for h in hues:
                r, g, b = colorsys.hsv_to_rgb(h / 360.0, 1.0, v)
                rgb = (int(r * 255), int(g * 255), int(b * 255))
                hexc = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
                swatches.append(
                    ft.Container(
                        width=24,
                        height=24,
                        bgcolor=hexc,
                        border_radius=6,
                        margin=2,
                        data=hexc,
                        on_click=self._on_accent_swatch_click,
                        tooltip=hexc,
                        border=ft.border.all(1, "#00000033"),
                    )
                )

        # Añadir escala de grises
        for i in range(12):
            g = int(255 * (i / 11))
            hexc = f"#{g:02x}{g:02x}{g:02x}"
            swatches.append(
                ft.Container(
                    width=24,
                    height=24,
                    bgcolor=hexc,
                    border_radius=6,
                    margin=2,
                    data=hexc,
                    on_click=self._on_accent_swatch_click,
                    tooltip=hexc,
                    border=ft.border.all(1, "#00000033"),
                )
            )

        # Compatibilidad: algunas versiones no tienen ft.Wrap; componemos filas fijas
        rows: List[ft.Control] = []
        row_size = 12
        for i in range(0, len(swatches), row_size):
            rows.append(ft.Row(controls=swatches[i : i + row_size], spacing=2))

        return ft.Column(
            controls=[
                ft.Text("Elige un color (gradiente)", size=12, color="#7f8c8d"),
                ft.Column(controls=rows, spacing=2),
            ]
        )

    def exportar_pdf(self, e=None):
        if getattr(self, "_pdf_export_running", False):
            try:
                self._log("Ya hay una exportaci��n de PDF en curso.")
            except Exception:
                pass
            try:
                self.page.snack_bar = ft.SnackBar(content=ft.Text("Exportaci��n de PDF en curso"), action="OK")
                self.page.snack_bar.open = True
                self.page.update()
            except Exception:
                pass
            return

        self._pdf_export_running = True
        prev_style = plt.rcParams.copy()
        self._set_pdf_progress(True, "Generando reporte PDF...")
        try:
            if self.current_df is None or getattr(self.current_df, 'empty', False):
                self._log("No hay datos para exportar")
                return

            time_col = getattr(self.time_dropdown, "value", None)
            fft_signal_col = getattr(self.fft_dropdown, "value", None)
            if not time_col or not fft_signal_col:
                self._log("Selecciona columnas de tiempo y señal antes de exportar.")
                return
            if time_col not in self.current_df.columns or fft_signal_col not in self.current_df.columns:
                self._log("Las columnas seleccionadas no existen en el DataFrame.")
                return

            reports_dir = os.path.join(os.getcwd(), "reports")
            os.makedirs(reports_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            base_name = os.path.splitext(os.path.basename(self.uploaded_files[0]))[0] if getattr(self, "uploaded_files", None) else "sin_nombre"
            pdf_name = f"{timestamp}_{base_name}.pdf"
            pdf_path = os.path.join(reports_dir, pdf_name)

            plt.style.use("seaborn-v0_8-whitegrid")
            plt.rcParams["font.family"] = "DejaVu Sans"

            t = self.current_df[time_col].to_numpy()
            signal = self.current_df[fft_signal_col].to_numpy()

            mask, start_t, end_t = self._resolve_analysis_period(t)
            if mask.size == 0 or np.count_nonzero(mask) < 2:
                self._log("Periodo seleccionado inválido; se utilizará el rango completo.")
                mask = np.isfinite(np.asarray(t, dtype=float))
                start_t = float(np.nanmin(t[mask])) if np.any(mask) else 0.0
                end_t = float(np.nanmax(t[mask])) if np.any(mask) else 0.0
            segment_idx = np.nonzero(mask)[0]
            t_seg_raw = t[segment_idx]
            sig_seg_raw = signal[segment_idx]
            segment_df = self.current_df.iloc[segment_idx]
            t_seg, acc_seg, _, _ = self._prepare_segment_for_analysis(t_seg_raw, sig_seg_raw, fft_signal_col)

            xf, mag_vel_mm, mag_vel = self._compute_fft_dual(acc_seg, t_seg)
            vel_time_mm = self._acc_to_vel_time_mm(acc_seg, t_seg)
            if vel_time_mm.size:
                rms_mm = float(np.sqrt(np.mean(vel_time_mm**2)))
                rms_ms = rms_mm / 1000.0
            else:
                rms_mm = 0.0
                rms_ms = 0.0
            severity_mm = self._classify_severity(rms_mm)

            if xf is not None:
                features_full = self._extract_features(t_seg, acc_seg, xf, mag_vel_mm)
            else:
                features_full = {"dom_freq": 0.0, "crest": 0.0, "rms_time_acc": 0.0, "peak_acc": 0.0, "pp_acc": 0.0,
                                 "e_low": 0.0, "e_mid": 0.0, "e_high": 0.0, "e_total": 1e-12, "r2x": 0.0, "r3x": 0.0}

            findings_pdf = self._diagnose(features_full) if xf is not None else ["Sin espectro válido para diagnóstico."]

            # Unificar cálculo con analizador (RMS de velocidad correcto)
            try:
                rpm_val = None
                if getattr(self, "rpm_hint_field", None) and getattr(self.rpm_hint_field, "value", ""):
                    rpm_val = float(self.rpm_hint_field.value)
            except Exception:
                rpm_val = None
            try:
                line_val = float(self.line_freq_dd.value) if getattr(self, "line_freq_dd", None) and getattr(self.line_freq_dd, "value", "") else None
            except Exception:
                line_val = None
            try:
                teeth_val = int(self.gear_teeth_field.value) if getattr(self, "gear_teeth_field", None) and getattr(self.gear_teeth_field, "value", "") else None
            except Exception:
                teeth_val = None
            # usando self._fldf para leer campos numéricos opcionales
            # Pre-decimación opcional basada en Máx FFT (Hz)
            try:
                _fmax_pre = float(self.hf_limit_field.value) if getattr(self, 'hf_limit_field', None) and getattr(self.hf_limit_field, 'value', '') else None
            except Exception:
                _fmax_pre = None
            bpfo_val = self._fldf(getattr(self, 'bpfo_field', None))
            bpfi_val = self._fldf(getattr(self, 'bpfi_field', None))
            bsf_val = self._fldf(getattr(self, 'bsf_field', None))
            ftf_val = self._fldf(getattr(self, 'ftf_field', None))
            env_lo_val = self._fldf(getattr(self, 'env_bp_lo_field', None))
            env_hi_val = self._fldf(getattr(self, 'env_bp_hi_field', None))
            self._reset_runtime_analysis_state(announce=False)
            res = analyze_vibration(
                t_seg,
                acc_seg,
                rpm=rpm_val,
                line_freq_hz=line_val,
                bpfo_hz=bpfo_val,
                bpfi_hz=bpfi_val,
                bsf_hz=bsf_val,
                ftf_hz=ftf_val,
                gear_teeth=teeth_val,
                pre_decimate_to_fmax_hz=_fmax_pre,
                env_bp_lo_hz=env_lo_val,
                env_bp_hi_hz=env_hi_val,
                fft_window=self.fft_window_type,
            )
            ml_bundle_pdf = res.get('ml', {})
            ml_result_pdf = (ml_bundle_pdf or {}).get('result') or {}
            xf = res['fft']['f_hz']
            mag_vel_mm = res['fft']['vel_spec_mm_s']
            selected_rms_mm = res['severity']['rms_mm_s']
            selected_label = res['severity']['label']
            selected_color = res['severity']['color']
            metadata_pdf = dict(res.get('metadata') or {})
            conflict_pdf = dict(res.get('iso_ml_conflict') or {})
            if axis_summaries_pdf:
                synced = False
                for entry in axis_summaries_pdf:
                    if entry.get("is_global"):
                        entry.update(
                            {
                                "rms_mm_s": selected_rms_mm,
                                "iso_label": selected_label,
                                "emoji_label": self._classify_severity(selected_rms_mm),
                                "color": selected_color,
                            }
                        )
                        synced = True
                        break
                if not synced:
                    axis_summaries_pdf.insert(
                        0,
                        {
                            "name": "Global",
                            "column": "global",
                            "rms_mm_s": selected_rms_mm,
                            "iso_label": selected_label,
                            "emoji_label": self._classify_severity(selected_rms_mm),
                            "color": selected_color,
                            "is_global": True,
                        },
                    )

            unit_map_local: Dict[str, str] = {}
            if isinstance(self.signal_unit_map, dict) and self.signal_unit_map:
                unit_map_local = dict(self.signal_unit_map)
            else:
                time_key_pdf = str(time_col)
                for col in self.current_df.columns:
                    col_str = str(col)
                    if col_str == time_key_pdf:
                        continue
                    name = col_str.lower()
                    if "acc" in name or "acel" in name:
                        unit_map_local[col_str] = "acc_ms2"
                    elif "vel" in name:
                        unit_map_local[col_str] = "vel_mm" if "mm" in name else "vel_ms"
                    elif "disp" in name or "despl" in name:
                        unit_map_local[col_str] = "disp_mm"
            if not unit_map_local and isinstance(self.signal_unit_map, dict):
                unit_map_local = dict(self.signal_unit_map)
            debug_msg = (
                f"[DEBUG] Análisis principal {fft_signal_col}: RMS={selected_rms_mm:.6f} mm/s, "
                f"segmento={start_t:.6f}-{end_t:.6f}s, muestras={int(mask.sum())}, "
                f"unidad entrada={getattr(self, 'input_signal_unit', 'N/D')}"
            )
            print(debug_msg)
            try:
                self._log(debug_msg)
            except Exception:
                pass
            if unit_map_local:
                try:
                    print(f"[DEBUG] Unidades activas: {unit_map_local}")
                except Exception:
                    pass
            axis_summaries_pdf, primary_entry_pdf = self._compute_axis_severity(
                time_col,
                mask,
                rpm_val,
                line_val,
                teeth_val,
                _fmax_pre,
                bpfo_val,
                bpfi_val,
                bsf_val,
                ftf_val,
                env_lo_val,
                env_hi_val,
                df=self.current_df,
                unit_map=unit_map_local,
            )
            if primary_entry_pdf is None:
                primary_entry_pdf = {
                    "name": self._axis_display_name(fft_signal_col),
                    "column": fft_signal_col,
                    "rms_mm_s": selected_rms_mm,
                    "iso_label": selected_label,
                    "emoji_label": self._classify_severity(selected_rms_mm),
                    "color": selected_color,
                    "is_global": False,
                }
                self._last_primary_severity = primary_entry_pdf
                if not getattr(self, "_last_axis_severity", []):
                    self._last_axis_severity = [primary_entry_pdf]
            else:
                self._last_primary_severity = primary_entry_pdf
            primary_rms_mm_pdf = float(primary_entry_pdf.get("rms_mm_s", selected_rms_mm))
            primary_label_pdf = primary_entry_pdf.get("iso_label", selected_label)
            primary_color_pdf = primary_entry_pdf.get("color", selected_color)
            features_full = self._extract_features(t_seg, acc_seg, xf, mag_vel_mm) if xf is not None else {
                "dom_freq": 0.0, "crest": 0.0, "rms_time_acc": 0.0, "peak_acc": 0.0, "pp_acc": 0.0,
                "e_low": 0.0, "e_mid": 0.0, "e_high": 0.0, "e_total": 1e-12, "r2x": 0.0, "r3x": 0.0
            }
            try:
                features_full["rms_vel_spec"] = float(primary_rms_mm_pdf)
            except Exception:
                pass
            self._last_xf = xf
            self._last_spec = mag_vel_mm
            self._last_tseg = t_seg
            self._last_accseg = acc_seg
            findings_pdf = res.get('diagnosis', []) if xf is not None else ["Sin espectro valido para diagnostico."]
            severity_entry_pdf = res.get('diagnosis_summary')
            findings_core_pdf = list(res.get('diagnosis_findings', []) or [])
            if not findings_core_pdf and findings_pdf:
                _, findings_core_pdf = _split_diagnosis(findings_pdf)
            charlotte_catalog_pdf = list(res.get('charlotte_catalog', []) or [])
            if not charlotte_catalog_pdf:
                charlotte_catalog_pdf = [dict(entry) for entry in CHARLOTTE_MOTOR_FAULTS]
            severity_mm = self._classify_severity(primary_rms_mm_pdf)
            rms_mm = primary_rms_mm_pdf
            axis_table_rows: List[List[str]] = []
            if axis_summaries_pdf:
                axis_table_rows.append(["Canal", "RMS (mm/s)", "Clasificación ISO"])
                for entry in axis_summaries_pdf:
                    try:
                        rms_txt = f"{float(entry.get('rms_mm_s', 0.0)):.3f}"
                    except Exception:
                        rms_txt = "N/D"
                    axis_table_rows.append([
                        entry.get("name", "Eje"),
                        rms_txt,
                        entry.get("iso_label", "N/D"),
                    ])

            try:
                unit_mode = getattr(self.time_unit_dd, 'value', 'vel_mm')
            except Exception:
                unit_mode = 'vel_mm'
            if unit_mode == 'vel_mm':
                _y_time = self._acc_to_vel_time_mm(acc_seg, t_seg)
                _ylabel = 'Velocidad [mm/s]'
                _rms_text = (
                    f"RMS velocidad: {self._calculate_rms(_y_time):.3f} mm/s"
                    if _y_time.size
                    else "RMS velocidad: 0.000 mm/s"
                )
            elif unit_mode == 'acc_g':
                _y_time = acc_seg / 9.80665
                _ylabel = 'Aceleración [g]'
                _rms_text = f"RMS acc: {self._calculate_rms(_y_time):.3f} g"
            else:
                _y_time = acc_seg
                _ylabel = 'Aceleración [m/s²]'
                _rms_text = f"RMS acc: {self._calculate_rms(_y_time):.3e} m/s^2"

            tmp_imgs: List[str] = []

            fig1, ax1 = plt.subplots(figsize=(8, 3))
            if len(t_seg) > 0 and len(_y_time) > 0:
                ax1.plot(t_seg, _y_time, color=self.time_plot_color)
            ax1.set_title(f"Señal {fft_signal_col} ({start_t:.2f}-{end_t:.2f}s)")
            ax1.set_xlabel("Tiempo (s)")
            ax1.set_ylabel(_ylabel)

            # Anotar RMS conforme a la unidad seleccionada
            try:
                text_color = "white" if self.is_dark_mode else "black"
                ax1.text(0.02, 0.95, _rms_text, transform=ax1.transAxes, va="top", color=text_color)
            except Exception:
                pass
            img_time = self._save_temp_plot(fig1, tmp_imgs)

            try:
                fc = float(self.lf_cutoff_field.value) if getattr(self, 'lf_cutoff_field', None) and getattr(self.lf_cutoff_field, 'value', '') else 0.5
            except Exception:
                fc = 0.5
            try:
                hide_lf = bool(getattr(self, 'hide_lf_cb', None).value)
            except Exception:
                hide_lf = True
            try:
                fmax_ui = float(self.hf_limit_field.value) if getattr(self, 'hf_limit_field', None) and getattr(self.hf_limit_field, 'value', '') else None
            except Exception:
                fmax_ui = None
            zoom_range = getattr(self, "_fft_zoom_range", None)
            zmin = zmax = None
            if zoom_range and len(zoom_range) == 2 and zoom_range[1] > zoom_range[0]:
                try:
                    zmin, zmax = float(zoom_range[0]), float(zoom_range[1])
                except Exception:
                    zmin = zmax = None

            top_peaks = []
            fig2, ax2 = plt.subplots(figsize=(8, 3))
            freq_scale = getattr(self, "_fft_display_scale", 1.0) or 1.0
            if not np.isfinite(freq_scale) or freq_scale <= 0:
                freq_scale = 1.0
            freq_unit = getattr(self, "_fft_display_unit", "Hz") or "Hz"

            if xf is not None and mag_vel_mm is not None:
                mask_vis = np.ones_like(xf, dtype=bool)
                if hide_lf:
                    mask_vis &= xf >= max(0.0, fc)
                if zmin is not None:
                    mask_vis &= (xf >= zmin) & (xf <= zmax)
                xpdf = xf[mask_vis]
                ypdf = mag_vel_mm[mask_vis]
                if xpdf.size == 0:
                    xpdf = xf
                    ypdf = mag_vel_mm
                xpdf_disp = np.asarray(xpdf, dtype=float) / freq_scale if xpdf is not None else xpdf
                ax2.plot(xpdf_disp, ypdf, color=self.fft_plot_color, linewidth=1.6)
                try:
                    K = 5
                    min_freq = (max(0.5, fc) if hide_lf else 0.5)
                    mask = xf >= min_freq
                    if zmin is not None:
                        mask &= (xf >= zmin) & (xf <= zmax)
                    xv = xf[mask]
                    yv = mag_vel_mm[mask]
                    if len(yv) > 0:
                        k = min(K, len(yv))
                        idx = np.argpartition(yv, -k)[-k:]
                        idx = idx[np.argsort(yv[idx])[::-1]]
                        peak_f = xv[idx]
                        peak_a = yv[idx]
                        ax2.scatter(peak_f / freq_scale, peak_a, color="#e74c3c", s=20, zorder=5)
                        f1 = self._get_1x_hz(features_full.get("dom_freq", 0.0))
                        peak_points: List[Tuple[float, float]] = []
                        peak_labels: List[str] = []
                        for pf, pa in zip(peak_f, peak_a):
                            try:
                                pf_f = float(pf)
                                pa_f = float(pa)
                            except Exception:
                                continue
                            order = None
                            if f1 and f1 > 0:
                                try:
                                    order = pf_f / float(f1)
                                except Exception:
                                    order = None
                            top_peaks.append((pf_f, pa_f, order))
                            peak_points.append((pf_f / freq_scale, pa_f))
                            peak_labels.append(self._format_peak_label(pf_f, pa_f, order))
                        if peak_points:
                            self._place_annotations(ax2, peak_points, peak_labels, color="#e74c3c")
                except Exception:
                    pass
                try:
                    bpfo = self._fldf(getattr(self, 'bpfo_field', None))
                    bpfi = self._fldf(getattr(self, 'bpfi_field', None))
                    bsf  = self._fldf(getattr(self, 'bsf_field', None))
                    ftf  = self._fldf(getattr(self, 'ftf_field', None))
                    marks_raw = [
                        (bpfo, 'BPFO', '#1f77b4'),
                        (bpfi, 'BPFI', '#ff7f0e'),
                        (bsf,  'BSF',  '#2ca02c'),
                        (ftf,  'FTF',  '#9467bd'),
                    ]
                    visible_marks = []
                    for f0, label, col in marks_raw:
                        if not (f0 and f0 > 0):
                            continue
                        try:
                            f0_f = float(f0)
                        except Exception:
                            continue
                        if zmin is not None and (f0_f < zmin or f0_f > zmax):
                            continue
                        try:
                            ax2.axvline(f0_f / freq_scale, color=col, linestyle='--', alpha=0.8, linewidth=1.2)
                        except Exception:
                            pass
                        visible_marks.append((f0_f / freq_scale, label, col))
                    zoom_scaled = None if zmin is None else (zmin / freq_scale, zmax / freq_scale)
                    self._draw_frequency_markers(ax2, visible_marks, zoom_scaled)
                except Exception:
                    pass
            ax2.set_title(f"FFT (Velocidad)")
            ax2.set_xlabel(f"Frecuencia ({freq_unit})")
            ax2.set_ylabel("Velocidad [mm/s]")
            try:
                ax2_rpm = ax2.twiny()
                xmin, xmax = ax2.get_xlim()
                ax2_rpm.set_xlim(xmin * freq_scale * 60.0, xmax * freq_scale * 60.0)
                ax2_rpm.set_xlabel("Frecuencia (RPM)")
            except Exception:
                pass
            img_fft = self._save_temp_plot(fig2, tmp_imgs)

            # Espectro de Envolvente (gráfica separada para PDF)
            img_env = None
            env_visible_peaks: List[Tuple[float, float, float]] = []
            try:
                xf_env = res.get('envelope', {}).get('f_hz', None)
                env_amp = res.get('envelope', {}).get('amp', None)
                peaks_env = res.get('envelope', {}).get('peaks', [])
                if xf_env is not None and env_amp is not None and len(xf_env) > 0:
                    if hide_lf:
                        m_env = xf_env >= max(0.0, fc)
                    else:
                        m_env = np.ones_like(xf_env, dtype=bool)
                    if fmax_ui and fmax_ui > 0:
                        m_env = m_env & (xf_env <= fmax_ui)
                    if zmin is not None:
                        m_env = m_env & (xf_env >= zmin) & (xf_env <= zmax)
                    xenv = xf_env[m_env]
                    yenv = env_amp[m_env]
                    env_fig, env_ax = plt.subplots(figsize=(8, 3))
                    xenv_disp = np.asarray(xenv, dtype=float) / freq_scale if xenv is not None else xenv
                    env_ax.plot(xenv_disp, yenv, color="#e67e22", linewidth=1.6)
                    env_ax.set_title("Espectro de Envolvente")
                    env_ax.set_xlabel(f"Frecuencia ({freq_unit})")
                    env_ax.set_ylabel("Amp [a.u.]")
                    try:
                        vis_peaks: List[Tuple[float, float, float]] = []
                        for p in (peaks_env or []):
                            f0 = float(p.get('f_hz', 0.0))
                            a0 = float(p.get('amp', 0.0))
                            snr = float(p.get('snr_db', 0.0))
                            if f0 <= 0 or a0 <= 0:
                                continue
                            if hide_lf and f0 < max(0.0, fc):
                                continue
                            if fmax_ui and fmax_ui > 0 and f0 > fmax_ui:
                                continue
                            if zmin is not None and (f0 < zmin or f0 > zmax):
                                continue
                            vis_peaks.append((f0, a0, snr))
                        if vis_peaks:
                            filtered = vis_peaks
                            if zmin is not None:
                                tmp = [(f0, a0, snr) for f0, a0, snr in vis_peaks if zmin <= f0 <= zmax]
                                if tmp:
                                    filtered = tmp
                            env_visible_peaks = [(float(f0), float(a0), float(snr)) for f0, a0, snr in filtered]
                            pfx, pfy = zip(*[(f0, a0) for f0, a0, _ in env_visible_peaks])
                            pfx_disp = [float(f0) / freq_scale for f0 in pfx]
                            env_ax.scatter(pfx_disp, pfy, color="#c0392b", s=36, zorder=5, edgecolors="white", linewidths=0.6)
                            peak_points = [(float(f0) / freq_scale, float(a0)) for f0, a0, _ in env_visible_peaks]
                            peak_labels = [f"{float(f0) / freq_scale:.2f} {freq_unit} | {a0:.3f} a.u." for f0, a0, _ in env_visible_peaks]
                            self._place_annotations(env_ax, peak_points, peak_labels, color="#c0392b", text_color="#c0392b")
                    except Exception:
                        pass
                    try:
                        bpfo = self._fldf(getattr(self, 'bpfo_field', None))
                        bpfi = self._fldf(getattr(self, 'bpfi_field', None))
                        bsf  = self._fldf(getattr(self, 'bsf_field', None))
                        ftf  = self._fldf(getattr(self, 'ftf_field', None))
                        marks_raw = [
                            (bpfo, 'BPFO', '#1f77b4'),
                            (bpfi, 'BPFI', '#ff7f0e'),
                            (bsf,  'BSF',  '#2ca02c'),
                            (ftf,  'FTF',  '#9467bd'),
                        ]
                        visible_marks = []
                        for f0, label, col in marks_raw:
                            if not (f0 and f0 > 0):
                                continue
                            try:
                                f0_f = float(f0)
                            except Exception:
                                continue
                            if zmin is not None and (f0_f < zmin or f0_f > zmax):
                                continue
                            env_ax.axvline(f0_f / freq_scale, color=col, linestyle='--', alpha=0.85, linewidth=1.2)
                            visible_marks.append((f0_f / freq_scale, label, col))
                        zoom_scaled_env = None if zmin is None else (zmin / freq_scale, zmax / freq_scale)
                        self._draw_frequency_markers(env_ax, visible_marks, zoom_scaled_env)
                    except Exception:
                        pass
                    img_env = self._save_temp_plot(env_fig, tmp_imgs)
            except Exception:
                img_env = None
                env_visible_peaks = []

            img_runup = None
            try:
                runup_enabled = False
                if getattr(self, 'runup_3d_cb', None):
                    runup_enabled = bool(getattr(self.runup_3d_cb, 'value', False))
                else:
                    runup_enabled = bool(getattr(self, 'runup_3d_enabled', False))
                if runup_enabled:
                    zoom_tuple = (zmin, zmax) if zmin is not None else None
                    try:
                        full_t_uniform, full_acc_uniform, _, _ = self._prepare_segment_for_analysis(t, signal, fft_signal_col)
                    except Exception:
                        full_t_uniform, full_acc_uniform = None, None
                    base_t = full_t_uniform if full_t_uniform is not None and full_acc_uniform is not None else t_seg
                    base_acc = full_acc_uniform if full_t_uniform is not None and full_acc_uniform is not None else acc_seg
                    try:
                        runup_mask, _, _ = self._resolve_runup_period(base_t)
                    except Exception:
                        runup_mask = None
                    if (
                        runup_mask is not None
                        and runup_mask.size == base_t.size
                        and np.count_nonzero(runup_mask) >= 2
                    ):
                        runup_t = base_t[runup_mask]
                        runup_acc = base_acc[runup_mask]
                    else:
                        runup_t = base_t
                        runup_acc = base_acc
                    runup_fig = self._generate_runup_3d_figure(
                        runup_t,
                        runup_acc,
                        fc,
                        hide_lf,
                        fmax_ui,
                        zoom_tuple,
                        False,
                        base_t,
                        base_acc,
                        self.fft_window_type,
                    )
                    if runup_fig is not None:
                        img_runup = self._save_temp_plot(runup_fig, tmp_imgs)
            except Exception:
                img_runup = None

            img_orbit = None
            try:
                orbit_enabled = False
                if getattr(self, 'orbit_cb', None):
                    orbit_enabled = bool(getattr(self.orbit_cb, 'value', False))
                else:
                    orbit_enabled = bool(getattr(self, 'orbit_plot_enabled', False))
                if orbit_enabled:
                    x_col = getattr(self, 'orbit_x_dd', None).value if getattr(self, 'orbit_x_dd', None) else self.orbit_axis_x_pref
                    y_col = getattr(self, 'orbit_y_dd', None).value if getattr(self, 'orbit_y_dd', None) else self.orbit_axis_y_pref
                    if x_col and y_col and x_col in self.current_df.columns and y_col in self.current_df.columns:
                        try:
                            x_seg_pdf = segment_df[x_col].to_numpy()
                            y_seg_pdf = segment_df[y_col].to_numpy()
                        except Exception:
                            x_seg_pdf = self.current_df[x_col].to_numpy()
                            y_seg_pdf = self.current_df[y_col].to_numpy()
                        t_pdf_orbit, x_pdf_orbit, y_pdf_orbit = self._trim_orbit_window(t_seg, x_seg_pdf, y_seg_pdf)
                        orbit_fig = self._generate_orbit_figure(
                            t_pdf_orbit,
                            x_pdf_orbit,
                            y_pdf_orbit,
                            x_col,
                            y_col,
                            fc,
                            hide_lf,
                            fmax_ui,
                            False,
                        )
                        if orbit_fig is not None:
                            img_orbit = self._save_temp_plot(orbit_fig, tmp_imgs)
            except Exception:
                img_orbit = None

            aux_imgs = []
            aux_selected = []
            try:
                aux_selected = [
                    (cb.label, color_dd.value, style_dd.value)
                    for cb, color_dd, style_dd in getattr(self, "aux_controls", [])
                    if getattr(cb, "value", False)
                ]
            except Exception:
                aux_selected = []
            for col, color, style in aux_selected:
                if col in self.current_df.columns:
                    aux_fig, aux_ax = plt.subplots(figsize=(8, 2))
                    aux_ax.plot(self.current_df[time_col], self.current_df[col], color=color, linestyle=style, linewidth=2, label=col)
                    aux_ax.set_title(f"{col} vs Tiempo")
                    aux_ax.legend()
                    aux_ax.set_xlabel("Tiempo (s)")
                    aux_ax.set_ylabel(col)
                    aux_imgs.append(self._save_temp_plot(aux_fig, tmp_imgs))

            doc = SimpleDocTemplate(pdf_path, pagesize=A4)
            styles = getSampleStyleSheet()
            try:
                accent_hex = self._accent_hex()
            except Exception:
                accent_hex = "#1f77b4"
            try:
                accent_color = colors.HexColor(accent_hex)
            except Exception:
                accent_color = colors.HexColor("#1f77b4")
            title_style = ParagraphStyle(
                "title",
                parent=styles['Title'],
                textColor=accent_color,
                spaceAfter=12,
            )
            styles.add(
                ParagraphStyle(
                    "HeadingAccent",
                    parent=styles['Heading1'],
                    textColor=colors.HexColor("#2c3e50"),
                    spaceAfter=8,
                )
            )
            styles.add(
                ParagraphStyle(
                    "SectionHeading",
                    parent=styles['Heading2'],
                    textColor=colors.HexColor("#2c3e50"),
                    spaceBefore=12,
                    spaceAfter=6,
                )
            )
            styles.add(
                ParagraphStyle(
                    "BulletItem",
                    parent=styles['Normal'],
                    leftIndent=18,
                    spaceAfter=4,
                )
            )
            styles.add(
                ParagraphStyle(
                    "Muted",
                    parent=styles['Normal'],
                    textColor=colors.HexColor("#7f8c8d"),
                    fontSize=10,
                )
            )
            styles.add(
                ParagraphStyle(
                    "Chip",
                    parent=styles['Normal'],
                    textColor=colors.HexColor("#2c3e50"),
                    backColor=colors.HexColor("#e6e6e6"),
                    alignment=1,
                    fontSize=12,
                    leading=14,
                    borderPadding=(4, 8, 4, 8),
                )
            )
            def _build_severity_semaphore(label: str) -> Optional[Table]:
                try:
                    def _sev_index(lbl: str) -> int:
                        s = (lbl or "").lower()
                        if "buena" in s:
                            return 0
                        if "satisfact" in s:
                            return 1
                        if "insatisfact" in s:
                            return 2
                        if "inaceptable" in s:
                            return 3
                        return -1

                    palette = [
                        ("Buena", "#2ecc71"),
                        ("Satisfactoria", "#f1c40f"),
                        ("Insatisfactoria", "#e67e22"),
                        ("Inaceptable", "#e74c3c"),
                    ]
                    idx = _sev_index(label)

                    def _lighten_hex(hx: str, factor: float):
                        try:
                            hx = hx.lstrip('#')
                            r = int(hx[0:2], 16) / 255.0
                            g = int(hx[2:4], 16) / 255.0
                            b = int(hx[4:6], 16) / 255.0
                            r = r + (1.0 - r) * factor
                            g = g + (1.0 - g) * factor
                            b = b + (1.0 - b) * factor
                            return colors.Color(r, g, b)
                        except Exception:
                            return colors.lightgrey

                    cells = ["", "", "", ""]
                    labels = [name for name, _ in palette]
                    sem_tbl = Table([cells, labels], colWidths=[90, 110, 130, 120])
                    styles_cmd: List[Tuple[Any, ...]] = []
                    for i, (name, hx) in enumerate(palette):
                        bg = colors.HexColor(hx) if i == idx and idx >= 0 else _lighten_hex(hx, 0.65)
                        styles_cmd.append(("BACKGROUND", (i, 0), (i, 0), bg))
                        styles_cmd.append(("BOX", (i, 0), (i, 0), 0.5, colors.black))
                        styles_cmd.append(("ALIGN", (i, 0), (i, 0), "CENTER"))
                        styles_cmd.append(("ALIGN", (i, 1), (i, 1), "CENTER"))
                    styles_cmd.append(("FONTNAME", (0, 1), (-1, 1), 'Helvetica'))
                    styles_cmd.append(("FONTSIZE", (0, 1), (-1, 1), 10))
                    styles_cmd.append(("GRID", (0, 1), (-1, 1), 0.25, colors.grey))
                    sem_tbl.setStyle(TableStyle(styles_cmd))
                    return sem_tbl
                except Exception:
                    return None

            elements = []
            elements.append(Paragraph("Informe Técnico de Vibraciones", title_style))
            elements.append(Spacer(1, 18))
            elements.append(Paragraph(f"Archivo analizado: {base_name}", styles['Normal']))
            elements.append(Spacer(1, 6))
            elements.append(Paragraph(f"Periodo analizado: {start_t:.2f}s - {end_t:.2f}s", styles['Normal']))
            elements.append(Spacer(1, 6))
            elements.append(Paragraph(f"Fecha de generacion: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            elements.append(Spacer(1, 6))
            elements.append(Paragraph(f"Aplicacion: V-Analyzer {APP_VERSION}", styles['Normal']))
            elements.append(Spacer(1, 14))

            severity_table = _build_severity_semaphore(severity_mm)
            if severity_table is not None:
                elements.append(Paragraph("Semáforo general de severidad", styles['SectionHeading']))
                elements.append(severity_table)
                elements.append(Spacer(1, 14))

            summary_rows = [
                ["Indicador", "Valor"],
                ["RMS velocidad global", f"{primary_rms_mm_pdf:.3f} mm/s"],
                ["Clasificación ISO global", primary_label_pdf],
                ["Frecuencia dominante", f"{features_full['dom_freq']:.2f} Hz"],
            ]
            if metadata_pdf:
                summary_rows.append(["Analizado en", metadata_pdf.get('analyzed_at', 'N/D')])
                summary_rows.append([
                    "Tiempo de análisis",
                    f"{float(metadata_pdf.get('runtime_s', 0.0)):.2f} s",
                ])
                if metadata_pdf.get("model_version"):
                    summary_rows.append(["Versión modelo ML", str(metadata_pdf.get("model_version"))])
            for entry in axis_summaries_pdf:
                if entry.get("is_global"):
                    continue
                try:
                    axis_rms_txt = f"{float(entry.get('rms_mm_s', 0.0)):.3f} mm/s"
                except Exception:
                    axis_rms_txt = "N/D"
                summary_rows.append([
                    f"RMS {entry.get('name', 'Eje')}",
                    f"{axis_rms_txt} → {entry.get('iso_label', 'N/D')}",
                ])
            summary_table = Table(summary_rows, colWidths=[200, 200])
            self._apply_table_style(summary_table)
            executive_body: List[Any] = [
                Paragraph("Condición general del activo", styles['HeadingAccent']),
                Paragraph(
                    f"El análisis indica una condición <b>{severity_mm}</b> con RMS global de {primary_rms_mm_pdf:.3f} mm/s.",
                    styles['Normal'],
                ),
                Spacer(1, 6),
                summary_table,
            ]
            elements.append(_pdf_card(executive_body, accent_color, background="#f4f4f4"))
            comparison_rows = [["Fuente", "Clasificación", "Detalle"]]
            comparison_rows.append([
                "ISO 20816",
                primary_label_pdf,
                f"RMS {primary_rms_mm_pdf:.3f} mm/s",
            ])
            ml_label_pdf = ml_result_pdf.get('label') if isinstance(ml_result_pdf, dict) else None
            if ml_result_pdf:
                comparison_rows.append([
                    "Modelo ML",
                    ml_label_pdf or ml_result_pdf.get('status', 'N/D'),
                    ml_result_pdf.get('message', '') if ml_result_pdf.get('status') != 'ok' else "",
                ])
            if conflict_pdf.get('conflict'):
                comparison_rows.append([
                    "Conflicto",
                    "⚠️ ISO vs ML",
                    conflict_pdf.get('note', 'Revisar discrepancia detectada.'),
                ])
            comp_table = Table(comparison_rows, colWidths=[130, 170, 220])
            self._apply_table_style(comp_table)
            comparison_body = [
                Paragraph("Comparativa ISO vs ML", styles['HeadingAccent']),
                comp_table,
            ]
            elements.append(_pdf_card(comparison_body, accent_color, background="#ffffff"))
            elements.append(Spacer(1, 18))
            elements.append(PageBreak())

            # Nota filtro visual (export): obtiene estado actual de la UI
            try:
                _pdf_fc = float(self.lf_cutoff_field.value) if getattr(self, 'lf_cutoff_field', None) and getattr(self.lf_cutoff_field, 'value', '') else 0.5
            except Exception:
                _pdf_fc = 0.5
            try:
                _pdf_hide_lf = bool(getattr(self, 'hide_lf_cb', None).value)
            except Exception:
                _pdf_hide_lf = True
            _pdf_fft_filter_note = f"Filtro visual FFT: oculta < {_pdf_fc:.2f} Hz" if _pdf_hide_lf else "Filtro visual FFT: sin ocultar"

            elements.append(Paragraph("Resumen Ejecutivo", styles['HeadingAccent']))
            elements.append(Spacer(1, 8))
            exec_findings_all = list(findings_core_pdf)
            exec_findings = self._select_main_findings(exec_findings_all)
            if not exec_findings:
                exec_findings = ["Sin anomalias evidentes segun reglas actuales."]
            elements.append(Paragraph(f"Clasificacion ISO global: {severity_mm}", styles['Normal']))
            elements.append(Spacer(1, 4))
            elements.append(Paragraph(f"RMS velocidad global: {rms_mm:.3f} mm/s", styles['Normal']))
            if axis_table_rows:
                tbl_axes = Table(axis_table_rows, colWidths=[160, 120, 180])
                self._apply_table_style(tbl_axes)
                elements.append(tbl_axes)
                elements.append(Spacer(1, 10))
            elements.append(Paragraph(f"Frecuencia dominante: {features_full['dom_freq']:.2f} Hz", styles['Normal']))
            elements.append(Spacer(1, 4))
            frac_low_pdf = float(features_full.get('frac_low', 0.0))
            frac_mid_pdf = float(features_full.get('frac_mid', 0.0))
            frac_high_pdf = float(features_full.get('frac_high', 0.0))
            if (frac_low_pdf + frac_mid_pdf + frac_high_pdf) > 0:
                balance_row = [
                    f"Baja (0-30 Hz): {frac_low_pdf * 100:.1f}%",
                    f"Media (30-120 Hz): {frac_mid_pdf * 100:.1f}%",
                    f"Alta (>120 Hz): {frac_high_pdf * 100:.1f}%",
                ]
                balance_tbl = Table([balance_row], colWidths=[150, 180, 160], hAlign="LEFT")
                balance_tbl.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#3498db")),
                            ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#9b59b6")),
                            ("BACKGROUND", (2, 0), (2, 0), colors.HexColor("#e67e22")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                            ("FONTSIZE", (0, 0), (-1, 0), 10),
                            ("BOX", (0, 0), (-1, 0), 0.25, colors.white),
                        ]
                    )
                )
                elements.append(balance_tbl)
                elements.append(Spacer(1, 10))
            elements.append(Paragraph(_pdf_fft_filter_note, styles['Normal']))
            elements.append(Spacer(1, 10))
            # Omitir bloque duplicado de diagnostico para evitar repeticion
            # elements.append(Paragraph("Diagnostico:", styles['Heading2']))
            for item in []:
                elements.append(Paragraph(f"- {item}", styles['Normal']))

            # Explicacion y recomendaciones (PDF)
            # Explicación y recomendaciones (unificadas con la app)
            exp_lines_pdf2 = self._build_explanations(res, exec_findings)
            elements.append(Paragraph("Explicacion y recomendaciones", styles['Heading2']))
            elements.append(Spacer(1, 6))
            if exp_lines_pdf2:
                items = [
                    ListItem(Paragraph(line, styles['Normal']), leftIndent=12, bulletColor=colors.HexColor("#2c3e50"))
                    for line in exp_lines_pdf2
                ]
                elements.append(ListFlowable(items, bulletType="bullet", start="•", spaceBefore=0, spaceAfter=6))
            elements.append(Spacer(1, 12))

            

            elements.append(Paragraph("Reporte de Análisis de Vibraciones", title_style))
            elements.append(Spacer(1, 8))
            elements.append(Paragraph(f"Archivo: {base_name}", styles['Normal']))
            elements.append(Spacer(1, 4))
            elements.append(Paragraph(f"Periodo: {start_t:.2f}s - {end_t:.2f}s", styles['Normal']))
            elements.append(Spacer(1, 12))

            # Top picos (FFT)
            if top_peaks:
                elements.append(Paragraph("Picos principales (FFT)", styles['Heading2']))
                elements.append(Spacer(1, 6))
                peaks_data = [[f"Frecuencia ({freq_unit})", "Amplitud (mm/s)", "Orden (X)"]]
                for pf, pa, order in top_peaks:
                    peaks_data.append([
                        f"{pf / freq_scale:.2f}",
                        f"{pa:.3f}",
                        f"{order:.2f}" if order else "-",
                    ])
                tbl_peaks = Table(peaks_data, colWidths=[120, 140, 120])
                self._apply_table_style(tbl_peaks)
                elements.append(tbl_peaks)
                elements.append(Spacer(1, 12))

            data_summary = [
                ["Metrica", "Valor"],
                ["RMS velocidad global", f"{rms_mm:.3f} mm/s"],
                ["Clasificacion ISO global", primary_label_pdf],
                [
                    "Frecuencia dominante",
                    f"{features_full['dom_freq'] / freq_scale:.2f} {freq_unit}",
                ],
            ]
            for entry in axis_summaries_pdf:
                if entry.get("is_global"):
                    continue
                try:
                    axis_rms_txt = f"{float(entry.get('rms_mm_s', 0.0)):.3f} mm/s"
                except Exception:
                    axis_rms_txt = "N/D"
                data_summary.append([
                    f"RMS {entry.get('name', 'Eje')}",
                    f"{axis_rms_txt} → {entry.get('iso_label', 'N/D')}"
                ])
            table_summary = Table(data_summary, colWidths=[200, 200])
            self._apply_table_style(table_summary)
            elements.append(table_summary)
            elements.append(Spacer(1, 16))

        except Exception as ex:
            self._log(f"Error exportando PDF: {ex}")
        finally:
            try:
                plt.rcParams.update(prev_style)
            except Exception:
                pass



    def _build_menu(self):

        def mb(icon, tip, key):

            b = MenuButton(icon, tip, self._on_menu_click, key, self.is_dark_mode)
            try:
                b.accent = self.accent
            except Exception:
                pass

            # Asegurar etiqueta de eje Y acorde a unidad seleccionada
            try:
                ax1.set_ylabel(_ylabel)
            except Exception:
                pass
            try:
                ax1.ticklabel_format(style="sci", axis="y", scilimits=(-2, 3))
            except Exception:
                pass

            self.menu_buttons[key] = b

            return b



        return ft.Container(

            width=80,  # Ancho fijo reducido

            expand=0,  # Sin expansión

            border_radius=0,

            bgcolor="#16213e" if self.is_dark_mode else "#ffffff",

            padding=ft.padding.only(

                left=5,

                right=5,

                top=20,

                bottom=20

            ),

            shadow=ft.BoxShadow(

                spread_radius=1,

                blur_radius=10,

                color=ft.Colors.with_opacity(0.3, "black"),

                offset=ft.Offset(2, 0),

            ),

            animate=ft.Animation(300, "easeInOut"),

            content=ft.Column(

                horizontal_alignment=ft.CrossAxisAlignment.CENTER,

                controls=[

                    ft.Container(

                        content=(
                            setattr(self, "menu_logo_icon", ft.Icon(ft.Icons.VIBRATION_ROUNDED, size=40, color=self._accent_ui()))
                            or self.menu_logo_icon
                        ),

                        padding=ft.padding.only(bottom=10),

                        visible=self.is_menu_expanded,

                        animate=ft.Animation(300, "easeInOut"),

                    ),

                    ft.Container(

                        content=(
                            setattr(self, "menu_logo_text", ft.Text(
                                "V-Analyzer",
                                size=12,
                                weight="bold",
                                color=self._accent_ui(),
                            ))
                            or self.menu_logo_text
                        ),

                        visible=self.is_menu_expanded,

                        animate=ft.Animation(300, "easeInOut"),

                    ),

                    ft.Divider(

                        height=20,

                        color=ft.Colors.with_opacity(0.2, "white"),

                        visible=self.is_menu_expanded

                    ),

                    ft.Column(expand=True, controls=[]),

                    *[mb(icon, tip, key) for icon, tip, key in [

                        (ft.Icons.HOME_ROUNDED, "Inicio", "welcome"),

                        (ft.Icons.FOLDER_OPEN_ROUNDED, "Archivos", "files"),

                        (ft.Icons.INSIGHTS_ROUNDED, "Análisis", "analysis"),

                        (ft.Icons.ASSESSMENT_ROUNDED, "Reportes", "reports"),

                        (ft.Icons.SETTINGS_ROUNDED, "Configuración", "settings"),

                    ]],

                    ft.Divider(height=20, color="transparent"),

                    ft.Container(

                        content=ft.Text(

                            APP_VERSION,

                            size=9,

                            color="#7f8c8d",

                            text_align="center",

                        ),

                        visible=self.is_menu_expanded,

                        animate=ft.Animation(300, "easeInOut"),

                    ),

                ],

            ),

        )



    def _toggle_menu(self, e):

        self.is_menu_expanded = not self.is_menu_expanded

        

        # Actualizar icono del botón

        e.control.icon = (

            ft.Icons.CHEVRON_LEFT_ROUNDED 

            if self.is_menu_expanded 

            else ft.Icons.CHEVRON_RIGHT_ROUNDED

        )

        

        # Actualizar el menú

        self.menu.width = 100 if self.is_menu_expanded else 80

        self.menu.padding = ft.padding.only(

            left=15 if self.is_menu_expanded else 10,

            right=15 if self.is_menu_expanded else 10,

            top=20,

            bottom=20

        )

        

        # Actualizar visibilidad de elementos

        for control in self.menu.content.controls:

            if isinstance(control, ft.Container) or isinstance(control, ft.Text) or isinstance(control, ft.Divider):

                if not isinstance(control, MenuButton):

                    control.visible = self.is_menu_expanded

    

        # Actualizar el menú

        self.menu.update()



    def _build_control_panel(self):

        self.tabs = ft.Tabs(

            selected_index=0,

            tabs=[

                ft.Tab(text="Acciones", icon=ft.Icons.TOUCH_APP_ROUNDED),

                ft.Tab(text="Ayuda", icon=ft.Icons.HELP_OUTLINE_ROUNDED),

                ft.Tab(text="Registro", icon=ft.Icons.HISTORY_ROUNDED)

            ],

            expand=1,

            on_change=self._on_tab_change,

            animation_duration=300,

        )

        

        # Crear botón de colapsar panel

        toggle_button = ft.IconButton(

            icon=ft.Icons.CHEVRON_LEFT_ROUNDED,

            icon_size=20,

            tooltip="Colapsar panel",

            on_click=self._toggle_panel,

        )



        panel_header = ft.Row(

            controls=[

                ft.Text("Panel de Control", size=18, weight="bold"),

                toggle_button

            ],

            alignment="space_between"

        )



        # Resto del contenido del panel

        upload_btn = ft.ElevatedButton(

            "Cargar Archivo",

            icon=ft.Icons.UPLOAD_FILE_ROUNDED,

            on_click=self._pick_files,

            style=ft.ButtonStyle(

                bgcolor=self._accent_ui(),

                color="white",

                shape=ft.RoundedRectangleBorder(radius=10),

            ),

            width=200,

            height=45,

        )

        try:
            self.btn_upload = upload_btn
        except Exception:
            pass

        self.quick_actions = ft.Column(

            spacing=15,

            controls=[

                upload_btn,

                ft.Row(

                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,

                    controls=[

                        ft.IconButton(

                            icon=ft.Icons.DARK_MODE_ROUNDED if not self.is_dark_mode else ft.Icons.LIGHT_MODE_ROUNDED,

                            tooltip="Cambiar tema",

                            on_click=self._toggle_theme,

                            icon_size=24,

                        ),

                        ft.IconButton(

                            icon=ft.Icons.ACCESS_TIME_ROUNDED,

                            tooltip="Formato hora",

                            on_click=self._toggle_clock_format,

                            icon_size=24,

                        ),

                    ]

                ),

                (setattr(self, "clock_card", ft.Container(
                    content=self.clock_text,
                    bgcolor=ft.Colors.with_opacity(0.1, self._accent_ui()),
                    padding=10,
                    border_radius=10,
                    alignment=ft.alignment.center,
                )) or self.clock_card),

            ]

        )

        

        self.log_panel = ft.ListView(expand=1, auto_scroll=False, spacing=2)

        self.help_panel = ft.Column(

            scroll="auto",

            spacing=10,

            controls=[

                ft.Container(

                    content=ft.Text("📋 Ayuda contextual", size=16, weight="bold"),

                    padding=ft.padding.only(bottom=10)

                ),

                ft.Text("Información de ayuda aparecerá aquí según la sección actual.", size=13)

            ]

        )

        

        self.tab_content = ft.Container(content=self.quick_actions, expand=True, padding=10)

        

        return ft.Container(

            width=350 if self.is_panel_expanded else 80,

            padding=20 if self.is_panel_expanded else 10,

            border_radius=0,

            bgcolor="#16213e" if self.is_dark_mode else "#ffffff",

            shadow=ft.BoxShadow(

                spread_radius=1,

                blur_radius=10,

                color=ft.Colors.with_opacity(0.3, "black"),

                offset=ft.Offset(-2, 0),

            ),

            animate=ft.Animation(300, "easeInOut"),  # Fixed animation syntax

            content=ft.Column(

                expand=True,

                controls=[

                    panel_header,

                    ft.Divider(color=ft.Colors.with_opacity(0.2, "white")),

                    ft.Container(

                        content=ft.Column(

                            controls=[

                                self.tabs,

                                ft.Divider(color=ft.Colors.with_opacity(0.2, "white")),

                                self.tab_content

                            ]

                        ),

                        visible=self.is_panel_expanded

                    )

                ]

            ),

        )



    def _build_welcome_view(self):

        return ft.Column(

            controls=[

                ft.Container(height=50),

                ft.Icon(ft.Icons.MONITOR_HEART_ROUNDED, size=100, color=self._accent_ui()),

                ft.Container(height=20),

                ft.Text(

                    "Sistema de Diagnóstico Predictivo",

                    size=32,

                    weight="bold",

                    text_align="center"

                ),

                ft.Text(

                    "Análisis de Vibraciones Mecánicas mediante FFT y Machine Learning",

                    size=16,

                    color="#7f8c8d",

                    text_align="center"

                ),

                ft.Container(height=40),

                ft.Row(

                    alignment="center",

                    controls=[

                        ft.ElevatedButton(

                            "Comenzar Análisis",

                            icon=ft.Icons.PLAY_ARROW_ROUNDED,

                            on_click=self._pick_files,

                            style=ft.ButtonStyle(

                                bgcolor=self._accent_ui(),

                                color="white",

                                shape=ft.RoundedRectangleBorder(radius=10),

                                padding=20,

                            ),

                            height=50,

                        ),

                        ft.OutlinedButton(

                            "Ver Documentación",

                            icon=ft.Icons.DESCRIPTION_ROUNDED,

                            on_click=lambda _: self.page.launch_url("https://drive.google.com/file/d/1UqlL1s7jGTq3A38UV2r6AE2eVb915w41/view?usp=sharing")

,

                            style=ft.ButtonStyle(

                                shape=ft.RoundedRectangleBorder(radius=10),

                                padding=20,

                            ),

                            height=50,

                        ),

                    ]

                ),

                ft.Container(height=40),

                ft.Container(

                    content=ft.Column(

                        horizontal_alignment="center",

                        controls=[

                            ft.Text("Características del Sistema", size=18, weight="bold"),

                            ft.Container(height=10),

                            ft.Row(

                                alignment="center",

                                spacing=30,

                                controls=[

                                    self._create_feature_card("FFT", "Transformada Rápida de Fourier", ft.Icons.TRANSFORM_ROUNDED),

                                    self._create_feature_card("ML", "Machine Learning Predictivo", ft.Icons.PSYCHOLOGY_ROUNDED),

                                    self._create_feature_card("ISO", "Normas ISO 20816-3", ft.Icons.VERIFIED_ROUNDED),

                                ]

                            )

                        ]

                    ),

                    padding=20,

                )
            
            
            

            ],

            horizontal_alignment="center",

            alignment="center",

            expand=True,

            

        )



    def _create_feature_card(self, title, subtitle, icon):

        return ft.Container(

            content=ft.Column(

                horizontal_alignment="center",

                spacing=5,

                controls=[

                    ft.Icon(icon, size=40, color=self._accent_ui()),

                    ft.Text(title, size=16, weight="bold"),

                    ft.Text(subtitle, size=12, color="#7f8c8d", text_align="center"),

                ]

            ),

            padding=15,

            border_radius=10,

            bgcolor=ft.Colors.with_opacity(0.05, self._accent_ui()),

            width=150,

            height=120,

        )


    def _build_ml_summary_card(self, ml_payload: Optional[Dict[str, Any]]) -> Optional[ft.Control]:
        """Renderiza un apartado visual con el resultado del modelo de ML en la UI."""

        if not ml_payload:
            return None
        ml_result = (ml_payload or {}).get("result") or {}
        status = ml_result.get("status")
        accent = self._accent_ui()
        header = ft.Row(
            [
                ft.Icon(ft.Icons.PSYCHOLOGY_ROUNDED, color=accent),
                ft.Text(
                    "Diagnóstico por Machine Learning",
                    size=16,
                    weight="bold",
                    expand=True,
                ),
            ],
            spacing=8,
            alignment="start",
        )

        def _metric_chip(title: str, value: str) -> ft.Container:
            return ft.Container(
                content=ft.Column(
                    [
                        ft.Text(title, size=11, color="#7f8c8d"),
                        ft.Text(value, weight="bold", size=13),
                    ],
                    spacing=2,
                ),
                bgcolor=ft.Colors.with_opacity(0.08, accent),
                border_radius=10,
                padding=ft.padding.symmetric(horizontal=12, vertical=8),
            )

        features = (ml_payload or {}).get("features") or {}
        metrics_row = ft.Row(
            [
                _metric_chip("RMS vel.", f"{features.get('rms_vel_mm_s', 0.0):.3f} mm/s"),
                _metric_chip("Frec. dominante", f"{features.get('dom_freq_hz', 0.0):.2f} Hz"),
                _metric_chip("Energía alta", f"{features.get('energy_high', 0.0) * 100:.1f}%"),
            ],
            spacing=12,
            run_spacing=12,
            wrap=True,
        )

        if status == "ok":
            label = str(ml_result.get("label", "Diagnóstico"))
            classes = list(ml_result.get("classes") or [])
            probabilities = list(ml_result.get("probabilities") or [])
            probs_view: List[ft.Control] = []
            if classes and probabilities and len(classes) == len(probabilities):
                ranked = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)
                for cls, prob in ranked:
                    prob_val = float(prob)
                    bar = ft.ProgressBar(value=max(0.0, min(1.0, prob_val)), color=accent)
                    probs_view.append(
                        ft.Column(
                            [
                                ft.Row(
                                    [
                                        ft.Text(str(cls), weight="bold", expand=True),
                                        ft.Text(f"{prob_val * 100:.1f}%", weight="bold"),
                                    ],
                                    alignment="spaceBetween",
                                ),
                                bar,
                            ],
                            spacing=4,
                        )
                    )
            highlight = ft.Container(
                content=ft.Text(
                    f"Resultado sugerido: {label}",
                    weight="bold",
                    color=accent,
                ),
                bgcolor=ft.Colors.with_opacity(0.15, accent),
                border_radius=12,
                padding=ft.padding.symmetric(horizontal=14, vertical=10),
            )
            body_controls: List[ft.Control] = [highlight, metrics_row]
            if probs_view:
                body_controls.append(
                    ft.Column(
                        probs_view,
                        spacing=6,
                    )
                )
            return ft.Container(
                content=ft.Column(
                    [
                        header,
                        ft.Text(
                            "Predicción automática entrenada con historial de vibraciones.",
                            size=12,
                            color="#7f8c8d",
                        ),
                        *body_controls,
                    ],
                    spacing=10,
                ),
                border_radius=12,
                padding=ft.padding.all(16),
                bgcolor=ft.Colors.with_opacity(0.06, accent),
            )

        message = ml_result.get("message") or "Modelo no disponible."
        status_text = "Modelo ML no disponible" if status == "unavailable" else "Diagnóstico ML no generado"
        return ft.Container(
            content=ft.Column(
                [
                    header,
                    ft.Container(
                        content=ft.Text(
                            status_text,
                            weight="bold",
                            color=accent,
                        ),
                        bgcolor=ft.Colors.with_opacity(0.1, accent),
                        padding=ft.padding.symmetric(horizontal=12, vertical=8),
                        border_radius=10,
                    ),
                    ft.Text(message, size=12, color="#7f8c8d"),
                ],
                spacing=8,
            ),
            border_radius=12,
            padding=ft.padding.all(16),
            bgcolor=ft.Colors.with_opacity(0.04, accent),
        )


    def _build_files_view(self):

        self._refresh_files_list()

        return ft.Column(

            controls=[

                ft.Container(

                    content=ft.Column(

                        controls=[

                            ft.Row(

                                alignment="space_between",

                                controls=[

                                    ft.Text("Gestión de Archivos", size=24, weight="bold"),

                                    ft.ElevatedButton(

                                        "Agregar Archivo",

                                        icon=ft.Icons.ADD_ROUNDED,

                                        on_click=self._pick_files,

                                        style=ft.ButtonStyle(

                                            bgcolor=self._accent_ui(),

                                            color="white",

                                            shape=ft.RoundedRectangleBorder(radius=8),

                                        ),

                                    ),

                                ]

                            ),

                            ft.Row([
                                ft.Text(f"Total de archivos: {len(self.uploaded_files)}", size=14, color="#7f8c8d"),
                                setattr(self, 'data_favs_only_cb', ft.Checkbox(label="Mostrar favoritos", value=bool(self.data_show_favs_only), on_change=lambda e: self._toggle_data_favs_filter())) or self.data_favs_only_cb,
                            ], alignment="spaceBetween"),

                            # Buscador de archivos en gestor
                            (setattr(self, 'data_search', ft.TextField(
                                hint_text="Buscar por nombre...",
                                expand=True,
                                on_change=lambda e: self._refresh_files_list(),
                                dense=True,
                            )) or self.data_search),

                        ]

                    ),

                    padding=ft.padding.only(bottom=20)

                ),

                ft.Container(

                    content=self.files_list_view,

                    border=ft.border.all(1, ft.Colors.with_opacity(0.2, "white")),

                    border_radius=10,

                    expand=True,

                    padding=5,

                )

            ],

            expand=True

        )



    def _export_pdf(self, e=None):
        """

        Exporta reporte PDF extendido con diagnóstico automático:

        - Señal principal (tiempo + FFT)

        - Segmento seleccionado

        - Variables auxiliares

        - Tablas y diagnóstico (reglas baseline)

        """
        # Legacy wrapper: delega al exportador unificado
        try:
            return self.exportar_pdf(e)
        except Exception:
            pass

        try:

            if self.current_df is None:

                self._log("No hay datos para exportar")

                return



            reports_dir = os.path.join(os.getcwd(), "reports")

            os.makedirs(reports_dir, exist_ok=True)



            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            base_name = os.path.splitext(os.path.basename(self.uploaded_files[0]))[0]

            pdf_name = f"{timestamp}_{base_name}.pdf"

            pdf_path = os.path.join(reports_dir, pdf_name)



            # --- Estilo claro solo para PDF ---

            plt.style.use("seaborn-v0_8-whitegrid")
            plt.rcParams["font.family"] = "DejaVu Sans"



            # Preparar análisis

            time_col = self.time_dropdown.value

            fft_signal_col = self.fft_dropdown.value

            t = self.current_df[time_col].to_numpy()

            signal = self.current_df[fft_signal_col].to_numpy()



            # Periodo seleccionado

            mask, start_t, end_t = self._resolve_analysis_period(t)
            if mask.size == 0 or np.count_nonzero(mask) < 2:
                mask = np.isfinite(np.asarray(t, dtype=float))
                start_t = float(np.nanmin(t[mask])) if np.any(mask) else 0.0
                end_t = float(np.nanmax(t[mask])) if np.any(mask) else 0.0
            segment_idx = np.nonzero(mask)[0]
            t_seg_raw = t[segment_idx]
            sig_seg_raw = signal[segment_idx]
            segment_df = self.current_df.iloc[segment_idx]
            t_seg, acc_seg, _, _ = self._prepare_segment_for_analysis(t_seg_raw, sig_seg_raw, fft_signal_col)



            # FFT -> velocidad (mm/s y m/s)
            xf, mag_vel_mm, mag_vel = self._compute_fft_dual(acc_seg, t_seg)
            vel_time_mm = self._acc_to_vel_time_mm(acc_seg, t_seg)
            if vel_time_mm.size:
                rms_mm = float(np.sqrt(np.mean(vel_time_mm**2)))
                rms_ms = rms_mm / 1000.0
            else:
                rms_mm = 0.0
                rms_ms = 0.0
            severity_mm = self._classify_severity(rms_mm)
            severity_label_ms, severity_color_ms = self._classify_severity_ms(rms_ms)

            # --- Features + diagnóstico para el PDF (usa mm/s) ---
            features_full = self._extract_features(t_seg, acc_seg, xf, mag_vel_mm)
            # Guardar última FFT/segmento para diagnóstico avanzado (PDF)
            self._last_xf = xf
            self._last_spec = mag_vel_mm
            self._last_tseg = t_seg
            self._last_accseg = acc_seg
            findings_pdf = res.get('diagnosis', [])
            severity_entry_pdf = res.get('diagnosis_summary')
            findings_core_pdf = list(res.get('diagnosis_findings', []) or [])
            if not findings_core_pdf and findings_pdf:
                _, findings_core_pdf = _split_diagnosis(findings_pdf)
            charlotte_catalog_pdf = list(res.get('charlotte_catalog', []) or [])
            if not charlotte_catalog_pdf:
                charlotte_catalog_pdf = [dict(entry) for entry in CHARLOTTE_MOTOR_FAULTS]


            # Guardar gráficas como imágenes

            tmp_imgs: List[str] = []



            try:
                unit_mode = getattr(self.time_unit_dd, 'value', 'vel_mm')
            except Exception:
                unit_mode = 'vel_mm'
            if unit_mode == 'vel_mm':
                _y_time = self._acc_to_vel_time_mm(acc_seg, t_seg)
                _ylabel = 'Velocidad [mm/s]'
                _rms_text = (
                    f"RMS velocidad: {self._calculate_rms(_y_time):.3f} mm/s"
                    if _y_time.size
                    else "RMS velocidad: 0.000 mm/s"
                )
            elif unit_mode == 'acc_g':
                _y_time = acc_seg / 9.80665
                _ylabel = 'Aceleración [g]'
                _rms_text = f"RMS acc: {self._calculate_rms(_y_time):.3f} g"
            else:
                _y_time = acc_seg
                _ylabel = 'Aceleración [m/s²]'
                _rms_text = f"RMS acc: {self._calculate_rms(_y_time):.3e} m/s^2"

            # Señal principal
            fig1, ax1 = plt.subplots(figsize=(8, 3))
            if len(t_seg) > 0 and len(_y_time) > 0:
                ax1.plot(t_seg, _y_time, color=self.time_plot_color)
            ax1.set_title(f"Señal {fft_signal_col} ({start_t:.2f}-{end_t:.2f}s)")
            ax1.set_xlabel("Tiempo (s)")
            ax1.set_ylabel(_ylabel)
            try:
                text_color = 'white' if self.is_dark_mode else 'black'
                ax1.text(0.02, 0.95, _rms_text, transform=ax1.transAxes, va='top', color=text_color)
            except Exception:
                pass
            img_time = self._save_temp_plot(fig1, tmp_imgs)

            fig2, ax2 = plt.subplots(figsize=(8, 3))
            freq_scale = getattr(self, "_fft_display_scale", 1.0) or 1.0
            if not np.isfinite(freq_scale) or freq_scale <= 0:
                freq_scale = 1.0
            freq_unit = getattr(self, "_fft_display_unit", "Hz") or "Hz"
            if xf is not None and mag_vel_mm is not None:
                xdisp = np.asarray(xf, dtype=float) / freq_scale
                ax2.plot(xdisp, mag_vel_mm, color=self.fft_plot_color)
            ax2.set_title(f"FFT (Velocidad)")
            ax2.set_xlabel(f"Frecuencia ({freq_unit})")
            ax2.set_ylabel("Velocidad [mm/s]")
            # Anotación RMS (m/s) y eje superior en RPM
            try:
                text_color = "white" if self.is_dark_mode else "black"
                ax2.text(0.02, 0.95, f"RMS velocidad: {rms_mm:.3f} mm/s", transform=ax2.transAxes,
                         va="top", color=text_color)
                ax2_rpm = ax2.twiny()
                xlim = ax2.get_xlim()
                ax2_rpm.set_xlim(xlim[0] * freq_scale * 60, xlim[1] * freq_scale * 60)
                ax2_rpm.set_xlabel("Frecuencia (RPM)")
            except Exception:
                pass
            # Eje superior en RPM
            try:
                ax2_rpm = ax2.twiny()
                xlim = ax2.get_xlim()
                ax2_rpm.set_xlim(xlim[0] * freq_scale * 60, xlim[1] * freq_scale * 60)
                ax2_rpm.set_xlabel("Frecuencia (RPM)")
            except Exception:
                pass
            img_fft = self._save_temp_plot(fig2, tmp_imgs)


            # Variables auxiliares (si las hay marcadas)

            aux_selected = [(cb.label, color_dd.value, style_dd.value)

                            for cb, color_dd, style_dd in self.aux_controls if cb.value]



            aux_imgs = []

            for col, color, style in aux_selected:

                aux_fig, aux_ax = plt.subplots(figsize=(8, 2))

                aux_ax.plot(self.current_df[time_col], self.current_df[col],

                            color=color, linestyle=style, linewidth=2, label=col)

                aux_ax.set_title(f"{col} vs Tiempo")

                aux_ax.legend()

                aux_ax.set_xlabel("Tiempo (s)")

                aux_ax.set_ylabel(col)

                aux_imgs.append(self._save_temp_plot(aux_fig, tmp_imgs))



            # Crear PDF

            doc = SimpleDocTemplate(pdf_path, pagesize=A4)

            styles = getSampleStyleSheet()
            try:
                accent_hex = self._accent_hex()
            except Exception:
                accent_hex = "#1f77b4"
            try:
                accent_color = colors.HexColor(accent_hex)
            except Exception:
                accent_color = colors.HexColor("#1f77b4")
            title_style = ParagraphStyle(
                "title",
                parent=styles['Title'],
                textColor=accent_color,
                spaceAfter=12,
            )
            styles.add(
                ParagraphStyle(
                    "HeadingAccent",
                    parent=styles['Heading1'],
                    textColor=colors.HexColor("#2c3e50"),
                    spaceAfter=8,
                )
            )
            styles.add(
                ParagraphStyle(
                    "SectionHeading",
                    parent=styles['Heading2'],
                    textColor=colors.HexColor("#2c3e50"),
                    spaceBefore=12,
                    spaceAfter=6,
                )
            )
            styles.add(
                ParagraphStyle(
                    "BulletItem",
                    parent=styles['Normal'],
                    leftIndent=18,
                    spaceAfter=4,
                )
            )
            styles.add(
                ParagraphStyle(
                    "Muted",
                    parent=styles['Normal'],
                    textColor=colors.HexColor("#7f8c8d"),
                    fontSize=10,
                )
            )
            styles.add(
                ParagraphStyle(
                    "Chip",
                    parent=styles['Normal'],
                    textColor=colors.HexColor("#2c3e50"),
                    backColor=colors.HexColor("#e6e6e6"),
                    alignment=1,
                    fontSize=12,
                    leading=14,
                    borderPadding=(4, 8, 4, 8),
                )
            )

            generated_at = datetime.now()

            def _draw_header_footer(canvas, doc_obj):
                canvas.saveState()
                header_height = 22
                canvas.setFillColor(accent_color)
                canvas.rect(0, A4[1] - header_height, A4[0], header_height, fill=1, stroke=0)
                canvas.setFillColor(colors.white)
                canvas.setFont("Helvetica-Bold", 11)
                canvas.drawString(36, A4[1] - header_height + 7, "Informe de Vibraciones")
                canvas.setFont("Helvetica", 8)
                canvas.drawRightString(A4[0] - 36, A4[1] - header_height + 7, base_name)
                canvas.setFillColor(colors.HexColor("#7f8c8d"))
                canvas.setFont("Helvetica", 8)
                canvas.drawString(36, 24, generated_at.strftime("%d/%m/%Y %H:%M"))
                canvas.drawRightString(A4[0] - 36, 24, f"Página {doc_obj.page}")
                canvas.restoreState()

            def _chunk_list(seq, size):
                for i in range(0, len(seq), size):
                    yield seq[i : i + size]

            styles.add(
                ParagraphStyle(
                    "CoverHeading",
                    parent=styles["Heading1"],
                    fontSize=26,
                    leading=30,
                    textColor=colors.white,
                    spaceAfter=4,
                )
            )
            styles.add(
                ParagraphStyle(
                    "CoverSubtitle",
                    parent=styles["Normal"],
                    fontSize=12,
                    textColor=colors.white,
                    spaceAfter=2,
                )
            )
            styles.add(
                ParagraphStyle(
                    "MetaLabel",
                    parent=styles["Normal"],
                    textColor=colors.HexColor("#6c7a89"),
                    fontSize=10,
                )
            )
            styles.add(
                ParagraphStyle(
                    "MetaValue",
                    parent=styles["Heading3"],
                    textColor=colors.HexColor("#2c3e50"),
                    fontSize=12,
                )
            )
            styles.add(
                ParagraphStyle(
                    "CardTitle",
                    parent=styles["Heading3"],
                    textColor=colors.HexColor("#2c3e50"),
                    fontSize=12,
                    spaceAfter=2,
                )
            )
            styles.add(
                ParagraphStyle(
                    "CardValue",
                    parent=styles["Title"],
                    textColor=accent_color,
                    fontSize=16,
                    spaceAfter=4,
                )
            )

            elements: List[Any] = []
            cover_banner = Table(
                [
                    [Paragraph("Informe de Análisis de Vibraciones", styles["CoverHeading"])],
                    [Paragraph(f"Archivo: <b>{base_name}</b>", styles["CoverSubtitle"])],
                    [Paragraph(f"Periodo analizado: {start_t:.2f}s – {end_t:.2f}s", styles["CoverSubtitle"])],
                ],
                colWidths=[doc.width],
            )
            cover_banner.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), accent_color),
                        ("LEFTPADDING", (0, 0), (-1, -1), 32),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 32),
                        ("TOPPADDING", (0, 0), (-1, -1), 32),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 28),
                    ]
                )
            )
            elements.append(cover_banner)
            elements.append(Spacer(1, 12))

            meta_rows = [
                [Paragraph("Generado el", styles["MetaLabel"]), Paragraph(generated_at.strftime("%d/%m/%Y %H:%M"), styles["MetaValue"])],
                [Paragraph("Duración analizada", styles["MetaLabel"]), Paragraph(f"{(end_t - start_t):.2f} s", styles["MetaValue"])],
                [Paragraph("Ventana FFT", styles["MetaLabel"]), Paragraph(self.fft_window_type.upper(), styles["MetaValue"])],
            ]
            meta_table = Table(meta_rows, colWidths=[doc.width * 0.35, doc.width * 0.65])
            meta_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f5f7fa")),
                        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#d5dde5")),
                        ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#e3e9ee")),
                        ("LEFTPADDING", (0, 0), (-1, -1), 12),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                        ("TOPPADDING", (0, 0), (-1, -1), 8),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ]
                )
            )
            elements.append(meta_table)
            elements.append(Spacer(1, 14))

            summary_cards = [
                _pdf_card(
                    [
                        Paragraph("RMS global", styles["CardTitle"]),
                        Paragraph(f"{primary_rms_mm_pdf:.3f} mm/s", styles["CardValue"]),
                        Paragraph(severity_mm, styles["Muted"]),
                    ],
                    accent_color,
                    background="#ffffff",
                ),
                _pdf_card(
                    [
                        Paragraph("Frecuencia dominante", styles["CardTitle"]),
                        Paragraph(f"{dom_freq:.2f} Hz", styles["CardValue"]),
                        Paragraph("Máximo espectral del análisis", styles["Muted"]),
                    ],
                    accent_color,
                    background="#ffffff",
                ),
                _pdf_card(
                    [
                        Paragraph("Ventana FFT", styles["CardTitle"]),
                        Paragraph(self.fft_window_type.upper(), styles["CardValue"]),
                        Paragraph("Configuración actual", styles["Muted"]),
                    ],
                    accent_color,
                    background="#ffffff",
                ),
            ]
            summary_table = Table([summary_cards], colWidths=[doc.width / 3.0] * 3, hAlign="CENTER")
            summary_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
            elements.append(summary_table)
            elements.append(Spacer(1, 14))

            axis_table_rows: List[List[str]] = []
            if axis_summaries_pdf:
                axis_table_rows.append(["Canal", "RMS (mm/s)", "Clasificación ISO"])
                for entry in axis_summaries_pdf:
                    try:
                        rms_txt = f"{float(entry.get('rms_mm_s', 0.0)):.3f}"
                    except Exception:
                        rms_txt = "N/D"
                    axis_table_rows.append(
                        [
                            entry.get("name", "Eje"),
                            rms_txt,
                            entry.get("iso_label", "N/D"),
                        ]
                    )

            elements.append(Paragraph("Resumen ejecutivo", styles["SectionHeading"]))
            exec_findings_all2 = list(findings_core_pdf)
            exec_findings = self._select_main_findings(exec_findings_all2)
            if not exec_findings:
                exec_findings = ["Sin anomalías evidentes según reglas actuales."]
            summary_card = _pdf_card(
                [
                    Paragraph(f"Clasificación ISO global: <b>{severity_mm}</b>", styles["Normal"]),
                    Paragraph(f"RMS velocidad global: <b>{rms_mm:.3f} mm/s</b>", styles["Normal"]),
                    Paragraph(f"Frecuencia dominante: <b>{features_full['dom_freq']:.2f} Hz</b>", styles["Normal"]),
                ],
                accent_color,
                background="#f9f9fb",
            )
            elements.append(summary_card)

            balance_rows = []
            frac_low_pdf2 = float(features_full.get("frac_low", 0.0))
            frac_mid_pdf2 = float(features_full.get("frac_mid", 0.0))
            frac_high_pdf2 = float(features_full.get("frac_high", 0.0))
            if (frac_low_pdf2 + frac_mid_pdf2 + frac_high_pdf2) > 0:
                balance_rows = [
                    ["Baja (0-30 Hz)", f"{frac_low_pdf2 * 100:.1f}%"],
                    ["Media (30-120 Hz)", f"{frac_mid_pdf2 * 100:.1f}%"],
                    ["Alta (>120 Hz)", f"{frac_high_pdf2 * 100:.1f}%"],
                ]
            if balance_rows:
                balance_table = Table(balance_rows, colWidths=[doc.width * 0.45, doc.width * 0.45])
                balance_table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#ffffff")),
                            ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#d7d7d7")),
                            ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#d0d0d0")),
                            ("LEFTPADDING", (0, 0), (-1, -1), 8),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                            ("TOPPADDING", (0, 0), (-1, -1), 4),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                        ]
                    )
                )
                elements.append(Spacer(1, 10))
                elements.append(balance_table)

            elements.append(Spacer(1, 12))
            elements.append(Paragraph("Hallazgos principales", styles["SectionHeading"]))
            finding_bullets = [
                ListItem(Paragraph(text, styles["Normal"]), bulletColor=colors.HexColor("#2c3e50"))
                for text in exec_findings
            ]
            elements.append(
                ListFlowable(
                    finding_bullets,
                    bulletType="bullet",
                    bulletColor=colors.HexColor("#2c3e50"),
                    start="bulletchar",
                    leftIndent=16,
                )
            )

            if axis_table_rows:
                elements.append(Spacer(1, 14))
                elements.append(Paragraph("Detalle por canal", styles["SectionHeading"]))
                axis_table = Table(axis_table_rows, colWidths=[doc.width * 0.35, doc.width * 0.25, doc.width * 0.4])
                axis_table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), accent_color),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 8),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                            ("TOPPADDING", (0, 0), (-1, -1), 6),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#d7d7d7")),
                        ]
                    )
                )
                elements.append(axis_table)

            chart_paths = [img for img in [img_time, img_fft, img_env, img_runup, img_orbit] if img]
            if chart_paths:
                elements.append(PageBreak())
                elements.append(Paragraph("Visualizaciones principales", styles["SectionHeading"]))
                chart_rows = []
                for chunk in _chunk_list(chart_paths, 2):
                    row = []
                    for img in chunk:
                        row.append(Image(img, width=(doc.width / 2) - 6, height=180))
                    if len(row) == 1:
                        row.append(Spacer(1, 0))
                    chart_rows.append(row)
                charts_table = Table(chart_rows, colWidths=[doc.width / 2, doc.width / 2], hAlign="CENTER")
                charts_table.setStyle(
                    TableStyle(
                        [
                            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                        ]
                    )
                )
                elements.append(charts_table)

            if aux_imgs:
                elements.append(Paragraph("Variables auxiliares", styles["SectionHeading"]))
                aux_rows = []
                for chunk in _chunk_list(aux_imgs, 2):
                    row = []
                    for img in chunk:
                        row.append(Image(img, width=(doc.width / 2) - 6, height=150))
                    if len(row) == 1:
                        row.append(Spacer(1, 0))
                    aux_rows.append(row)
                aux_table_img = Table(aux_rows, colWidths=[doc.width / 2, doc.width / 2], hAlign="CENTER")
                aux_table_img.setStyle(
                    TableStyle(
                        [
                            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                        ]
                    )
                )
                elements.append(aux_table_img)

            elements.append(Spacer(1, 12))
            diagnostic_card = _pdf_card(
                [
                    Paragraph(
                        f"El valor RMS calculado es <b>{rms_mm:.3f} mm/s</b>, lo cual corresponde a la condición <b>{severity_mm}</b>.",
                        styles["Normal"],
                    )
                ],
                accent_color,
                background="#ffffff",
            )
            elements.append(Paragraph("Diagnóstico consolidado", styles["SectionHeading"]))
            elements.append(diagnostic_card)



            ml_status_pdf = ml_result_pdf.get('status') if isinstance(ml_result_pdf, dict) else None
            if ml_status_pdf:
                elements.append(Spacer(1, 12))
                ml_card_body: List[Any] = [Paragraph("Diagnóstico por Machine Learning", styles['HeadingAccent'])]
                ml_message = None
                accent_hex_value = "#1f77b4"
                try:
                    if accent_color:
                        hv = accent_color.hexval() if callable(getattr(accent_color, "hexval", None)) else None
                        if isinstance(hv, str):
                            accent_hex_value = hv.replace("0x", "#") if hv.startswith("0x") else hv
                except Exception:
                    accent_hex_value = "#1f77b4"
                if ml_status_pdf == 'ok':
                    ml_label = str(ml_result_pdf.get('label', ''))
                    ml_card_body.append(Paragraph(f"Resultado sugerido: {ml_label}", styles['Chip']))
                    ml_card_body.append(Spacer(1, 6))
                    ml_card_body.append(Paragraph("Características principales evaluadas:", styles['Muted']))
                    ml_features_pdf = (ml_bundle_pdf or {}).get('features') or {}
                    feature_metrics = [
                        ("RMS velocidad", f"{ml_features_pdf.get('rms_vel_mm_s', 0.0):.3f} mm/s"),
                        ("Frecuencia dominante", f"{ml_features_pdf.get('dom_freq_hz', 0.0):.2f} Hz"),
                        ("Relación 2X", f"{ml_features_pdf.get('r2x', 0.0):.2f}"),
                        ("Energía alta", f"{ml_features_pdf.get('energy_high', 0.0) * 100:.1f}%"),
                    ]
                    ml_card_body.append(_pdf_metric_grid(feature_metrics, accent_color))
                    classes = list(ml_result_pdf.get('classes') or [])
                    probabilities = list(ml_result_pdf.get('probabilities') or [])
                    if classes and probabilities and len(classes) == len(probabilities):
                        ranked = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)
                        proba_rows: List[List[Any]] = []
                        for cls, prob in ranked:
                            proba_rows.append([
                                Paragraph(f"<b>{str(cls)}</b>", styles['Normal']),
                                ProbabilityBar(float(prob), fill_color=accent_hex_value, back_color="#eaf3ff"),
                                Paragraph(f"{float(prob) * 100:.1f}%", styles['Normal']),
                            ])
                        proba_tbl = Table(proba_rows, colWidths=[200, 150, 90])
                        proba_tbl.setStyle(
                            TableStyle(
                                [
                                    ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d7d7d7")),
                                ]
                            )
                        )
                        ml_card_body.append(Spacer(1, 6))
                        ml_card_body.append(Paragraph("Probabilidades por clase:", styles['Muted']))
                        ml_card_body.append(proba_tbl)
                else:
                    ml_message = ml_result_pdf.get('message') or "No se pudo obtener el diagnóstico automático."
                if ml_message:
                    ml_card_body.append(Paragraph(ml_message, styles['Muted']))
                elements.append(_pdf_card(ml_card_body, accent_color, background="#ffffff"))

            if charlotte_catalog_pdf:
                elements.append(PageBreak())
                elements.append(Paragraph("Referencia Tabla de Charlotte (Motores eléctricos)", styles['SectionHeading']))
                charlotte_table = _build_charlotte_reference_table(
                    charlotte_catalog_pdf, styles, accent_color
                )
                if charlotte_table is not None:
                    elements.append(charlotte_table)

            doc.build(elements)



            # Restaurar estilo de la app

            if self.is_dark_mode:

                plt.style.use("dark_background")

            else:

                plt.style.use("seaborn-v0_8-whitegrid")



            if not hasattr(self, "generated_reports"):

                self.generated_reports = []

            self.generated_reports.append(pdf_path)



            self._log(f"Reporte exportado: {pdf_path}")

            self.page.snack_bar = ft.SnackBar(content=ft.Text(f"✅ Reporte PDF generado: {pdf_name}"), action="OK")

            self.page.snack_bar.open = True

            self.page.update()
            try:
                self._refresh_report_list()
            except Exception:
                pass



        except Exception as ex:

            self._log(f"Error exportando PDF: {ex}")



    def _build_analysis_view(self):

        """

        Construye la vista de análisis:

        - Selección de tiempo, FFT y señales

        - Variables auxiliares con checkbox + color + estilo

        - Periodo de análisis

        - Configuración colapsable

        - Gráfica combinada de señales seleccionadas

        """



        if self.current_df is None:

            return ft.Column(

                controls=[

                    ft.Icon(ft.Icons.WARNING_AMBER_ROUNDED, size=80, color="#e74c3c"),

                    ft.Text("No hay datos cargados para análisis", size=18),

                    ft.ElevatedButton(

                        "Ir a Archivos",

                        icon=ft.Icons.FOLDER_OPEN_ROUNDED,

                        on_click=lambda e: self._select_menu("files"),

                        style=ft.ButtonStyle(bgcolor=self._accent_ui(), color="white")

                    )

                ],

                alignment="center",

                horizontal_alignment="center",

                expand=True

            )



        # --- Detectar columnas ---

        numeric_cols = self.current_df.select_dtypes(include=np.number).columns.tolist()

        initial_time_col = "t_s" if "t_s" in numeric_cols else (numeric_cols[0] if numeric_cols else None)



        # Dropdown tiempo

        self.time_dropdown = ft.Dropdown(

            label="Tiempo",

            options=[ft.dropdown.Option(col) for col in numeric_cols],

            value=initial_time_col,

            expand=True

        )



        # Dropdown FFT

        available_signals = [col for col in numeric_cols if col != initial_time_col]

        self.fft_dropdown = ft.Dropdown(

            label="Señal FFT",

            options=[ft.dropdown.Option(col) for col in available_signals],

            value=available_signals[0] if available_signals else None,

            expand=True

        )

        # Parámetros de máquina (opcionales) para diagnóstico avanzado
        # Selección de unidad para la señal en tiempo (aceleración vs velocidad)
        self.input_signal_unit_dd = ft.Dropdown(
            label="Unidad original",
            options=[
                ft.dropdown.Option("acc_ms2", "Aceleración (m/s²)"),
                ft.dropdown.Option("acc_g", "Aceleración (g)"),
                ft.dropdown.Option("vel_ms", "Velocidad (m/s)"),
                ft.dropdown.Option("vel_mm", "Velocidad (mm/s)"),
                ft.dropdown.Option("vel_ips", "Velocidad (in/s)"),
                ft.dropdown.Option("disp_m", "Desplazamiento (m)"),
                ft.dropdown.Option("disp_mm", "Desplazamiento (mm)"),
                ft.dropdown.Option("disp_um", "Desplazamiento (µm)"),
            ],
            value=self.input_signal_unit if getattr(self, "input_signal_unit", None) else "acc_ms2",
            width=220,
            on_change=self._on_input_signal_unit_change,
        )
        self.time_unit_dd = ft.Dropdown(
            label="Señal de tiempo",
            options=[
                ft.dropdown.Option("acc", "Aceleración (m/s^2)"),
                ft.dropdown.Option("acc_g", "Aceleración (g)"),
                ft.dropdown.Option("vel_mm", "Velocidad (mm/s)"),
            ],
            value="vel_mm",
            width=220,
        )

        self.rpm_hint_field = ft.TextField(label="RPM (opc.)", value="", width=120)
        self.line_freq_dd = ft.Dropdown(label="Línea", options=[ft.dropdown.Option("50"), ft.dropdown.Option("60")], value="60", width=110)
        self.gear_teeth_field = ft.TextField(label="Dientes engrane (opc.)", value="", width=160)
        self.bpfo_field = ft.TextField(label="BPFO Hz (opc.)", value="", width=120)
        self.bpfi_field = ft.TextField(label="BPFI Hz (opc.)", value="", width=120)
        self.bsf_field = ft.TextField(label="BSF Hz (opc.)", value="", width=110)
        self.ftf_field = ft.TextField(label="FTF Hz (opc.)", value="", width=110)

        # Modo de análisis: automático parcial vs asistido (rodamientos)
        # Modo de análisis: automático parcial vs asistido (rodamientos)
        # Usa estado previo si existe (self.analysis_mode)
        self.analysis_mode_dd = ft.Dropdown(
            label="Modo de análisis",
            options=[
                ft.dropdown.Option("auto", "Automático parcial"),
                ft.dropdown.Option("assist", "Asistido (rodamientos)"),
            ],
            value=self.analysis_mode if getattr(self, 'analysis_mode', None) in ("auto", "assist") else "auto",
            width=220,
            on_change=self._on_mode_change,
        )
        # Asistido: datos de rodamiento + base opcional
        # Preseleccionar modelo si hay uno en estado
        self.bearing_model_dd = ft.Dropdown(
            label="Modelo rodamiento (opcional)",
            options=self._bearing_db_model_options(),
            width=250,
            on_change=self._on_bearing_model_change,
            value=(self.selected_bearing_model if getattr(self, 'selected_bearing_model', '') else None),
        )
        self.br_n_field = ft.TextField(label="# Elementos (n)", value="", width=150)
        self.br_d_mm_field = ft.TextField(label="d (mm)", value="", width=110)
        self.br_D_mm_field = ft.TextField(label="D (mm)", value="", width=110)
        self.br_theta_deg_field = ft.TextField(label="Ángulo (°)", value="0", width=110)
        self.assisted_box = ft.Container(
            content=ft.Column([
                ft.Text("Asistido (rodamientos)", size=14),
                ft.Row([self.bearing_model_dd, ft.TextButton("Refrescar base", on_click=lambda e: (self._load_bearing_db(), setattr(self.bearing_model_dd, 'options', self._bearing_db_model_options()), self.bearing_model_dd.update()))], spacing=10, wrap=True),
                ft.Row([self.br_n_field, self.br_d_mm_field, self.br_D_mm_field, self.br_theta_deg_field], spacing=10, wrap=True),
                ft.Row([
                    (setattr(self, 'env_bp_lo_field', ft.TextField(label="Env BP lo (Hz)", value="", width=120)) or self.env_bp_lo_field),
                    (setattr(self, 'env_bp_hi_field', ft.TextField(label="Env BP hi (Hz)", value="", width=120)) or self.env_bp_hi_field),
                ], spacing=10, wrap=True),
                ft.Row([ft.ElevatedButton("Calcular frecuencias", icon=ft.Icons.FUNCTIONS, on_click=self._compute_bearing_freqs_click)], alignment="start")
            ], spacing=8),
            visible=(self.analysis_mode == "assist"),
        )



        # Señales de tiempo

        self.signal_checkboxes = [

            ft.Checkbox(label=col, value=(col.startswith("a")))

            for col in numeric_cols if col != initial_time_col

        ]

        self.combine_signals_cb = ft.Checkbox(
            label='Unificar señales seleccionadas (RMS vectorial)',
            value=self.combine_signals_enabled,
            tooltip='Combina las señales marcadas en una sola magnitud RMS para el análisis principal.',
            on_change=self._on_combine_signals_toggle,
        )




        # 📌 Nueva gráfica principal combinada

        self.multi_chart_container = ft.Container(

            expand=True,

            content=ft.Text("Seleccione señales para graficar..."),

            bgcolor=ft.Colors.with_opacity(0.02, "white" if self.is_dark_mode else "black"),

            border_radius=10,

            padding=15,

            margin=ft.margin.only(top=10)

        )



        # Conectar checkboxes a actualización dinámica

        for cb in self.signal_checkboxes:

            cb.on_change = self._update_multi_chart



        # Variables auxiliares

        aux_cols = [col for col in numeric_cols if col not in [initial_time_col, self.fft_dropdown.value]]

        color_options = [
            ("#3498db", "Azul"),
            ("#e74c3c", "Rojo"),
            ("#2ecc71", "Verde"),
            ("#f39c12", "Naranja"),
            ("#9b59b6", "Violeta"),
        ]
        existing_colors = {c for c, _ in color_options}
        for extra_color in (self.time_plot_color, self.fft_plot_color):
            if extra_color and extra_color not in existing_colors:
                label = extra_color.upper() if extra_color.startswith('#') else extra_color
                color_options.append((extra_color, label))
                existing_colors.add(extra_color)

        style_options = [("-", "Sólida"), ("--", "Guiones"), ("-.", "Guion-punto"), (":", "Punteada")]



        self.aux_controls = []

        for col in aux_cols:

            cb = ft.Checkbox(label=col, value=True)

            color_dd = ft.Dropdown(

                options=[ft.dropdown.Option(c, t) for c, t in color_options],

                value=color_options[len(self.aux_controls) % len(color_options)][0],

                width=110

            )

            style_dd = ft.Dropdown(

                options=[ft.dropdown.Option(s, n) for s, n in style_options],

                value="-", width=110

            )

            self.aux_controls.append((cb, color_dd, style_dd))

        time_color_options = [ft.dropdown.Option(c, t) for c, t in color_options]
        fft_color_options = [ft.dropdown.Option(c, t) for c, t in color_options]
        time_color_default = self.time_plot_color if any(c == self.time_plot_color for c, _ in color_options) else color_options[0][0]
        fft_color_default = self.fft_plot_color if any(c == self.fft_plot_color for c, _ in color_options) else color_options[0][0]

        self.time_color_dd = ft.Dropdown(
            label='Color senal tiempo',
            options=time_color_options,
            value=time_color_default,
            width=200,
            on_change=self._on_time_color_change,
        )

        self.fft_color_dd = ft.Dropdown(
            label='Color FFT',
            options=fft_color_options,
            value=fft_color_default,
            width=200,
            on_change=self._on_fft_color_change,
        )

        self.fft_window_dd = ft.Dropdown(
            label="Ventana FFT",
            options=[
                ft.dropdown.Option("hann", "Hann"),
                ft.dropdown.Option("flattop", "Flat-Top"),
            ],
            value=self.fft_window_type,
            width=160,
            on_change=self._on_fft_window_change,
        )




        # Campos de periodo

        self.start_time_field = ft.TextField(label="Inicio (s)", value="0.0", width=100)

        self.end_time_field = ft.TextField(label="Fin (s)", value="", width=100)

        self.runup_start_field = ft.TextField(
            label="Inicio arranque (s)",
            value="",
            width=140,
            tooltip="Delimita el análisis específico para la cascada de arranque/paro.",
        )

        self.runup_end_field = ft.TextField(
            label="Fin arranque (s)",
            value="",
            width=140,
            tooltip="Delimita el análisis específico para la cascada de arranque/paro.",
        )

        # Opciones visuales de frecuencias en FFT
        self.hide_lf_cb = ft.Checkbox(label="Ocultar bajas frecuencias", value=True)
        self.lf_cutoff_field = ft.TextField(label="Corte LF (Hz)", value="0.5", width=100)
        self.hf_limit_field = ft.TextField(label="Máx FFT (Hz)", value="", width=120)
        orbit_options = [ft.dropdown.Option(col) for col in available_signals]
        default_orbit_x = (
            self.orbit_axis_x_pref
            if self.orbit_axis_x_pref in available_signals
            else (available_signals[0] if available_signals else None)
        )
        default_orbit_y = (
            self.orbit_axis_y_pref
            if self.orbit_axis_y_pref in available_signals
            else (
                available_signals[1]
                if len(available_signals) > 1
                else (available_signals[0] if available_signals else None)
            )
        )
        if default_orbit_y == default_orbit_x and len(available_signals) > 1:
            for candidate in available_signals:
                if candidate != default_orbit_x:
                    default_orbit_y = candidate
                    break
        self._remember_orbit_axis("x", default_orbit_x)
        self._remember_orbit_axis("y", default_orbit_y)
        self.runup_3d_cb = ft.Checkbox(
            label="Arranque/paro 3D",
            value=self.runup_3d_enabled,
            tooltip="Agrega cascada 3D para análisis de arranque y paro",
            on_change=self._on_runup_3d_toggle,
        )
        orbit_disabled = (not self.orbit_plot_enabled) or (len(available_signals) == 0)
        self.orbit_cb = ft.Checkbox(
            label="Análisis de órbita",
            value=self.orbit_plot_enabled,
            tooltip="Genera la órbita X-Y del rotor para evaluar restricciones de movimiento",
            on_change=self._on_orbit_toggle,
        )
        self.orbit_x_dd = ft.Dropdown(
            label="Órbita eje X",
            options=orbit_options,
            value=default_orbit_x,
            width=200,
            on_change=self._on_orbit_axis_change,
            disabled=orbit_disabled,
        )
        self.orbit_y_dd = ft.Dropdown(
            label="Órbita eje Y",
            options=orbit_options,
            value=default_orbit_y,
            width=200,
            on_change=self._on_orbit_axis_change,
            disabled=orbit_disabled,
        )
        if self.orbit_period_seconds is not None:
            period_value_txt = f"{self.orbit_period_seconds:.2f}".rstrip("0").rstrip(".")
        else:
            period_value_txt = ""
        self.orbit_period_field = ft.TextField(
            label="Periodo órbita (0.0 a 1.0)",
            value=period_value_txt,
            width=150,
            tooltip="Duración máxima en segundos usada para la gráfica de órbita (vacío = todo el rango).",
            on_change=self._on_orbit_period_change,
            disabled=orbit_disabled,
        )
        self.fft_zoom_text = ft.Text("Zoom FFT: completo", size=12)
        self.fft_zoom_slider = ft.RangeSlider(
            0.0,
            1.0,
            min=0.0,
            max=1.0,
            divisions=100,
            on_change=self._on_fft_zoom_preview,
            on_change_end=self._on_fft_zoom_commit,
            disabled=True,
            expand=True,
        )
        # Escala en dBV real (re 1 V) y parámetros de calibración
        self.db_scale_cb = ft.Checkbox(label="Ver FFT en dBV (re 1 V)", value=False)
        # Parámetros de calibración para convertir a Voltios
        self.sens_unit_dd = ft.Dropdown(
            label="Tipo de sensor",
            options=[
                ft.dropdown.Option("mV/g", "Acelerómetro (mV/g)"),
                ft.dropdown.Option("V/g", "Acelerómetro (V/g)"),
                ft.dropdown.Option("mV/(mm/s)", "Velocímetro (mV/(mm/s))"),
                ft.dropdown.Option("V/(mm/s)", "Velocímetro (V/(mm/s))"),
            ],
            value="mV/g",
            width=180,
        )
        self.sensor_sens_field = ft.TextField(label="Sensibilidad", value="100", width=120, tooltip="p.ej. 100 mV/g o 10 mV/(mm/s)")
        self.gain_field = ft.TextField(label="Ganancia (V/V)", value="1.0", width=120)
        # Campos de rango Y para dB/dBV
        self.db_ref_field = ft.TextField(label="Ref. dB (genérico)", value="1.0", width=140, tooltip="Solo para dB genérico; dBV usa 1 V por definición.")
        self.db_ymin_field = ft.TextField(label="Y mín (dB)", value="", width=100)
        self.db_ymax_field = ft.TextField(label="Y máx (dB)", value="", width=100)
        # Recalcular al cambiar estas opciones
        self.start_time_field.on_change = self._update_analysis
        self.end_time_field.on_change = self._update_analysis
        self.runup_start_field.on_change = self._update_analysis
        self.runup_end_field.on_change = self._update_analysis
        self.hide_lf_cb.on_change = self._update_analysis
        self.lf_cutoff_field.on_change = self._update_analysis
        self.db_scale_cb.on_change = self._update_analysis
        self.sens_unit_dd.on_change = self._update_analysis
        self.sensor_sens_field.on_change = self._update_analysis
        self.gain_field.on_change = self._update_analysis
        self.db_ref_field.on_change = self._update_analysis
        self.db_ymin_field.on_change = self._update_analysis
        self.db_ymax_field.on_change = self._update_analysis
        self.hf_limit_field.on_change = self._update_analysis



        # --- Contenedor de configuración ---

        self.config_expanded = True

        general_settings = ft.Column(
            [
                ft.Text("Columnas base", size=14, weight="bold"),
                ft.Row([self.time_dropdown, self.fft_dropdown], spacing=10),
                ft.Text("Unidades y colores", size=14, weight="bold"),
                ft.Row([self.input_signal_unit_dd, self.time_unit_dd], spacing=10, wrap=True),
                ft.Row([self.time_color_dd, self.fft_color_dd], spacing=10),
            ],
            spacing=12,
            tight=True,
        )

        signal_settings = ft.Column(
            [
                ft.Text("📊 Señales en tiempo", size=14, weight="bold"),
                ft.Row(self.signal_checkboxes, wrap=True, spacing=10),
                self.combine_signals_cb,
                ft.Text("📌 Variables auxiliares", size=14, weight="bold"),
                ft.Column([
                    ft.Row([cb, color_dd, style_dd], spacing=10)
                    for cb, color_dd, style_dd in self.aux_controls
                ], spacing=6),
            ],
            spacing=12,
            tight=True,
        )

        spectrum_settings = ft.Column(
            [
                ft.Text("⏱️ Periodo de análisis", size=14, weight="bold"),
                ft.Row([self.start_time_field, self.end_time_field], spacing=10, wrap=True),
                ft.Text("🚀 Periodo arranque/paro", size=14, weight="bold"),
                ft.Row([self.runup_start_field, self.runup_end_field], spacing=10, wrap=True),
                ft.Text("Opciones de espectro", size=14, weight="bold"),
                ft.Row([self.hide_lf_cb, self.lf_cutoff_field, self.hf_limit_field, self.fft_window_dd, self.runup_3d_cb], spacing=10, wrap=True),
                ft.Row([self.orbit_cb, self.orbit_x_dd, self.orbit_y_dd, self.orbit_period_field], spacing=10, wrap=True),
                ft.Column([self.fft_zoom_text, self.fft_zoom_slider], spacing=4),
                ft.Text("Escala y calibración", size=14, weight="bold"),
                ft.Row([self.db_scale_cb, self.sens_unit_dd, self.sensor_sens_field, self.gain_field], spacing=10, wrap=True),
                ft.Row([self.db_ref_field, self.db_ymin_field, self.db_ymax_field], spacing=10, wrap=True),
            ],
            spacing=12,
            tight=True,
        )

        diagnosis_settings = ft.Column(
            [
                ft.Text("Parámetros de máquina", size=14, weight="bold"),
                ft.Row([
                    self.analysis_mode_dd,
                    self.rpm_hint_field,
                    self.line_freq_dd,
                    self.gear_teeth_field,
                    ft.OutlinedButton("Rodamientos", icon=ft.Icons.LIST_ALT_ROUNDED, on_click=self._goto_bearings_view),
                ], spacing=10, wrap=True),
                self.assisted_box,
                ft.Row([self.bpfo_field, self.bpfi_field, self.bsf_field, self.ftf_field], spacing=10, wrap=True),
            ],
            spacing=12,
            tight=True,
        )

        config_tab_wrappers = {
            "inputs": ft.Container(content=general_settings, padding=10),
            "signals": ft.Container(content=signal_settings, padding=10),
            "spectrum": ft.Container(content=spectrum_settings, padding=10),
            "diagnostics": ft.Container(content=diagnosis_settings, padding=10),
        }

        tab_definitions = [
            ("inputs", "Entradas", ft.Icons.TUNE_ROUNDED),
            ("signals", "Señales", ft.Icons.SHOW_CHART_ROUNDED),
            ("spectrum", "Espectro", ft.Icons.GRAPHIC_EQ_ROUNDED),
            ("diagnostics", "Diagnóstico", ft.Icons.MEDICAL_SERVICES_ROUNDED),
        ]

        self.config_tab_views = config_tab_wrappers
        self.config_tab_keys = [key for key, *_ in tab_definitions]
        self.active_config_tab = self.config_tab_keys[0]

        self.config_tabs = ft.Tabs(
            animation_duration=250,
            selected_index=0,
            tabs=[ft.Tab(text=label, icon=icon) for key, label, icon in tab_definitions],
            on_change=self._on_config_tab_change,
        )

        self.config_tab_body = ft.Container(
            content=self.config_tab_views[self.active_config_tab],
            padding=ft.padding.only(top=6),
        )

        action_buttons = ft.Row(
            alignment="center",
            spacing=20,
            controls=[
                ft.ElevatedButton(
                    "Generar",
                    icon=ft.Icons.ANALYTICS_ROUNDED,
                    on_click=self._update_analysis,
                    style=ft.ButtonStyle(bgcolor=self._accent_ui(), color="white"),
                ),
                ft.OutlinedButton(
                    "Exportar",
                    icon=ft.Icons.DOWNLOAD_ROUNDED,
                    on_click=self.exportar_pdf,
                ),
            ],
        )

        self.config_container = ft.Container(
            content=ft.Column(
                [
                    self.config_tabs,
                    self.config_tab_body,
                    action_buttons,
                ],
                spacing=16,
            ),
            visible=self.config_expanded,
        )



        # Botón de colapsar

        toggle_btn = ft.IconButton(

            icon=ft.Icons.ARROW_DROP_DOWN_CIRCLE if self.config_expanded else ft.Icons.ARROW_RIGHT,

            tooltip="Mostrar/Ocultar configuración",

            on_click=self._toggle_config_panel

        )



        # Panel final

        controls_panel = ft.Container(

            padding=20,

            border_radius=15,

            bgcolor=ft.Colors.with_opacity(0.05, self._accent_ui()),

            content=ft.Column([

                ft.Row([

                    ft.Text("⚙️ Configuración de Gráficos", weight="bold", size=18),

                    toggle_btn

                ], alignment="spaceBetween"),

                self.config_container

            ], spacing=10)

        )



        # Contenedor de gráficas

        initial_loading = ft.Column(

            [ft.ProgressRing(), ft.Text("Preparando análisis...")],

            alignment="center",

            expand=True,

        )

        self.chart_container = ft.Container(

            expand=True,

            content=self._wrap_chart_with_notice(initial_loading),

            bgcolor=ft.Colors.with_opacity(0.02, "white" if self.is_dark_mode else "black"),

            border_radius=15,

            padding=20,

            margin=ft.margin.only(top=20)

        )

        if not self.interactive_charts_enabled and not self._interactive_notice_logged:

            self._log(

                "Las gráficas interactivas requieren WebView y no están disponibles en esta plataforma; "

                "se mostrarán gráficos estáticos."

            )

            self._interactive_notice_logged = True



        # Si venimos desde 'Usar en análisis', activar asistido y aplicar modelo seleccionado
        try:
            if getattr(self, 'analysis_mode', 'auto') == 'assist':
                try:
                    self.assisted_box.visible = True
                    self.assisted_box.update() if self.assisted_box.page else None
                except Exception:
                    pass
                self._on_bearing_model_change()
        except Exception:
            pass

        self.page.run_task(self._update_analysis_async)



        return ft.Column(

            controls=[

                ft.Text("Análisis FFT y Diagnóstico", size=24, weight="bold"),

                controls_panel,

                self.chart_container         

            ],

            expand=True,

            spacing=10,

            scroll="auto"

        )

    

    async def _update_analysis_async(self, e=None):

        if self.chart_container:

            loading_view = ft.Column(

                [ft.ProgressRing(), ft.Text("Generando análisis FFT...")],

                horizontal_alignment="center",

                alignment="center",

                expand=True

            )

            self.chart_container.content = self._wrap_chart_with_notice(loading_view)

            if self.chart_container.page:

                self.chart_container.update()

            await asyncio.sleep(0.1)



            new_chart = self._create_plot()



            self.chart_container.content = self._wrap_chart_with_notice(new_chart)

            if self.chart_container.page:

                self.chart_container.update()



    def _update_analysis(self, e=None):
        self.page.run_task(self._update_analysis_async)
        try:
            self._update_multi_chart()
        except Exception:
            pass

    def _goto_bearings_view(self, e=None):
        try:
            self._select_menu("bearings", force_rebuild=True)
        except Exception:
            pass



    def _calculate_rms(self, signal):
        """
        Calcula el valor RMS de una señal en el dominio del tiempo, removiendo
        componentes DC y tendencia lineal para evitar sesgos subsíncronos.
        """
        sig = np.asarray(signal, dtype=float).ravel()
        if sig.size == 0:
            return 0.0
        sig = sig[np.isfinite(sig)]
        if sig.size == 0:
            return 0.0
        sig = sig - np.mean(sig)
        if sig.size > 3:
            x = np.arange(sig.size, dtype=float)
            try:
                p = np.polyfit(x, sig, 1)
                sig = sig - (p[0] * x + p[1])
            except Exception:
                pass
        return float(np.sqrt(np.mean(sig**2))) if sig.size else 0.0

    def _format_peak_label(self, freq_hz: float, amp_mm_s: float, order: float | None = None, unit: str = "mm/s") -> str:
        label = f"{freq_hz:.2f} Hz | {amp_mm_s:.3f} {unit}"
        try:
            if order is not None and np.isfinite(order):
                label += f" | {float(order):.2f}X"
        except Exception:
            pass
        return label

    def _place_annotations(self, ax, points: List[Tuple[float, float]], labels: List[str], color: str = "#e74c3c", text_color: str | None = None):
        try:
            if not points or not labels:
                return
            text_color = text_color or ("white" if self.is_dark_mode else "black")
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            x_span = xmax - xmin if xmax > xmin else max(abs(xmax), 1.0)
            y_span = ymax - ymin if ymax > ymin else max(abs(ymax), 1.0)
            y_step = 0.04 * y_span if y_span > 0 else 1.0
            x_offset = 0.02 * x_span if x_span > 0 else 0.5
            placements: List[Tuple[float, float]] = []
            bg_color = "#1b1f24" if self.is_dark_mode else "white"
            for (x, y), label in zip(points, labels):
                try:
                    x = float(x)
                    y = float(y)
                except Exception:
                    continue
                tx = x + x_offset
                align = "left"
                if tx > xmax:
                    tx = x - x_offset
                    align = "right"
                ty = y + y_step
                attempts = 0
                while placements and any(abs(ty - py) < 0.8 * y_step for _, py in placements) and attempts < 20:
                    ty += y_step
                    attempts += 1
                    if ty > ymax:
                        ty = y - y_step
                if ty < ymin:
                    ty = ymin + 0.05 * y_span
                ax.annotate(
                    label,
                    xy=(x, y),
                    xytext=(tx, ty),
                    textcoords="data",
                    ha=align,
                    va="bottom" if ty >= y else "top",
                    fontsize=8,
                    color=text_color,
                    bbox=dict(boxstyle="round,pad=0.2", fc=bg_color, ec="none", alpha=0.8),
                    arrowprops=dict(arrowstyle="->", lw=0.8, color=color),
                    zorder=6,
                    clip_on=False,
                )
                placements.append((tx, ty))
        except Exception:
            pass

    def _draw_frequency_markers(self, ax, marks: List[Tuple[float, str, str]], zoom_range: Optional[Tuple[float, float]] = None):
        try:
            if not marks:
                return
            xmin, xmax = ax.get_xlim()
            x_span = xmax - xmin if xmax > xmin else max(abs(xmax), 1.0)
            transform = ax.get_xaxis_transform()
            used: List[Tuple[float, float]] = []
            base_y = 0.98
            step = 0.08
            bg_color = "#1b1f24" if self.is_dark_mode else "white"
            freq_tol = 0.025 * x_span if x_span > 0 else 1.0
            offset_values = [0.0, 0.012 * x_span, -0.012 * x_span, 0.024 * x_span, -0.024 * x_span]
            for freq, label, color in marks:
                try:
                    freq = float(freq)
                except Exception:
                    continue
                if freq <= 0:
                    continue
                if zoom_range and (freq < zoom_range[0] or freq > zoom_range[1]):
                    continue
                slot = base_y
                attempts = 0
                offset_idx = 0
                freq_adj = freq
                while used and any(
                    abs(slot - other_y) < 0.05 and abs(freq_adj - other_x) < freq_tol
                    for other_x, other_y in used
                ) and attempts < 40:
                    slot -= step
                    attempts += 1
                    if slot < base_y - 4 * step:
                        slot = base_y
                        offset_idx = (offset_idx + 1) % len(offset_values)
                        freq_adj = freq + offset_values[offset_idx]
                ax.text(
                    freq_adj,
                    slot,
                    label,
                    rotation=90,
                    color=color,
                    fontsize=8,
                    va="top",
                    ha="center",
                    transform=transform,
                    bbox=dict(boxstyle="round,pad=0.15", fc=bg_color, ec="none", alpha=0.75),
                    clip_on=False,
                )
                used.append((freq_adj, slot))
        except Exception:
            pass

    def _generate_runup_3d_figure(
        self,
        t_segment: np.ndarray,
        signal_segment: np.ndarray,
        fc: float,
        hide_lf: bool,
        fmax_ui: Optional[float],
        zoom_range: Optional[Tuple[float, float]],
        dark_mode: bool,
        fallback_time: Optional[np.ndarray] = None,
        fallback_signal: Optional[np.ndarray] = None,
        window_type: Optional[str] = None,
    ):
        try:
            primary_t = np.asarray(t_segment, dtype=float).ravel()
            primary_y = np.asarray(signal_segment, dtype=float).ravel()
            fallback_t = (
                np.asarray(fallback_time, dtype=float).ravel()
                if fallback_time is not None
                else None
            )
            fallback_y = (
                np.asarray(fallback_signal, dtype=float).ravel()
                if fallback_signal is not None
                else None
            )
            min_samples = 256
            t = primary_t
            y = primary_y
            if t.size < min_samples or y.size != t.size:
                if (
                    fallback_t is not None
                    and fallback_y is not None
                    and fallback_t.size == fallback_y.size
                    and fallback_t.size >= min_samples
                ):
                    t = fallback_t
                    y = fallback_y
                else:
                    min_samples = 128
                    if t.size < min_samples or y.size != t.size:
                        return None
            dt = float(np.median(np.diff(t)))
            if not (np.isfinite(dt) and dt > 0):
                return None
            n = y.size
            approx = min(n, 2048)
            if approx < 128:
                return None
            power = int(np.floor(np.log2(max(128, approx))))
            nfft = int(2 ** power)
            nfft = min(nfft, n)
            if nfft < 128:
                return None
            window, cg = _build_fft_window(window_type, nfft)
            if window.size != nfft:
                window = np.hanning(nfft)
                cg = float(np.mean(window)) if np.any(window) else 1.0
                if not np.isfinite(cg) or abs(cg) < 1e-12:
                    cg = 1.0
            step = max(1, int(nfft * 0.25))
            if step >= nfft:
                step = max(1, nfft // 4)
            spectra: List[np.ndarray] = []
            times: List[float] = []
            freq_axis = None
            for start in range(0, n - nfft + 1, step):
                segment = y[start:start + nfft]
                windowed = segment * window
                fft_vals = np.fft.rfft(windowed)
                mag_acc = np.abs(fft_vals) / (nfft * cg)
                if nfft % 2 == 0 and mag_acc.size >= 2:
                    mag_acc[1:-1] *= 2.0
                elif mag_acc.size >= 2:
                    mag_acc[1:] *= 2.0
                if freq_axis is None:
                    freq_axis = np.fft.rfftfreq(nfft, dt)
                vel_spec = np.zeros_like(mag_acc)
                pos = freq_axis > 0
                vel_spec[pos] = (mag_acc[pos] / (2.0 * np.pi * freq_axis[pos])) * 1000.0
                spectra.append(vel_spec)
                seg_t = t[start:start + nfft]
                times.append(float(np.mean(seg_t)))
            if not spectra or freq_axis is None:
                return None
            spec_arr = np.vstack(spectra)
            times_arr = np.asarray(times, dtype=float)
            freq_mask = np.ones_like(freq_axis, dtype=bool)
            if hide_lf:
                freq_mask &= freq_axis >= max(0.0, fc)
            if fmax_ui and fmax_ui > 0:
                freq_mask &= freq_axis <= float(fmax_ui)
            if zoom_range and zoom_range[1] > zoom_range[0]:
                freq_mask &= (freq_axis >= zoom_range[0]) & (freq_axis <= zoom_range[1])
            if not np.any(freq_mask):
                freq_mask = np.ones_like(freq_axis, dtype=bool)
            freq_sel = freq_axis[freq_mask]
            amp = spec_arr[:, freq_mask].T  # freq x time
            if freq_sel.size == 0 or amp.size == 0:
                return None
            freq_scale = getattr(self, "_fft_display_scale", 1.0) or 1.0
            if not np.isfinite(freq_scale) or freq_scale <= 0:
                freq_scale = 1.0
            freq_unit = getattr(self, "_fft_display_unit", "Hz") or "Hz"
            freq_sel_disp = freq_sel / freq_scale
            fig = plt.figure(figsize=(10, 6))
            face = "#0f141b" if dark_mode else "white"
            fig.patch.set_facecolor(face)
            ax = fig.add_subplot(111, projection="3d")
            ax.set_facecolor(face)
            T, F = np.meshgrid(times_arr, freq_sel_disp)
            surf = ax.plot_surface(T, F, amp, cmap="viridis", linewidth=0, antialiased=True)
            ax.set_xlabel("Tiempo (s)")
            ax.set_ylabel(f"Frecuencia ({freq_unit})")
            ax.set_zlabel("Velocidad [mm/s]")
            ax.set_title("Arranque/Paro - Cascada 3D")
            try:
                vmax = float(np.nanmax(amp))
                if np.isfinite(vmax) and vmax > 0:
                    ax.set_zlim(0.0, vmax * 1.05)
            except Exception:
                pass
            ax.view_init(elev=32, azim=-130)
            axis_color = "white" if dark_mode else "black"
            ax.xaxis.label.set_color(axis_color)
            ax.yaxis.label.set_color(axis_color)
            ax.zaxis.label.set_color(axis_color)
            ax.title.set_color(axis_color)
            for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
                for tick in axis.get_ticklabels():
                    tick.set_color(axis_color)
            fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label="Velocidad [mm/s]")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                fig.tight_layout()
            return fig
        except Exception:
            return None

    def _generate_orbit_figure(
        self,
        t_segment: np.ndarray,
        x_segment: np.ndarray,
        y_segment: np.ndarray,
        x_label: str,
        y_label: str,
        fc: float,
        hide_lf: bool,
        fmax_ui: Optional[float],
        dark_mode: bool,
    ):
        try:
            t = np.asarray(t_segment, dtype=float).ravel()
            x = np.asarray(x_segment, dtype=float).ravel()
            y = np.asarray(y_segment, dtype=float).ravel()
            if t.size < 32 or x.size != t.size or y.size != t.size:
                return None
            valid = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
            if np.count_nonzero(valid) < 32:
                return None
            t = t[valid]
            x = x[valid]
            y = y[valid]
            dt = None
            if t.size > 1:
                try:
                    dt_val = float(np.median(np.diff(t)))
                    if np.isfinite(dt_val) and dt_val > 0:
                        dt = dt_val
                except Exception:
                    dt = None

            def _band_filter(arr: np.ndarray) -> np.ndarray:
                base = np.asarray(arr, dtype=float)
                base = base - np.nanmean(base)
                base = np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)
                if dt is None or base.size < 32:
                    return base
                spec = np.fft.rfft(base)
                freqs = np.fft.rfftfreq(base.size, dt)
                if hide_lf and fc and fc > 0:
                    spec[freqs < max(0.0, float(fc))] = 0
                if fmax_ui and fmax_ui > 0:
                    spec[freqs > float(fmax_ui)] = 0
                try:
                    filtered = np.fft.irfft(spec, n=base.size)
                except Exception:
                    filtered = base
                return filtered

            x_filt = _band_filter(x)
            y_filt = _band_filter(y)
            finite = np.isfinite(x_filt) & np.isfinite(y_filt)
            if np.count_nonzero(finite) < 16:
                return None
            x_filt = x_filt[finite]
            y_filt = y_filt[finite]
            if x_filt.size > 6000:
                idx = np.linspace(0, x_filt.size - 1, 3000, dtype=int)
                x_filt = x_filt[idx]
                y_filt = y_filt[idx]
            if x_filt.size < 16 or y_filt.size < 16:
                return None
            stack = np.vstack((x_filt, y_filt))
            radial_source = np.hypot(x_filt, y_filt)
            x_plot = x_filt
            y_plot = y_filt
            overlay_original = False
            balance_note = None
            corr_val = None
            std_x = None
            std_y = None
            eig_ratio = None
            eigvals_raw = None
            eigvecs = None
            try:
                std_x = float(np.nanstd(x_filt))
                std_y = float(np.nanstd(y_filt))
            except Exception:
                std_x = std_y = None
            try:
                corr_val = float(np.corrcoef(x_filt, y_filt)[0, 1])
            except Exception:
                corr_val = None
            try:
                cov = np.cov(stack)
                eigvals_raw, eigvecs = np.linalg.eigh(cov)
                order = np.argsort(eigvals_raw)[::-1]
                eigvals_raw = eigvals_raw[order]
                eigvecs = eigvecs[:, order]
                if eigvals_raw[0] > 0:
                    eig_ratio = float(eigvals_raw[0] / max(eigvals_raw[-1], 1e-18))
            except Exception:
                eigvals_raw = None
                eigvecs = None
            degenerate = False
            if std_x is not None and std_y is not None and std_x >= 0 and std_y >= 0:
                if std_x < 1e-9 or std_y < 1e-9:
                    degenerate = True
            if not degenerate and corr_val is not None and np.isfinite(corr_val):
                if abs(corr_val) > 0.985:
                    degenerate = True
            if not degenerate and eig_ratio is not None and np.isfinite(eig_ratio):
                if eig_ratio > 80.0:
                    degenerate = True
            if degenerate and eigvals_raw is not None and eigvecs is not None:
                centered = stack - np.mean(stack, axis=1, keepdims=True)
                safe = np.sqrt(np.maximum(eigvals_raw, 1e-18))
                transform = eigvecs @ np.diag(1.0 / safe) @ eigvecs.T
                balanced = transform @ centered
                x_plot = balanced[0]
                y_plot = balanced[1]
                overlay_original = True
                balance_note = "Órbita auto-equilibrada (señales muy correlacionadas)"
            face = "#0f141b" if dark_mode else "white"
            fig, ax = plt.subplots(figsize=(6, 6))
            fig.patch.set_facecolor(face)
            ax.set_facecolor(face)
            accent = self._accent_ui()
            if overlay_original:
                ax.plot(
                    x_filt,
                    y_filt,
                    color="#95a5a6",
                    linewidth=0.9,
                    alpha=0.5,
                    label="Original (sin balance)",
                )
            ax.plot(x_plot, y_plot, color=accent, linewidth=1.4, alpha=0.9)
            progress = np.linspace(0.0, 1.0, x_plot.size, dtype=float)
            sc = ax.scatter(
                x_plot,
                y_plot,
                c=progress,
                cmap="plasma",
                s=10,
                alpha=0.6,
                linewidths=0,
            )
            radial = np.hypot(x_plot, y_plot)
            try:
                span = float(np.nanpercentile(np.concatenate((np.abs(x_plot), np.abs(y_plot))), 99.5))
            except Exception:
                span = float(np.nanmax(radial)) if radial.size else 0.0
            if not np.isfinite(span) or span <= 0:
                span = float(np.nanmax(radial)) if radial.size else 1.0
            lim = float(span) * 1.1 if span > 0 else 1.0
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            peak_idx = np.array([], dtype=int)
            if radial.size >= 16:
                top_n = min(3, radial.size)
                peak_idx = np.argpartition(radial, -top_n)[-top_n:]
                peak_idx = peak_idx[np.argsort(radial[peak_idx])[::-1]]
                ax.scatter(
                    x_plot[peak_idx],
                    y_plot[peak_idx],
                    color="#f9ca24",
                    edgecolors="#2c3e50",
                    linewidths=0.6,
                    marker="*",
                    s=90,
                    label="Picos máximos",
                    zorder=6,
                )
            try:
                ax.scatter([x_plot[0]], [y_plot[0]], color="#2ecc71", s=50, label="Inicio")
                ax.scatter([x_plot[-1]], [y_plot[-1]], color="#e74c3c", s=50, label="Fin")
            except Exception:
                pass
            ax.set_title("Análisis de órbita")
            ax.set_xlabel(x_label or "Canal X")
            ax.set_ylabel(y_label or "Canal Y")
            ax.set_aspect("equal")
            axis_color = "white" if dark_mode else "black"
            ax.axhline(0.0, color=axis_color, linewidth=0.8, alpha=0.2)
            ax.axvline(0.0, color=axis_color, linewidth=0.8, alpha=0.2)
            if peak_idx.size:
                for idx_peak in peak_idx:
                    raw_val = float(radial_source[idx_peak]) if idx_peak < radial_source.size else float(radial[idx_peak])
                    ax.text(
                        x_plot[idx_peak],
                        y_plot[idx_peak],
                        f"{raw_val:.2f}",
                        color=axis_color,
                        fontsize=8,
                        ha="left",
                        va="bottom",
                    )
            grid_color = "#34495e" if dark_mode else "#bdc3c7"
            ax.grid(True, linestyle="--", alpha=0.25 if dark_mode else 0.35, color=grid_color)
            ax.xaxis.label.set_color(axis_color)
            ax.yaxis.label.set_color(axis_color)
            ax.title.set_color(axis_color)
            for axis in [ax.xaxis, ax.yaxis]:
                for tick in axis.get_ticklabels():
                    tick.set_color(axis_color)
            info_lines = []
            if corr_val is not None and np.isfinite(corr_val):
                info_lines.append(f"ρ(X,Y) = {corr_val:.3f}")
            if std_x is not None and std_y is not None and std_y > 0:
                info_lines.append(f"σx/σy = {(std_x / std_y):.2f}")
            if eig_ratio is not None and np.isfinite(eig_ratio):
                info_lines.append(f"κ = {eig_ratio:.1f}")
            if info_lines:
                ax.text(
                    0.02,
                    0.98,
                    "\n".join(info_lines),
                    transform=ax.transAxes,
                    fontsize=8,
                    color=axis_color,
                    va="top",
                )
            if balance_note:
                ax.text(
                    0.02,
                    0.02,
                    balance_note,
                    transform=ax.transAxes,
                    fontsize=8,
                    color=axis_color,
                    va="bottom",
                    bbox=dict(
                        boxstyle="round,pad=0.25",
                        facecolor="#1b2633" if dark_mode else "white",
                        alpha=0.3,
                        edgecolor="none",
                    ),
                )
            cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.015)
            cbar.set_label("Progreso temporal")
            if dark_mode:
                cbar.ax.yaxis.label.set_color("white")
                for tick in cbar.ax.get_yticklabels():
                    tick.set_color("white")
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="upper right", fontsize=8)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                fig.tight_layout()
            return fig
        except Exception:
            return None

    def _format_fft_zoom_label(self, start: float, end: float, full_range: Tuple[float, float]) -> str:
        min_val, max_val = full_range
        scale = getattr(self, "_fft_display_scale", 1.0) or 1.0
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        unit = getattr(self, "_fft_display_unit", "Hz") or "Hz"
        precision = 3 if scale >= 1000.0 else 1
        fmt = "{:." + str(precision) + "f}"
        min_txt = fmt.format(min_val / scale)
        max_txt = fmt.format(max_val / scale)
        start_txt = fmt.format(start / scale)
        end_txt = fmt.format(end / scale)
        if abs(start - min_val) <= 1e-6 and abs(end - max_val) <= 1e-6:
            return f"Zoom FFT: completo ({min_txt} - {max_txt} {unit})"
        return f"Zoom FFT: {start_txt} - {end_txt} {unit}"

    def _apply_fft_display_units(
        self,
        full_range: Optional[Tuple[float, float]],
        zoom_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Selecciona Hz o kHz según el rango visible en la FFT."""

        freq_max: Optional[float] = None
        for rng in (zoom_range, full_range):
            if rng and len(rng) == 2 and rng[1] > rng[0] and np.isfinite(rng[1]):
                freq_max = float(rng[1])
                break
        if freq_max is not None and freq_max >= 1500.0:
            self._fft_display_scale = 1000.0
            self._fft_display_unit = "kHz"
        else:
            self._fft_display_scale = 1.0
            self._fft_display_unit = "Hz"

    def _update_fft_zoom_controls(self, full_range: Optional[Tuple[float, float]], current_range: Optional[Tuple[float, float]]):
        slider = getattr(self, 'fft_zoom_slider', None)
        label = getattr(self, 'fft_zoom_text', None)
        if not slider and not label:
            return
        self._fft_zoom_syncing = True
        try:
            if not full_range or full_range[1] <= full_range[0]:
                if slider:
                    slider.disabled = True
                    slider.start_value = 0.0
                    slider.end_value = 1.0
                    if slider.page:
                        slider.update()
                if label:
                    label.value = 'Zoom FFT: sin datos'
                    if label.page:
                        label.update()
                return
            min_val, max_val = full_range
            if current_range is None:
                start_val, end_val = min_val, max_val
            else:
                start_val = max(min_val, min(current_range[0], max_val))
                end_val = max(start_val + 1e-6, min(current_range[1], max_val))
            self._apply_fft_display_units(full_range, (start_val, end_val))
            if slider:
                slider.min = min_val
                slider.max = max_val
                slider.start_value = start_val
                slider.end_value = end_val
                slider.disabled = False
                if slider.page:
                    slider.update()
            if label:
                label.value = self._format_fft_zoom_label(start_val, end_val, (min_val, max_val))
                if label.page:
                    label.update()
        finally:
            self._fft_zoom_syncing = False

    def _on_fft_zoom_preview(self, e):
        if self._fft_zoom_syncing:
            return
        try:
            slider = e.control
            start = float(slider.start_value)
            end = float(slider.end_value)
        except Exception:
            return
        label = getattr(self, 'fft_zoom_text', None)
        if label and self._fft_full_range:
            self._apply_fft_display_units(self._fft_full_range, (start, end))
            label.value = self._format_fft_zoom_label(start, end, self._fft_full_range)
            if label.page:
                label.update()

    def _on_fft_zoom_commit(self, e):
        if self._fft_zoom_syncing:
            return
        slider = e.control
        if not self._fft_full_range:
            return
        try:
            start = float(slider.start_value)
            end = float(slider.end_value)
        except Exception:
            return
        full_start, full_end = self._fft_full_range
        tol = max(1e-6, 0.002 * max(full_end - full_start, 1.0))
        if end <= start + tol:
            end = min(full_end, start + tol)
        if abs(start - full_start) <= tol and abs(end - full_end) <= tol:
            new_range = None
        else:
            new_range = (start, end)
        if new_range == self._fft_zoom_range or (new_range is None and self._fft_zoom_range is None):
            return
        self._fft_zoom_range = new_range
        self._update_fft_zoom_controls(self._fft_full_range, self._fft_zoom_range)
        self._update_analysis()

    def _select_main_findings(self, findings: List[str], max_items: int = 2) -> List[str]:
        """
        Selecciona hallazgos principales para motores eléctricos (prioriza: Eléctrico, Desalineación,
        Desbalanceo, Rodamientos, Engranes, Resonancia). Excluye la línea de severidad ISO.
        """
        if not findings:
            return []
        try:
            items = [f for f in findings if not str(f).startswith("Severidad ISO:")]
        except Exception:
            items = findings[:]
        order = ["Eléctrico", "Eléctrico", "El?ctrico", "Desalineaci", "Desbalanceo", "Rodamientos", "Engranes", "Resonancia estructural"]
        selected: List[str] = []
        for key in order:
            for f in items:
                try:
                    if key in f and f not in selected:
                        selected.append(f)
                        if len(selected) >= max_items:
                            return selected
                except Exception:
                    continue
        for f in items:
            if f not in selected:
                selected.append(f)
                if len(selected) >= max_items:
                    break
        return selected

    def _build_explanations(self, res: Dict[str, Any], findings: List[str]) -> List[str]:
        """Genera una única revisión con motivo y acción alineados a ISO."""

        def _normalize(text: str) -> str:
            if text is None:
                return ""
            normalized = unicodedata.normalize("NFD", str(text))
            return normalized.encode("ascii", "ignore").decode("ascii").lower()

        explanations: List[str] = []
        try:
            severity = res.get("severity", {}) if isinstance(res, dict) else {}
            iso_label = str(severity.get("label", "Sin clasificación ISO"))
            try:
                rms_val = float(severity.get("rms_mm_s", 0.0))
                rms_txt = f"{rms_val:.3f} mm/s"
            except Exception:
                rms_val = None
                rms_txt = "N/D"

            iso_contexts = [
                {"key": "Buena", "zone": "Zona A", "range": "≤ 2.8 mm/s"},
                {"key": "Satisfactoria", "zone": "Zona B", "range": "2.8 – 4.5 mm/s"},
                {"key": "Insatisfactoria", "zone": "Zona C", "range": "4.5 – 7.1 mm/s"},
                {"key": "Inaceptable", "zone": "Zona D", "range": "> 7.1 mm/s"},
            ]
            zone_clause = None
            for ctx in iso_contexts:
                if ctx["key"].lower() in iso_label.lower():
                    zone_clause = f"{ctx['zone']} ({ctx['range']})"
                    break
            severity_clause = iso_label
            if zone_clause:
                severity_clause = f"{iso_label} – {zone_clause}"
            if rms_val is not None:
                severity_clause = f"{severity_clause} con {rms_txt}"

            normalized_findings = [str(item).strip() for item in (findings or []) if str(item).strip()]
            main_finding = normalized_findings[0] if normalized_findings else ""
            main_norm = _normalize(main_finding)

            profiles = [
                {
                    "keywords": ["desbalanceo"],
                    "reason": "Predominio del armónico 1X y energía concentrada en baja frecuencia.",
                    "action": "Programar balanceo dinámico y revisar acoplamientos y sujeciones.",
                    "charlotte": "Tabla Charlotte – EM01 Desbalanceo del rotor",
                },
                {
                    "keywords": ["desalineacion", "desalineaci"],
                    "reason": "Armónicos 2X/3X elevados respecto a 1X según la severidad ISO.",
                    "action": "Verificar alineación angular/paralela y rigidez de la base.",
                    "charlotte": "Tabla Charlotte – EM02/EM03 Desalineación",
                },
                {
                    "keywords": ["holgura"],
                    "reason": "Múltiples armónicos de 1X con modulación que indica holgura mecánica.",
                    "action": "Inspeccionar fijaciones, tolerancias y aprietes estructurales.",
                    "charlotte": "Tabla Charlotte – EM04 Holgura mecánica",
                },
                {
                    "keywords": ["engran"],
                    "reason": "Componentes de malla y bandas laterales asociadas al tren de engranes.",
                    "action": "Revisar desgaste de dientes, juego y lubricación del engranaje.",
                    "charlotte": "Tabla Charlotte – EM19 Problemas en acoplamiento/engranes",
                },
                {
                    "keywords": ["rodamiento"],
                    "reason": "Picos en envolvente en BPFO/BPFI/BSF/FTF compatibles con daño de rodamiento.",
                    "action": "Evaluar lubricación, holguras y condición de pistas y elementos rodantes.",
                    "charlotte": "Tabla Charlotte – EM14–EM17 Fallas en rodamientos",
                },
                {
                    "keywords": ["electrico", "linea"],
                    "reason": "Componentes a la frecuencia de línea y su 2X excediendo la zona ISO.",
                    "action": "Balancear fases, revisar variador y conexiones del estator.",
                    "charlotte": "Tabla Charlotte – EM12/EM13 Problemas eléctricos",
                },
                {
                    "keywords": ["resonancia"],
                    "reason": "Picos agudos con factor Q alto fuera de armónicos cinemáticos conocidos.",
                    "action": "Verificar rigidez estructural y considerar prueba modal/FRF.",
                    "charlotte": "Tabla Charlotte – EM06 Resonancia estructural",
                },
            ]

            selected_profile = None
            for profile in profiles:
                if any(keyword in main_norm for keyword in profile["keywords"]):
                    selected_profile = profile
                    break
            if not selected_profile and main_norm.startswith("sin anomalia"):
                selected_profile = {
                    "reason": "Los niveles de vibración se mantienen dentro de los límites aceptables de la ISO.",
                    "action": "Continuar con monitoreo rutinario y verificación periódica de condiciones.",
                    "charlotte": None,
                }
            if not selected_profile:
                selected_profile = {
                    "reason": "No se identificó una anomalía predominante en la evaluación automática.",
                    "action": "Corroborar parámetros operativos y asegurar montaje correcto antes de intervenir.",
                    "charlotte": None,
                }

            energy_text = None
            try:
                energy = res.get("fft", {}).get("energy", {}) if isinstance(res, dict) else {}
                total = float(energy.get("total", 0.0))
                if total > 0:
                    low = float(energy.get("low", 0.0)) / total
                    mid = float(energy.get("mid", 0.0)) / total
                    high = float(energy.get("high", 0.0)) / total
                    energy_text = f"Energía espectral: baja {low:.0%}, media {mid:.0%}, alta {high:.0%}."
            except Exception:
                energy_text = None

            parts: List[str] = [f"Revisión ISO 10816/20816 – {severity_clause}."]
            if selected_profile.get("reason"):
                parts.append(f"Motivo principal: {selected_profile['reason']}")
            if selected_profile.get("action"):
                parts.append(f"Acción recomendada: {selected_profile['action']}")
            if selected_profile.get("charlotte"):
                parts.append(f"Referencia: {selected_profile['charlotte']}.")
            if energy_text:
                parts.append(energy_text)

            explanations.append(" ".join(parts))
        except Exception:
            explanations.append("Revisión ISO 10816/20816 – información no disponible. Verifique los datos de entrada.")
        return explanations

    def _resolve_analysis_period(
        self, time_vector: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        """Calcula el periodo válido de análisis según la UI y devuelve la máscara correspondiente.

        Respeta el rango seleccionado por el usuario; si no se especifica inicio/fin,
        analiza todo el registro disponible.
        """
        t_arr = np.asarray(time_vector, dtype=float).ravel()
        if t_arr.size == 0:
            return np.zeros(0, dtype=bool), 0.0, 0.0
        valid = np.isfinite(t_arr)
        if not np.any(valid):
            return np.zeros_like(t_arr, dtype=bool), 0.0, 0.0
        t_valid = t_arr[valid]
        full_start = float(np.nanmin(t_valid))
        full_end = float(np.nanmax(t_valid))
        start_val: Optional[float] = None
        end_val: Optional[float] = None
        try:
            raw_start = getattr(self.start_time_field, "value", None)
            if raw_start not in (None, ""):
                start_val = float(raw_start)
        except Exception:
            start_val = None
        try:
            raw_end = getattr(self.end_time_field, "value", None)
            if raw_end not in (None, ""):
                end_val = float(raw_end)
        except Exception:
            end_val = None
        start_t = start_val if start_val is not None else full_start
        end_t = end_val if end_val is not None else full_end
        start_t = min(max(start_t, full_start), full_end)
        end_t = min(max(end_t, full_start), full_end)
        if start_val is not None and end_val is not None and end_t <= start_t:
            start_t, end_t = end_t, start_t
        if end_t <= start_t:
            start_t, end_t = full_start, full_end
        mask = valid & (t_arr >= start_t) & (t_arr <= end_t)
        if np.count_nonzero(mask) < 2:
            mask = valid.copy()
            start_t, end_t = full_start, full_end
        return mask.astype(bool), float(start_t), float(end_t)

    def _resolve_runup_period(
        self, time_vector: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        """Delimita el periodo exclusivo para la cascada de arranque/paro."""

        t_arr = np.asarray(time_vector, dtype=float).ravel()
        if t_arr.size == 0:
            return np.zeros(0, dtype=bool), 0.0, 0.0
        valid = np.isfinite(t_arr)
        if not np.any(valid):
            return np.zeros_like(t_arr, dtype=bool), 0.0, 0.0
        t_valid = t_arr[valid]
        full_start = float(np.nanmin(t_valid))
        full_end = float(np.nanmax(t_valid))
        start_val: Optional[float] = None
        end_val: Optional[float] = None
        try:
            raw_start = getattr(self.runup_start_field, "value", None)
            if raw_start not in (None, ""):
                start_val = float(raw_start)
        except Exception:
            start_val = None
        try:
            raw_end = getattr(self.runup_end_field, "value", None)
            if raw_end not in (None, ""):
                end_val = float(raw_end)
        except Exception:
            end_val = None
        start_t = start_val if start_val is not None else full_start
        end_t = end_val if end_val is not None else full_end
        start_t = min(max(start_t, full_start), full_end)
        end_t = min(max(end_t, full_start), full_end)
        if start_val is not None and end_val is not None and end_t <= start_t:
            start_t, end_t = end_t, start_t
        if end_t <= start_t:
            start_t, end_t = full_start, full_end
        mask = valid & (t_arr >= start_t) & (t_arr <= end_t)
        if np.count_nonzero(mask) < 2:
            mask = valid.copy()
            start_t, end_t = full_start, full_end
        return mask.astype(bool), float(start_t), float(end_t)

    def _normalize_time_series(self, t: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Normaliza la serie temporal: limpia NaN, ordena, arranca en 0 y fuerza muestreo uniforme."""
        t_arr = np.asarray(t, dtype=float).ravel()
        y_arr = np.asarray(y, dtype=float).ravel()
        mask = np.isfinite(t_arr) & np.isfinite(y_arr)
        t_arr = t_arr[mask]
        y_arr = y_arr[mask]
        if t_arr.size == 0:
            return t_arr, y_arr, 0.0
        order = np.argsort(t_arr)
        t_arr = t_arr[order]
        y_arr = y_arr[order]
        unique_t, unique_idx = np.unique(t_arr, return_index=True)
        if unique_t.size != t_arr.size:
            t_arr = unique_t
            y_arr = y_arr[unique_idx]
        t_arr = t_arr - float(t_arr[0])
        if t_arr.size < 2:
            return t_arr.astype(float), y_arr.astype(float), 0.0
        diffs = np.diff(t_arr)
        valid_diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if valid_diffs.size == 0:
            target_dt = 0.0
        else:
            target_dt = float(np.median(valid_diffs))
        if not np.isfinite(target_dt) or target_dt <= 0:
            target_dt = float(np.mean(valid_diffs)) if valid_diffs.size else 0.0
        if not np.isfinite(target_dt) or target_dt <= 0:
            target_dt = 1.0
        uniform_t = np.linspace(0.0, target_dt * (t_arr.size - 1), t_arr.size)
        need_interp = False
        if valid_diffs.size:
            tol = max(1e-9, 1e-3 * target_dt)
            need_interp = np.max(np.abs(diffs - target_dt)) > tol
        if need_interp:
            y_uniform = np.interp(uniform_t, t_arr, y_arr)
        else:
            y_uniform = y_arr
        return uniform_t.astype(float), np.asarray(y_uniform, dtype=float), float(target_dt)

    def _convert_signal_to_acceleration(self, y: np.ndarray, dt: float, signal_label: Optional[str] = None) -> np.ndarray:
        """Convierte una señal (velocidad/desplazamiento/aceleración) a aceleración [m/s²]."""
        unit_map = getattr(self, "signal_unit_map", {}) or {}
        unit = None
        if signal_label and signal_label in unit_map:
            unit = unit_map.get(signal_label)
        if not unit:
            unit = getattr(self, "input_signal_unit", "acc_ms2") or "acc_ms2"
        y = np.asarray(y, dtype=float).ravel()
        if y.size == 0:
            return y
        g = 9.80665
        if unit == "acc_ms2":
            acc = y
        elif unit == "acc_g":
            acc = y * g
        elif unit == "vel_ms":
            acc = np.gradient(y, dt, edge_order=2 if y.size > 2 else 1) if dt > 0 else np.zeros_like(y)
        elif unit == "vel_mm":
            vel = y / 1000.0
            acc = np.gradient(vel, dt, edge_order=2 if vel.size > 2 else 1) if dt > 0 else np.zeros_like(vel)
        elif unit == "vel_ips":
            vel = y * 0.0254
            acc = np.gradient(vel, dt, edge_order=2 if vel.size > 2 else 1) if dt > 0 else np.zeros_like(vel)
        elif unit == "disp_m":
            if dt > 0:
                vel = np.gradient(y, dt, edge_order=2 if y.size > 2 else 1)
                acc = np.gradient(vel, dt, edge_order=2 if vel.size > 2 else 1)
            else:
                acc = np.zeros_like(y)
        elif unit == "disp_mm":
            disp = y / 1000.0
            if dt > 0:
                vel = np.gradient(disp, dt, edge_order=2 if disp.size > 2 else 1)
                acc = np.gradient(vel, dt, edge_order=2 if vel.size > 2 else 1)
            else:
                acc = np.zeros_like(disp)
        elif unit == "disp_um":
            disp = y * 1e-6
            if dt > 0:
                vel = np.gradient(disp, dt, edge_order=2 if disp.size > 2 else 1)
                acc = np.gradient(vel, dt, edge_order=2 if vel.size > 2 else 1)
            else:
                acc = np.zeros_like(disp)
        else:
            acc = y
        return np.asarray(acc, dtype=float)

    def _prepare_segment_for_analysis(
        self, t: np.ndarray, y: np.ndarray, signal_label: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """Prepara la señal recortada para el análisis y devuelve tiempo uniforme y aceleración."""
        t_uniform, y_uniform, dt = self._normalize_time_series(t, y)
        acc = self._convert_signal_to_acceleration(y_uniform, dt, signal_label)
        return t_uniform, acc, dt, y_uniform

    def _acc_to_vel_time_mm(self, acc: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Integra la aceleración para obtener velocidad temporal (mm/s) evitando arrastre entre análisis.
        - Aplica detrending lineal para eliminar la deriva.
        - Integra con suma acumulativa y corrige pendiente/offset residuales.
        """
        acc = np.asarray(acc, dtype=float).ravel()
        t = np.asarray(t, dtype=float).ravel()
        if acc.size < 2 or t.size < 2:
            return np.asarray([], dtype=float)
        t_rel = t - float(t[0])
        diffs = np.diff(t_rel)
        if diffs.size == 0:
            return np.asarray([], dtype=float)
        dt = float(np.median(diffs[np.isfinite(diffs)])) if np.any(np.isfinite(diffs)) else 0.0
        if not np.isfinite(dt) or dt <= 0:
            return np.asarray([], dtype=float)
        try:
            if acc.size >= 2:
                p = np.polyfit(t_rel, acc, 1)
                trend = p[0] * t_rel + p[1]
                acc_detrended = acc - trend
            else:
                acc_detrended = acc - float(np.mean(acc))
        except Exception:
            acc_detrended = acc - float(np.mean(acc))
        t_axis = t_rel
        vel_time = np.cumsum(acc_detrended) * dt
        if vel_time.size:
            vel_time -= float(np.mean(vel_time))
            try:
                if vel_time.size >= 2:
                    p_vel = np.polyfit(t_axis, vel_time, 1)
                    vel_time = vel_time - (p_vel[0] * t_axis + p_vel[1])
            except Exception:
                vel_time -= float(np.mean(vel_time))
        return 1000.0 * vel_time  # mm/s

    def _compute_fft_dual(self, acc: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcula simultáneamente el espectro de aceleración y velocidad
        (en mm/s) a partir de una señal temporal de aceleración.
        """
        acc = np.asarray(acc, dtype=float).ravel()
        t = np.asarray(t, dtype=float).ravel()
        if acc.size < 2 or t.size < 2:
            return np.array([]), np.array([]), np.array([])
        dt = float(t[1] - t[0])
        if not np.isfinite(dt) or dt == 0.0:
            return np.array([]), np.array([]), np.array([])
        N = acc.size
        yf = np.fft.fft(acc)
        xf = np.fft.fftfreq(N, dt)[: N // 2]
        mag_acc = 2.0 / N * np.abs(yf[: N // 2])
        mag_vel = np.zeros_like(mag_acc)
        pos = xf > 0
        if np.any(pos):
            mag_vel[pos] = mag_acc[pos] / (2.0 * np.pi * xf[pos])
        mag_vel_mm = mag_vel * 1000.0
        return xf, mag_vel_mm, mag_vel

    def _save_temp_plot(self, fig, store: List[str]) -> str:
        """
        Guarda una figura temporalmente y asegura su registro para limpieza posterior.
        """
        path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        store.append(path)
        return path

    def _apply_table_style(self, tbl: Table) -> None:
        """Aplica el estilo uniforme de tablas dentro de los reportes."""
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2f2f2")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d0d0d0")),
                ]
            )
        )

    def _trim_orbit_window(
        self,
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Aplica el periodo de órbita configurado y devuelve las señales recortadas."""

        t_arr = np.asarray(t, dtype=float).ravel()
        x_arr = np.asarray(x, dtype=float).ravel()
        y_arr = np.asarray(y, dtype=float).ravel()
        if t_arr.size == 0 or x_arr.size == 0 or y_arr.size == 0:
            return t_arr, x_arr, y_arr
        min_len = min(t_arr.size, x_arr.size, y_arr.size)
        if min_len <= 0:
            return t_arr, x_arr, y_arr
        t_arr = t_arr[:min_len]
        x_arr = x_arr[:min_len]
        y_arr = y_arr[:min_len]
        period = getattr(self, "orbit_period_seconds", None)
        try:
            period = float(period) if period is not None else None
        except Exception:
            period = None
        if period is None or not np.isfinite(period) or period <= 0:
            return t_arr, x_arr, y_arr
        try:
            t_valid = t_arr[np.isfinite(t_arr)]
            if t_valid.size == 0:
                return t_arr, x_arr, y_arr
            t_end = float(np.max(t_valid))
            mask = (t_arr >= (t_end - period)) & np.isfinite(t_arr)
        except Exception:
            return t_arr, x_arr, y_arr
        if np.count_nonzero(mask) < 16:
            return t_arr, x_arr, y_arr
        return t_arr[mask], x_arr[mask], y_arr[mask]

    def _classify_severity(self, rms_value):
        """
        Clasifica el nivel de severidad basado en ISO 10816/20816-3
        para motores eléctricos usando velocidad en mm/s.
        """
        if rms_value <= 2.8:
            return "✅ Buena (Aceptable)"
        elif rms_value <= 4.5:
            return "⚠️ Satisfactoria (Zona de vigilancia)"
        elif rms_value <= 7.1:
            return "❌ Insatisfactoria (Crítica)"
        else:
            return "🔥 Inaceptable (Riesgo de daño)"

    def _classify_severity_ms(self, rms_ms):
        """
        Variante en unidades SI (m/s). Devuelve (label, color_hex).
        Umbrales equivalentes a 2.8, 4.5, 7.1 mm/s.
        """
        if rms_ms <= 0.0028:
            return "Buena (Aceptable)", "#2ecc71"
        elif rms_ms <= 0.0045:
            return "Satisfactoria (Zona de vigilancia)", "#f1c40f"
        elif rms_ms <= 0.0071:
            return "Insatisfactoria (Crítica)", "#e67e22"
        else:
            return "Inaceptable (Riesgo de daño)", "#e74c3c"


    def _band_energy(self, xf, spec, f0, f1):

        """Energía (suma de amplitud^2) entre f0–f1 Hz."""

        if xf is None or spec is None or len(xf) == 0:

            return 0.0

        idx = (xf >= f0) & (xf < f1)

        return float(np.sum((spec[idx] ** 2))) if np.any(idx) else 0.0



    def _amp_near(self, xf, spec, f, tol=2.0):
        """Amplitud máx cerca de frecuencia objetivo f ± tol. Devuelve 0.0 si no hay datos o f no es válida."""
        if xf is None or spec is None or len(xf) == 0 or f is None or not np.isfinite(f) or f <= 0:
            return 0.0
        idx = (xf >= (f - tol)) & (xf <= (f + tol))
        return float(np.max(spec[idx])) if np.any(idx) else 0.0


    def _analytic_signal(self, y: np.ndarray):
        y = np.asarray(y, dtype=float)
        N = len(y)
        if N < 2:
            return y.astype(complex)
        Y = np.fft.fft(y)
        h = np.zeros(N)
        if N % 2 == 0:
            h[0] = 1
            h[N // 2] = 1
            h[1:N // 2] = 2
        else:
            h[0] = 1
            h[1:(N + 1) // 2] = 2
        Z = np.fft.ifft(Y * h)
        return Z

    def _compute_envelope_spectrum(self, y: np.ndarray, T: float):
        N = len(y)
        if N < 2:
            return None, None
        z = self._analytic_signal(y)
        env = np.abs(z)
        env = env - float(np.mean(env))
        Ef = np.fft.fft(env)
        xf = np.fft.fftfreq(N, T)[: N // 2]
        mag = 2.0 / N * np.abs(Ef[: N // 2])
        return xf, mag
        return float(np.max(spec[idx])) if np.any(idx) else 0.0



    def _extract_features(self, t, acc_signal, xf, vel_spec):
        """
        Extrae features de tiempo (aceleración) y frecuencia (velocidad [mm/s]).
        - t: vector de tiempo del segmento
        - acc_signal: aceleración del segmento (m/s^2)
        - xf: frecuencias FFT
        - vel_spec: magnitud espectral de velocidad [mm/s]
        """
        acc = acc_signal.astype(float)

        rms_time_acc = float(np.sqrt(np.mean(acc**2))) if len(acc) else 0.0

        peak = float(np.max(np.abs(acc))) if len(acc) else 0.0

        pp = float(np.ptp(acc)) if len(acc) else 0.0

        mu = float(np.mean(acc)) if len(acc) else 0.0

        std = float(np.std(acc)) + 1e-12

        skew = float(np.mean(((acc - mu) / std) ** 3))

        kurt = float(np.mean(((acc - mu) / std) ** 4) - 3.0)

        crest = float(peak / (rms_time_acc + 1e-12))

        vel_time_mm = self._acc_to_vel_time_mm(acc, t)
        rms_vel_time = float(np.sqrt(np.mean(vel_time_mm**2))) if vel_time_mm.size else 0.0



        if vel_spec is None or len(vel_spec) == 0:

            dom_freq = 0.0

            dom_amp = 0.0

            total_energy = 1e-12

            e_low = e_mid = e_high = 0.0

            r2x = r3x = 0.0

            rms_vel_spec = rms_vel_time

        else:

            dom_idx = int(np.argmax(vel_spec))

            dom_freq = float(xf[dom_idx]) if len(xf) else 0.0

            dom_amp = float(vel_spec[dom_idx])

            total_energy = float(np.sum(vel_spec**2)) + 1e-12



            # Bandas típicas (ajústalas a tu máquina)

            e_low = self._band_energy(xf, vel_spec, 0.0, 30.0)

            e_mid = self._band_energy(xf, vel_spec, 30.0, 120.0)

            max_f = float(xf.max()) if len(xf) else 500.0

            e_high = self._band_energy(xf, vel_spec, 120.0, max_f)



            r2x = self._amp_near(xf, vel_spec, 2 * dom_freq) / (dom_amp + 1e-12)

            r3x = self._amp_near(xf, vel_spec, 3 * dom_freq) / (dom_amp + 1e-12)

            rms_vel_spec = rms_vel_time



        return {
            "rms_time_acc": rms_time_acc,
            "peak_acc": peak,
            "pp_acc": pp,
            "crest": crest,
            "skew": skew,
            "kurt": kurt,
            "dom_freq": dom_freq,
            "dom_amp": dom_amp,
            "r2x": r2x,
            "r3x": r3x,
            "e_low": e_low,
            "e_mid": e_mid,
            "e_high": e_high,
            "e_total": total_energy,
            "frac_low": (e_low / total_energy) if total_energy > 0 else 0.0,
            "frac_mid": (e_mid / total_energy) if total_energy > 0 else 0.0,
            "frac_high": (e_high / total_energy) if total_energy > 0 else 0.0,
            "rms_vel_spec": rms_vel_spec,
            "energy_low_frac": (e_low / total_energy) if total_energy > 0 else 0.0,
            "energy_mid_frac": (e_mid / total_energy) if total_energy > 0 else 0.0,
            "energy_high_frac": (e_high / total_energy) if total_energy > 0 else 0.0,
            "rms_vel_time_mm": rms_vel_time,
        }

    # ---- Helpers avanzados de diagnóstico basado en FFT ----
    def _get_float(self, tf):
        try:
            if tf and getattr(tf, "value", None):
                return float(tf.value)
        except Exception:
            return None
        return None

    def _get_1x_hz(self, dom_freq_guess: float | None = None):
        """Obtiene 1X en Hz desde RPM (si la hay) o una conjetura (dom_freq)."""
        try:
            rpm = self._get_float(getattr(self, "rpm_hint_field", None))
            if rpm and rpm > 0:
                return rpm / 60.0
        except Exception:
            pass
        try:
            if dom_freq_guess and dom_freq_guess > 0:
                return float(dom_freq_guess)
        except Exception:
            pass
        return 0.0

    def _estimate_rpm(self, xf, spec):
        if xf is None or spec is None or len(xf) == 0:
            return None
        try:
            # usar el pico dominante como 1X si está por debajo de 200 Hz
            idx = int(np.argmax(spec))
            f = float(xf[idx])
            if f <= 0:
                return None
            return f * 60.0
        except Exception:
            return None

    def _peak_amp_near(self, xf, spec, f, tol_rel=0.03, tol_abs=1.0):
        if xf is None or spec is None or f is None or f <= 0:
            return 0.0
        bw = max(tol_abs, tol_rel * f)
        idx = (xf >= (f - bw)) & (xf <= (f + bw))
        return float(np.max(spec[idx])) if np.any(idx) else 0.0

    def _sideband_score(self, xf, spec, center, spacing, n=3, tol_rel=0.03):
        if center is None or spacing is None or center <= 0 or spacing <= 0:
            return 0.0
        amps = []
        for k in range(1, n + 1):
            amps.append(self._peak_amp_near(xf, spec, center - k * spacing, tol_rel))
            amps.append(self._peak_amp_near(xf, spec, center + k * spacing, tol_rel))
        return float(np.mean(amps)) if amps else 0.0

    def _detect_faults(self, xf, vel_spec, features):
        """Devuelve hallazgos detallados por subsistema: balanceo, alineación,
        holguras, rodamientos, engranes, eléctrico.
        Usa parámetros opcionales: RPM, BPFO/BPFI/BSF/FTF, dientes, línea.
        """
        findings = []
        if xf is None or vel_spec is None or len(xf) == 0:
            return findings

        # 1X desde RPM o dom_freq (seguro)
        f1 = self._get_1x_hz(features.get("dom_freq", 0.0))

        # Amplitudes 1X..4X
        a1 = self._peak_amp_near(xf, vel_spec, f1)
        a2 = self._peak_amp_near(xf, vel_spec, 2 * f1)
        a3 = self._peak_amp_near(xf, vel_spec, 3 * f1)
        a4 = self._peak_amp_near(xf, vel_spec, 4 * f1)

        # Unbalance
        if f1 > 0 and a1 > 0 and (a2 < 0.5 * a1) and (a3 < 0.4 * a1) and (features.get("e_low", 0) / max(features.get("e_total", 1e-12), 1e-12) > 0.5):
            findings.append(f"Desbalanceo probable (1X dominante, 2X/3X bajos). 1X={f1:.2f} Hz")

        # Misalignment
        if f1 > 0 and (a2 >= 0.6 * a1 or a3 >= 0.4 * a1):
            findings.append("Desalineación probable (armónicos 2X/3X elevados respecto a 1X)")

        # Looseness
        harmonics = [a1, a2, a3, a4]
        if f1 > 0 and sum(1 for a in harmonics if a > 0.3 * max(harmonics)) >= 3:
            findings.append("Holgura mecánica (múltiples armónicos de 1X significativos)")

        # Gear mesh
        gear_teeth = self._get_float(getattr(self, "gear_teeth_field", None))
        if f1 > 0 and gear_teeth and gear_teeth > 0:
            fgm = f1 * gear_teeth
            a_gm = self._peak_amp_near(xf, vel_spec, fgm, tol_rel=0.02, tol_abs=2.0)
            sb = self._sideband_score(xf, vel_spec, fgm, f1, n=3)
            if a_gm > 0 and sb > 0.2 * a_gm:
                findings.append(f"Engranes: frecuencia de malla ~{fgm:.1f} Hz con bandas laterales ±1X")

        # Bearings (si el usuario conoce frecuencias)
        bpfo = self._get_float(getattr(self, "bpfo_field", None))
        bpfi = self._get_float(getattr(self, "bpfi_field", None))
        bsf  = self._get_float(getattr(self, "bsf_field", None))
        ftf  = self._get_float(getattr(self, "ftf_field", None))
        bearing_hits = []
        for name, freq in (("BPFO", bpfo), ("BPFI", bpfi), ("BSF", bsf), ("FTF", ftf)):
            if freq and freq > 0:
                amp = self._peak_amp_near(xf, vel_spec, freq, tol_rel=0.02, tol_abs=2.0)
                if amp > 0.2 * max(vel_spec) if len(vel_spec) else 0:
                    sb = self._sideband_score(xf, vel_spec, freq, f1 if f1 else freq, n=2)
                    bearing_hits.append(f"{name} (~{freq:.1f} Hz){' con bandas laterales' if sb>0 else ''}")
        if bearing_hits:
            findings.append("Rodamientos: patrones en " + ", ".join(bearing_hits))

        # Eléctrico (línea)
        line_opt = getattr(self, "line_freq_dd", None)
        try:
            line_hz = float(line_opt.value) if line_opt and line_opt.value else None
        except Exception:
            line_hz = None
        if line_hz:
            a_line = self._peak_amp_near(xf, vel_spec, line_hz, tol_rel=0.01, tol_abs=1.0)
            a_2line = self._peak_amp_near(xf, vel_spec, 2 * line_hz, tol_rel=0.01, tol_abs=1.0)
            if (a_line > 0.2 * max(vel_spec)) or (a_2line > 0.2 * max(vel_spec)):
                findings.append(f"Eléctrico (suministro): picos en {line_hz:.0f} Hz y/o {2*line_hz:.0f} Hz")

        return findings


    def _diagnose(self, f):
        """
        Reglas simples de diagnóstico (baseline). Devuelve lista de hallazgos.
        Usa velocidad espectral (mm/s) para severidad ISO.
        """
        findings = []

        # Severidad global (tu función ya clasifica con mm/s)
        sev = self._classify_severity(f.get("rms_vel_spec", 0.0))
        findings.append(f"Severidad ISO (velocidad RMS): {sev}")

        # Desbalanceo: pico 1X dominante, armónicos bajos, energía en baja frecuencia
        if f["dom_freq"] > 0 and f["dom_freq"] < 60 and f["r2x"] < 0.5 and (f["e_low"] / f["e_total"]) > 0.5:
            findings.append("⚠️ Posible desbalanceo: pico 1X dominante y baja energía en armónicos (2X/3X).")

        # Desalineación: armónicos 2X/3X fuertes
        if f["r2x"] >= 0.6 or f["r3x"] >= 0.4:
            findings.append("⚠️ Posible desalineación: armónicos altos (2X/3X) significativos.")

        # Rodamientos: energía predominante en alta frecuencia
        if f["dom_freq"] > 200 and (f["e_high"] / f["e_total"]) > 0.5:
            findings.append("❌ Posible falla en rodamientos: energía predominante en alta frecuencia.")

        # Resonancia: RMS muy alto + crest alto
        if f["rms_vel_spec"] >= 7.1 and f["crest"] > 3.0:
            findings.append("🔥 Posible resonancia: RMS muy alto y alto crest factor.")

        # Reglas avanzadas con espectro actual
        try:
            xf = getattr(self, "_last_xf", None)
            spec = getattr(self, "_last_spec", None)
            if xf is not None and spec is not None:
                findings.extend(self._detect_faults(xf, spec, f))
        except Exception:
            pass

        if len(findings) == 1:
            findings.append("Sin anomalías evidentes según reglas actuales.")

        return findings




    def _create_plot(self):
            
        """
        Genera las gráficas y el diagnóstico.
        🔥 CORRECCIÓN: Se asegura de usar una copia 100% limpia del DataFrame original
        en cada ejecución para evitar la contaminación de datos entre análisis.
        """
        try:
            # 🔥 PASO 1: No confiar en self.current_df. Crear una copia local y limpia.
            if self._raw_current_df is None or self._raw_current_df.empty:
                return ft.Text("No hay datos cargados para analizar.", size=16)

            df_limpio = self._raw_current_df.copy(deep=True)
            time_key = str(self.time_dropdown.value) if getattr(self, "time_dropdown", None) else "t_s"
            unit_map_local: Dict[str, str] = {}
            if isinstance(self.signal_unit_map, dict) and self.signal_unit_map:
                unit_map_local = dict(self.signal_unit_map)
            else:
                # Inferir unidades básicas a partir del nombre de la columna.
                for col in df_limpio.columns:
                    col_str = str(col)
                    if col_str == time_key:
                        continue
                    name = col_str.lower()
                    if "acc" in name or "acel" in name:
                        unit_map_local[col_str] = "acc_ms2"
                    elif "vel" in name:
                        unit_map_local[col_str] = "vel_mm" if "mm" in name else "vel_ms"
                    elif "disp" in name or "despl" in name:
                        unit_map_local[col_str] = "disp_mm"
            self.signal_unit_map = dict(unit_map_local)
            try:
                self.current_df = df_limpio.copy(deep=True)
            except Exception:
                self.current_df = df_limpio

            # 🔥 PASO 2: A partir de aquí, usar SIEMPRE 'df_limpio' en lugar de 'self.current_df'.
            time_col = self.time_dropdown.value
            fft_signal_col = self.fft_dropdown.value
            
            # Verificar si las columnas existen en el DataFrame limpio
            if time_col not in df_limpio.columns or fft_signal_col not in df_limpio.columns:
                return ft.Text(f"Error: Las columnas '{time_col}' o '{fft_signal_col}' no se encontraron.", color="#e74c3c")

            t = df_limpio[time_col].to_numpy()
            signal = df_limpio[fft_signal_col].to_numpy()
            try:
                print(f"[DEBUG] Primeros valores crudos {fft_signal_col}: {signal[:5]}")
            except Exception:
                pass

            # --- Filtrar periodo ---
            mask, start_t, end_t = self._resolve_analysis_period(t)
            if mask.size == 0 or np.count_nonzero(mask) < 2:
                return ft.Text("⚠️ Rango de tiempo inválido.", size=14, color="#e74c3c")

            segment_idx = np.nonzero(mask)[0]
            t_segment_raw = t[segment_idx]
            signal_segment_raw = signal[segment_idx]
            
            # 🔥 PASO 3: Asegurarse de que el DataFrame para el segmento también sea el limpio.
            segment_df = df_limpio.iloc[segment_idx]

            t_segment, acc_segment, _, _ = self._prepare_segment_for_analysis(t_segment_raw, signal_segment_raw, fft_signal_col)
            try:
                print(f"[DEBUG] Primeros valores acc_segment ({fft_signal_col}): {acc_segment[:5]}")
            except Exception:
                pass
            
            # ... El resto de tu función continúa exactamente igual desde aquí ...
            # ... No necesitas cambiar nada más en el resto de la función ...

            if len(acc_segment) < 2:

                return ft.Text("⚠️ Rango inválido", size=14, color="#e74c3c")





            # --- Features + diagnóstico baseline ---

            self._reset_runtime_analysis_state(announce=False)

            # Analizar con función robusta
            try:
                rpm_val = None
                if getattr(self, "rpm_hint_field", None) and getattr(self.rpm_hint_field, "value", ""):
                    rpm_val = float(self.rpm_hint_field.value)
            except Exception:
                rpm_val = None
            try:
                line_val = float(self.line_freq_dd.value) if getattr(self, "line_freq_dd", None) and getattr(self.line_freq_dd, "value", "") else None
            except Exception:
                line_val = None
            try:
                teeth_val = int(self.gear_teeth_field.value) if getattr(self, "gear_teeth_field", None) and getattr(self.gear_teeth_field, "value", "") else None
            except Exception:
                teeth_val = None
            # usar helper unificado para lectura de floats opcionales
            try:
                _fmax_pre = float(self.hf_limit_field.value) if getattr(self, 'hf_limit_field', None) and getattr(self.hf_limit_field, 'value', '') else None
            except Exception:
                _fmax_pre = None
            bpfo_val = self._fldf(getattr(self, 'bpfo_field', None))
            bpfi_val = self._fldf(getattr(self, 'bpfi_field', None))
            bsf_val = self._fldf(getattr(self, 'bsf_field', None))
            ftf_val = self._fldf(getattr(self, 'ftf_field', None))
            env_lo_val = self._fldf(getattr(self, 'env_bp_lo_field', None))
            env_hi_val = self._fldf(getattr(self, 'env_bp_hi_field', None))
            res = analyze_vibration(
                t_segment,
                acc_segment,
                rpm=rpm_val,
                line_freq_hz=line_val,
                bpfo_hz=bpfo_val,
                bpfi_hz=bpfi_val,
                bsf_hz=bsf_val,
                ftf_hz=ftf_val,
                gear_teeth=teeth_val,
                pre_decimate_to_fmax_hz=_fmax_pre,
                env_bp_lo_hz=env_lo_val,
                env_bp_hi_hz=env_hi_val,
                fft_window=self.fft_window_type,
            )
            # Sustituir espectros por los del analizador
            xf = res['fft']['f_hz']
            mag_vel_mm = res['fft']['vel_spec_mm_s']
            if xf is not None and len(xf) > 0:
                try:
                    arr = np.asarray(xf, dtype=float)
                    arr = arr[np.isfinite(arr)]
                    if arr.size > 0:
                        full_min = float(arr.min())
                        full_max = float(arr.max())
                        if full_max > full_min:
                            self._fft_full_range = (full_min, full_max)
                            current_zoom = self._fft_zoom_range
                            if current_zoom is not None:
                                start_val = max(full_min, min(current_zoom[0], full_max))
                                end_val = max(start_val + 1e-6, min(current_zoom[1], full_max))
                                self._fft_zoom_range = (start_val, end_val)
                            self._apply_fft_display_units(self._fft_full_range, self._fft_zoom_range)
                            self._update_fft_zoom_controls(self._fft_full_range, self._fft_zoom_range)
                        else:
                            self._fft_full_range = None
                            self._fft_zoom_range = None
                            self._apply_fft_display_units(None, None)
                            self._update_fft_zoom_controls(None, None)
                    else:
                        self._fft_full_range = None
                        self._fft_zoom_range = None
                        self._apply_fft_display_units(None, None)
                        self._update_fft_zoom_controls(None, None)
                except Exception:
                    self._fft_full_range = None
                    self._fft_zoom_range = None
                    self._apply_fft_display_units(None, None)
                    self._update_fft_zoom_controls(None, None)
            else:
                self._fft_full_range = None
                self._fft_zoom_range = None
                self._apply_fft_display_units(None, None)
                self._update_fft_zoom_controls(None, None)
            dom_freq = res['fft']['dom_freq_hz']
            dom_amp = res['fft']['dom_amp_mm_s']
            selected_rms_mm = res['severity']['rms_mm_s']
            selected_label = res['severity']['label']
            selected_color = res['severity']['color']
            axis_summaries, primary_entry = self._compute_axis_severity(
                time_col,
                mask,
                rpm_val,
                line_val,
                teeth_val,
                _fmax_pre,
                bpfo_val,
                bpfi_val,
                bsf_val,
                ftf_val,
                env_lo_val,
                env_hi_val,
                df=df_limpio,
                unit_map=unit_map_local,
            )
            if primary_entry is None:
                primary_entry = {
                    "name": self._axis_display_name(fft_signal_col),
                    "column": fft_signal_col,
                    "rms_mm_s": selected_rms_mm,
                    "iso_label": selected_label,
                    "emoji_label": self._classify_severity(selected_rms_mm),
                    "color": selected_color,
                    "is_global": False,
                }
                self._last_primary_severity = primary_entry
                if not getattr(self, "_last_axis_severity", []):
                    self._last_axis_severity = [primary_entry]
            else:
                self._last_primary_severity = primary_entry
            primary_rms_mm = float(primary_entry.get("rms_mm_s", selected_rms_mm))
            primary_label = primary_entry.get("iso_label", selected_label)
            primary_color = primary_entry.get("color", selected_color)
            raw_findings = res.get('diagnosis', [])
            findings_core = list(res.get('diagnosis_findings', []) or [])
            if not findings_core and raw_findings:
                _, findings_core = _split_diagnosis(raw_findings)
            findings = findings_core
            # Explicación y revisiones sugeridas (basado en hallazgos y métricas)
            exp_lines = []
            # Reducir hallazgos a los principales (para explicaciones)
            try:
                _sel = self._select_main_findings(findings)
            except Exception:
                _sel = findings
            findings = _sel
            # Enfoque explícito: motor eléctrico
            exp_lines.append("Enfoque: motor eléctrico")
            try:
                en = res.get('fft', {}).get('energy', {})
                e_total = float(en.get('total', 1e-12))
                frac_low = (float(en.get('low', 0.0)) / e_total) if e_total > 0 else 0.0
                frac_mid = (float(en.get('mid', 0.0)) / e_total) if e_total > 0 else 0.0
                frac_high = (float(en.get('high', 0.0)) / e_total) if e_total > 0 else 0.0
            except Exception:
                frac_low = frac_mid = frac_high = 0.0
            try:
                exp_lines.append(f"Severidad por RMS de velocidad (ISO): {primary_rms_mm:.3f} mm/s → {primary_label}.")
            except Exception:
                pass
            def _has(txt: str) -> bool:
                try:
                    return any((txt in s) for s in (findings or []))
                except Exception:
                    return False
            charlotte_refs = {
                "Desbalanceo": "EM01 – Desbalanceo del rotor",
                "Desalineaci": "EM02/EM03 – Desalineación (angular/paralela)",
                "Holgura": "EM04 – Holgura mecánica",
                "Engranes": "EM19 – Problemas en acoplamiento/engranes",
                "Rodamientos": "EM14–EM17 – Fallas en rodamientos",
                "Elctrico": "EM12/EM13 – Problemas eléctricos del estator / armónicos de línea",
                "Eléctrico": "EM12/EM13 – Problemas eléctricos del estator / armónicos de línea",
                "El?ctrico": "EM12/EM13 – Problemas eléctricos del estator / armónicos de línea",
                "Resonancia": "EM06 – Resonancia estructural",
            }

            def _charlotte_line(key: str) -> Optional[str]:
                ref = charlotte_refs.get(key)
                return f"Tabla Charlotte: {ref}." if ref else None

            if _has("Desbalanceo"):
                exp_lines.append("Motivo: 1X dominante, 2X/3X bajos y energía concentrada en baja frecuencia.")
                exp_lines.append("Revisar: balanceo del rotor/acoplamiento, fijaciones y suciedad/excentricidad.")
                ref_line = _charlotte_line("Desbalanceo")
                if ref_line:
                    exp_lines.append(ref_line)
            if _has("Desalineaci"):
                exp_lines.append("Motivo: armónicos 2X/3X elevados respecto a 1X.")
                exp_lines.append("Revisar: alineación de ejes, calces y planitud de la base.")
                ref_line = _charlotte_line("Desalineaci")
                if ref_line:
                    exp_lines.append(ref_line)
            if _has("Holgura"):
                exp_lines.append("Motivo: múltiples armónicos de 1X significativos en el espectro.")
                exp_lines.append("Revisar: sujeciones, tolerancias mecánicas y posibles juegos en componentes.")
                ref_line = _charlotte_line("Holgura")
                if ref_line:
                    exp_lines.append(ref_line)
            if _has("Engranes"):
                exp_lines.append("Motivo: componente de malla de engranajes apreciable.")
                exp_lines.append("Revisar: desgaste de dientes, juego y lubricación.")
                ref_line = _charlotte_line("Engranes")
                if ref_line:
                    exp_lines.append(ref_line)
            if _has("Rodamientos"):
                exp_lines.append("Motivo: picos en envolvente en frecuencias características de rodamientos.")
                exp_lines.append("Revisar: lubricación, holgura y posible daño en pistas/elementos.")
                ref_line = _charlotte_line("Rodamientos")
                if ref_line:
                    exp_lines.append(ref_line)
            if _has("Elctrico") or _has("Eléctrico") or _has("El?ctrico"):
                exp_lines.append("Motivo: componentes a frecuencia de línea y/o su 2x.")
                exp_lines.append("Revisar: balance de fases, variador, conexiones y carga del motor.")
                for key in ("Elctrico", "Eléctrico", "El?ctrico"):
                    ref_line = _charlotte_line(key)
                    if ref_line and ref_line not in exp_lines:
                        exp_lines.append(ref_line)
            if _has("Resonancia estructural"):
                exp_lines.append("Motivo: picos agudos con Q alto fuera de armónicos conocidos.")
                exp_lines.append("Revisar: rigidez/soportes, aprietes y realizar prueba modal/FRF si es posible.")
                ref_line = _charlotte_line("Resonancia")
                if ref_line:
                    exp_lines.append(ref_line)
            if frac_low + frac_mid + frac_high > 0:
                try:
                    exp_lines.append(f"Distribución de energía: baja {frac_low:.0%}, media {frac_mid:.0%}, alta {frac_high:.0%} (guía del tipo de fallo).")
                except Exception:
                    pass
            # Guardar última FFT/segmento para diagnóstico avanzado
            self._last_xf = xf
            self._last_spec = mag_vel_mm
            self._last_tseg = t_segment
            self._last_accseg = acc_segment



            # --- Gráficas principales ---

            plt.style.use('dark_background' if self.is_dark_mode else 'seaborn-v0_8-whitegrid')
            plt.rcParams["font.family"] = "DejaVu Sans"

            # Preparar senal de tiempo segun unidad seleccionada
            try:
                unit_mode = getattr(self, "time_unit_dd", None).value if getattr(self, "time_unit_dd", None) else "vel_mm"
            except Exception:
                unit_mode = "vel_mm"
            if unit_mode == "vel_mm":
                _y_time = self._acc_to_vel_time_mm(acc_segment, t_segment)
                _ylabel = "Velocidad [mm/s]"
                _rms_text = (
                    f"RMS velocidad: {self._calculate_rms(_y_time):.3f} mm/s"
                    if _y_time.size
                    else "RMS velocidad: 0.000 mm/s"
                )
            elif unit_mode == "acc_g":
                _y_time = acc_segment / 9.80665
                _ylabel = "Aceleración [g]"
                _rms_text = f"RMS acc: {self._calculate_rms(_y_time):.3f} g"
            else:
                _y_time = acc_segment
                _ylabel = "Aceleración [m/s²]"
                _rms_text = f"RMS acc: {self._calculate_rms(_y_time):.3e} m/s^2"
            # Aplicar filtros visuales de frecuencia (LF y/o límite HF)
            try:
                fc = float(self.lf_cutoff_field.value) if getattr(self, 'lf_cutoff_field', None) and getattr(self.lf_cutoff_field, 'value', '') else 0.5
            except Exception:
                fc = 0.5
            try:
                fmax_ui = float(self.hf_limit_field.value) if getattr(self, 'hf_limit_field', None) and getattr(self.hf_limit_field, 'value', '') else None
            except Exception:
                fmax_ui = None
            try:
                hide_lf = bool(getattr(self, 'hide_lf_cb', None).value)
            except Exception:
                hide_lf = True
            zoom_range = getattr(self, "_fft_zoom_range", None)
            zmin = zmax = None
            if zoom_range and len(zoom_range) == 2 and zoom_range[1] > zoom_range[0]:
                try:
                    zmin, zmax = float(zoom_range[0]), float(zoom_range[1])
                except Exception:
                    zmin = zmax = None
            freq_scale = getattr(self, "_fft_display_scale", 1.0) or 1.0
            if not np.isfinite(freq_scale) or freq_scale <= 0:
                freq_scale = 1.0
            freq_unit = getattr(self, "_fft_display_unit", "Hz") or "Hz"
            if xf is not None and mag_vel_mm is not None:
                mask_vis = np.ones_like(xf, dtype=bool)
                if hide_lf:
                    mask_vis &= xf >= max(0.0, fc)
                if fmax_ui and fmax_ui > 0:
                    mask_vis &= xf <= fmax_ui
                if zmin is not None:
                    mask_vis &= (xf >= zmin) & (xf <= zmax)
                xplot = xf[mask_vis]
                yplot = mag_vel_mm[mask_vis]
                if xplot.size == 0:
                    xplot = xf
                    yplot = mag_vel_mm
            else:
                xplot = xf
                yplot = mag_vel_mm
                mask_vis = None

            # Escala dBV opcional sobre el espectro (re 1 V) usando calibración
            use_dbv = False
            try:
                use_dbv = bool(getattr(self, 'db_scale_cb', None) and getattr(self.db_scale_cb, 'value', False))
            except Exception:
                use_dbv = False

            xplot_disp = np.asarray(xplot, dtype=float) / freq_scale if xplot is not None else xplot
            yplot_dbv = None
            db_axis_min = db_axis_max = None
            if use_dbv:
                try:
                    # Espectro de aceleración para dBV si el sensor es de acc
                    acc_spec = res.get('fft', {}).get('acc_spec_ms2', None)
                    if acc_spec is None:
                        acc_spec = np.zeros_like(xf)
                    acc_plot = acc_spec[mask_vis] if (mask_vis is not None) else acc_spec
                    # Leer parámetros de calibración
                    sens_unit = getattr(self.sens_unit_dd, 'value', 'mV/g') if getattr(self, 'sens_unit_dd', None) else 'mV/g'
                    try:
                        sens_val = float(getattr(self.sensor_sens_field, 'value', '100')) if getattr(self, 'sensor_sens_field', None) else 100.0
                    except Exception:
                        sens_val = 100.0
                    try:
                        gain_vv = float(getattr(self, 'gain_field', '1.0').value) if getattr(self, 'gain_field', None) else 1.0
                    except Exception:
                        try:
                            gain_vv = float(getattr(self, 'gain_field', None).value)
                        except Exception:
                            gain_vv = 1.0
                    # Convertir a Voltios según tipo de sensor
                    if sens_unit == 'mV/g':
                        sens_v_per_g = sens_val * 1e-3
                        V_amp = (acc_plot / 9.80665) * sens_v_per_g * gain_vv
                    elif sens_unit == 'V/g':
                        V_amp = (acc_plot / 9.80665) * sens_val * gain_vv
                    elif sens_unit == 'mV/(mm/s)':
                        V_amp = yplot * (sens_val * 1e-3) * gain_vv
                    elif sens_unit == 'V/(mm/s)':
                        V_amp = yplot * sens_val * gain_vv
                    else:
                        V_amp = yplot * 0.0
                    eps = 1e-12
                    yplot_dbv = 20.0 * np.log10(np.maximum(np.asarray(V_amp, dtype=float), eps) / 1.0)
                    try:
                        db_axis_min = float(self.db_ymin_field.value) if getattr(self, 'db_ymin_field', None) and getattr(self.db_ymin_field, 'value', '') != '' else None
                    except Exception:
                        db_axis_min = None
                    try:
                        db_axis_max = float(self.db_ymax_field.value) if getattr(self, 'db_ymax_field', None) and getattr(self.db_ymax_field, 'value', '') != '' else None
                    except Exception:
                        db_axis_max = None
                except Exception:
                    yplot_dbv = None
                    db_axis_min = db_axis_max = None

            # Marcar picos principales (Top-N)
            peak_points: List[Tuple[float, float]] = []
            peak_labels: List[str] = []
            try:
                K = 5
                min_freq = (max(0.5, fc) if hide_lf else 0.5)
                if xf is not None and mag_vel_mm is not None:
                    mask = xf >= min_freq
                    if zmin is not None:
                        mask &= (xf >= zmin) & (xf <= zmax)
                    xv = xf[mask]
                    yv = mag_vel_mm[mask]
                    if len(yv) > 0:
                        k = min(K, len(yv))
                        idx = np.argpartition(yv, -k)[-k:]
                        idx = idx[np.argsort(yv[idx])[::-1]]
                        peak_f = xv[idx]
                        peak_a = yv[idx]
                        f1 = self._get_1x_hz(dom_freq)
                        for pf, pa in zip(peak_f, peak_a):
                            try:
                                pf_f = float(pf)
                                pa_f = float(pa)
                            except Exception:
                                continue
                            order = None
                            if f1 and f1 > 0:
                                try:
                                    order = pf_f / float(f1)
                                except Exception:
                                    order = None
                            peak_points.append((pf_f / freq_scale, pa_f))
                            peak_labels.append(self._format_peak_label(pf_f, pa_f, order))
            except Exception:
                peak_points = []
                peak_labels = []

            # Líneas guía de frecuencias teóricas (modo asistido)
            visible_marks: List[Tuple[float, str, str]] = []
            try:
                bpfo = self._fldf(getattr(self, 'bpfo_field', None))
                bpfi = self._fldf(getattr(self, 'bpfi_field', None))
                bsf  = self._fldf(getattr(self, 'bsf_field', None))
                ftf  = self._fldf(getattr(self, 'ftf_field', None))
                marks_raw = [
                    (bpfo, 'BPFO', '#1f77b4'),
                    (bpfi, 'BPFI', '#ff7f0e'),
                    (bsf,  'BSF',  '#2ca02c'),
                    (ftf,  'FTF',  '#9467bd'),
                ]
                for f0, label, col in marks_raw:
                    if not (f0 and f0 > 0):
                        continue
                    try:
                        f0_f = float(f0)
                    except Exception:
                        continue
                    if zmin is not None and (f0_f < zmin or f0_f > zmax):
                        continue
                    visible_marks.append((f0_f / freq_scale, label, col))
            except Exception:
                visible_marks = []

            hover_precision = ".3f" if unit_mode in {"vel_mm", "acc_g"} else ".3e"

            fig_time_title = "Señal en el tiempo"
            fig_freq_title = "FFT (Velocidad)"

            def _build_matplotlib_time_freq_chart() -> Optional[MatplotlibChart]:
                try:
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
                except Exception:
                    return None

                try:
                    ax1.plot(t_segment, _y_time, color=self.time_plot_color, linewidth=2)
                    ax1.set_title(fig_time_title)
                    ax1.set_xlabel("Tiempo (s)")
                    ax1.set_ylabel(_ylabel)
                    try:
                        text_color = "white" if self.is_dark_mode else "black"
                        ax1.text(
                            0.02,
                            0.95,
                            _rms_text,
                            transform=ax1.transAxes,
                            va="top",
                            color=text_color,
                        )
                    except Exception:
                        pass
                except Exception:
                    pass

                try:
                    xplot_disp_local = xplot_disp if xplot_disp is not None else []
                    yplot_local = yplot if yplot is not None else []
                    ax2.plot(xplot_disp_local, yplot_local, color=self.fft_plot_color, linewidth=2)
                    ax2.fill_between(xplot_disp_local, yplot_local, alpha=0.3, color=self.fft_plot_color)
                    ax2.set_title(fig_freq_title)
                    ax2.set_xlabel(f"Frecuencia ({freq_unit})")
                    ax2.set_ylabel("Velocidad [mm/s]")
                    if yplot_dbv is not None:
                        try:
                            ax2_db = ax2.twinx()
                            ax2_db.plot(xplot_disp_local, yplot_dbv, color="#9b59b6", linewidth=1.6, linestyle="--")
                            ax2_db.set_ylabel("Nivel [dBV]")
                            lower = db_axis_min
                            upper = db_axis_max
                            if lower is not None or upper is not None:
                                try:
                                    current_min = float(np.nanmin(yplot_dbv)) if len(yplot_dbv) > 0 else None
                                except Exception:
                                    current_min = None
                                try:
                                    current_max = float(np.nanmax(yplot_dbv)) if len(yplot_dbv) > 0 else None
                                except Exception:
                                    current_max = None
                                y_lower = lower if lower is not None else current_min
                                y_upper = upper if upper is not None else current_max
                                if (
                                    y_lower is not None
                                    and y_upper is not None
                                    and np.isfinite(y_lower)
                                    and np.isfinite(y_upper)
                                    and y_upper != y_lower
                                ):
                                    ax2_db.set_ylim(y_lower, y_upper)
                        except Exception:
                            pass
                    if peak_points:
                        try:
                            px, py = zip(*peak_points)
                            ax2.scatter(px, py, color="#e74c3c", s=30, zorder=5)
                            self._place_annotations(ax2, peak_points, peak_labels, color="#e74c3c")
                        except Exception:
                            pass
                    if visible_marks:
                        try:
                            for pos, label, color_hex in visible_marks:
                                try:
                                    ax2.axvline(pos, color=color_hex, linestyle="--", alpha=0.85, linewidth=1.2)
                                except Exception:
                                    continue
                            zoom_scaled = None if zmin is None else (zmin / freq_scale, zmax / freq_scale)
                            self._draw_frequency_markers(ax2, visible_marks, zoom_scaled)
                        except Exception:
                            pass
                    try:
                        if zmin is not None:
                            ax2.set_xlim(left=zmin / freq_scale, right=zmax / freq_scale)
                        elif fmax_ui and fmax_ui > 0:
                            ax2.set_xlim(left=0.0, right=float(fmax_ui) / freq_scale)
                    except Exception:
                        pass
                except Exception:
                    pass

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        fig.tight_layout()
                except Exception:
                    pass

                chart_local = MatplotlibChart(fig, expand=True, isolated=True)
                plt.close(fig)
                return chart_local

            chart = None
            plotly_error: Optional[Exception] = None
            if self.interactive_charts_enabled:
                try:
                    template_name = "plotly_dark" if self.is_dark_mode else "plotly_white"
                    plotly_fig = make_subplots(
                        rows=2,
                        cols=1,
                        shared_xaxes=False,
                        vertical_spacing=0.08,
                        specs=[[{}], [{"secondary_y": True}]],
                    )
    
                    time_hover = "Tiempo: %{x:.3f} s"
                    time_hover += f"<br>{_ylabel}: %{{y:{hover_precision}}}"
                    time_hover += "<extra></extra>"
                    plotly_fig.add_trace(
                        go.Scatter(
                            x=t_segment,
                            y=_y_time,
                            mode="lines",
                            line=dict(color=self.time_plot_color, width=2),
                            name=fig_time_title,
                            hovertemplate=time_hover,
                        ),
                        row=1,
                        col=1,
                    )
    
                    annotation_color = "#ffffff" if self.is_dark_mode else "#000000"
                    plotly_fig.add_annotation(
                        x=0.01,
                        y=0.98,
                        xref="paper",
                        yref="paper",
                        text=_rms_text,
                        showarrow=False,
                        align="left",
                        font=dict(color=annotation_color, size=12),
                        bgcolor="rgba(0,0,0,0)",
                    )
    
                    if xplot is not None and yplot is not None:
                        fft_custom = None
                        try:
                            fft_custom = np.column_stack(
                                [
                                    np.asarray(xplot, dtype=float),
                                    np.asarray(xplot, dtype=float) * 60.0,
                                ]
                            )
                        except Exception:
                            fft_custom = None
    
                        fft_hover = f"Frecuencia ({freq_unit}): %{{x:.3f}}"
                        if fft_custom is not None:
                            fft_hover += "<br>Frecuencia (Hz): %{customdata[0]:.3f}"
                        fft_hover += "<br>Velocidad: %{y:.3f} mm/s"
                        if fft_custom is not None:
                            fft_hover += "<br>RPM: %{customdata[1]:.0f}"
                        fft_hover += "<extra></extra>"
    
                        plotly_fig.add_trace(
                            go.Scatter(
                                x=xplot_disp,
                                y=yplot,
                                mode="lines",
                                line=dict(color=self.fft_plot_color, width=2),
                                fill="tozeroy",
                                name=fig_freq_title,
                                customdata=fft_custom,
                                hovertemplate=fft_hover,
                            ),
                            row=2,
                            col=1,
                            secondary_y=False,
                        )
    
                    if peak_points:
                        peak_x_disp = [pt for pt, _ in peak_points]
                        peak_y = [amp for _, amp in peak_points]
                        plotly_fig.add_trace(
                            go.Scatter(
                                x=peak_x_disp,
                                y=peak_y,
                                mode="markers",
                                marker=dict(color="#e74c3c", size=9),
                                name="Picos principales",
                                hovertext=peak_labels,
                                hoverinfo="text",
                            ),
                            row=2,
                            col=1,
                            secondary_y=False,
                        )
    
                    if yplot_dbv is not None:
                        plotly_fig.add_trace(
                            go.Scatter(
                                x=xplot_disp,
                                y=yplot_dbv,
                                mode="lines",
                                line=dict(color="#9b59b6", width=1.6, dash="dash"),
                                name="Nivel [dBV]",
                                hovertemplate="Frecuencia: %{x:.3f}<br>Nivel: %{y:.2f} dBV<extra></extra>",
                            ),
                            row=2,
                            col=1,
                            secondary_y=True,
                        )
    
                    if visible_marks:
                        try:
                            fft_ymax = float(np.nanmax(yplot)) if yplot is not None and len(yplot) > 0 else 0.0
                        except Exception:
                            fft_ymax = 0.0
                        if not np.isfinite(fft_ymax) or fft_ymax <= 0:
                            fft_ymax = 1.0
                        for pos, label, color_hex in visible_marks:
                            plotly_fig.add_shape(
                                type="line",
                                x0=pos,
                                x1=pos,
                                y0=0,
                                y1=fft_ymax,
                                xref="x2",
                                yref="y2",
                                line=dict(color=color_hex, dash="dash", width=1.2, opacity=0.85),
                            )
                            plotly_fig.add_annotation(
                                x=pos,
                                y=fft_ymax,
                                xref="x2",
                                yref="y2",
                                text=label,
                                showarrow=False,
                                font=dict(color=color_hex, size=11),
                                yanchor="bottom",
                            )
    
                    plotly_fig.update_xaxes(title_text="Tiempo (s)", row=1, col=1)
                    plotly_fig.update_yaxes(title_text=_ylabel, row=1, col=1)
                    plotly_fig.update_xaxes(title_text=f"Frecuencia ({freq_unit})", row=2, col=1)
                    plotly_fig.update_yaxes(title_text="Velocidad [mm/s]", row=2, col=1, secondary_y=False)
    
                    if yplot_dbv is not None:
                        plotly_fig.update_yaxes(title_text="Nivel [dBV]", row=2, col=1, secondary_y=True)
                        try:
                            max_db = float(np.nanmax(yplot_dbv)) if len(yplot_dbv) > 0 else 0.0
                            min_db = float(np.nanmin(yplot_dbv)) if len(yplot_dbv) > 0 else 0.0
                        except Exception:
                            max_db = 0.0
                            min_db = 0.0
                        lower = db_axis_min if db_axis_min is not None else min_db
                        upper = db_axis_max if db_axis_max is not None else max_db
                        if np.isfinite(lower) and np.isfinite(upper) and upper != lower:
                            plotly_fig.update_yaxes(range=[lower, upper], row=2, col=1, secondary_y=True)
    
                    if zmin is not None:
                        plotly_fig.update_xaxes(range=[zmin / freq_scale, zmax / freq_scale], row=2, col=1)
                    elif fmax_ui and fmax_ui > 0:
                        plotly_fig.update_xaxes(range=[0.0, float(fmax_ui) / freq_scale], row=2, col=1)
    
                    plotly_fig.update_layout(
                        template=template_name,
                        height=650,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
                        margin=dict(l=60, r=40, t=60, b=60),
                        hovermode="x unified",
                        title=dict(text="Análisis tiempo / frecuencia", x=0.5),
                    )
    
                    chart = PlotlyChart(plotly_fig, expand=True)
                except Exception as exc:
                    plotly_error = exc
                    chart = None
    
            if chart is None:
                chart = _build_matplotlib_time_freq_chart()
                if chart is None:
                    chart = ft.Text("No fue posible renderizar la gráfica principal.")
                if plotly_error is not None:
                    try:
                        print(f"[WARN] Plotly chart fallback due to error: {plotly_error}")
                    except Exception:
                        pass

            # Gráfica separada de Envolvente con picos
            env_chart = None
            try:
                xf_env = res.get('envelope', {}).get('f_hz', None)
                env_amp = res.get('envelope', {}).get('amp', None)
                peaks_env = res.get('envelope', {}).get('peaks', [])
                if xf_env is not None and env_amp is not None and len(xf_env) > 0:
                    if hide_lf:
                        m_env = xf_env >= max(0.0, fc)
                    else:
                        m_env = np.ones_like(xf_env, dtype=bool)
                    if fmax_ui and fmax_ui > 0:
                        m_env = m_env & (xf_env <= fmax_ui)
                    if zmin is not None:
                        m_env = m_env & (xf_env >= zmin) & (xf_env <= zmax)
                    xenv = xf_env[m_env]
                    yenv = env_amp[m_env]
                    env_fig, env_ax = plt.subplots(figsize=(14, 3))
                    xenv_disp = np.asarray(xenv, dtype=float) / freq_scale
                    env_ax.plot(xenv_disp, yenv, color="#e67e22", linewidth=1.6)
                    env_ax.set_title("Espectro de Envolvente")
                    env_ax.set_xlabel(f"Frecuencia ({freq_unit})")
                    env_ax.set_ylabel("Amp [a.u.]")
                    # Picos anotados
                    try:
                        vis_peaks = []
                        for p in (peaks_env or []):
                            f0 = float(p.get('f_hz', 0.0))
                            a0 = float(p.get('amp', 0.0))
                            if f0 <= 0 or a0 <= 0:
                                continue
                            if hide_lf and f0 < max(0.0, fc):
                                continue
                            if fmax_ui and fmax_ui > 0 and f0 > fmax_ui:
                                continue
                            if zmin is not None and (f0 < zmin or f0 > zmax):
                                continue
                            vis_peaks.append((f0, a0))
                        if vis_peaks:
                            filtered = vis_peaks
                            if zmin is not None:
                                tmp = [(f0, a0) for f0, a0 in vis_peaks if zmin <= f0 <= zmax]
                                if tmp:
                                    filtered = tmp
                            pfx, pfy = zip(*filtered)
                            pfx_disp = [float(f0) / freq_scale for f0 in pfx]
                            env_ax.scatter(pfx_disp, pfy, color="#c0392b", s=24, zorder=5)
                            peak_points = [(float(f0) / freq_scale, float(a0)) for f0, a0 in filtered]
                            peak_labels = [f"{float(f0) / freq_scale:.2f} {freq_unit}" for f0, _ in filtered]
                            self._place_annotations(env_ax, peak_points, peak_labels, color="#c0392b", text_color="#c0392b")
                    except Exception:
                        pass
                    # Líneas guía teóricas
                    try:
                        bpfo = self._fldf(getattr(self, 'bpfo_field', None))
                        bpfi = self._fldf(getattr(self, 'bpfi_field', None))
                        bsf  = self._fldf(getattr(self, 'bsf_field', None))
                        ftf  = self._fldf(getattr(self, 'ftf_field', None))
                        marks_raw = [
                            (bpfo, 'BPFO', '#1f77b4'),
                            (bpfi, 'BPFI', '#ff7f0e'),
                            (bsf,  'BSF',  '#2ca02c'),
                            (ftf,  'FTF',  '#9467bd'),
                        ]
                        visible_marks = []
                        for f0, label, col in marks_raw:
                            if not (f0 and f0 > 0):
                                continue
                            try:
                                f0_f = float(f0)
                            except Exception:
                                continue
                            if zmin is not None and (f0_f < zmin or f0_f > zmax):
                                continue
                            try:
                                env_ax.axvline(f0_f / freq_scale, color=col, linestyle='--', alpha=0.85, linewidth=1.2)
                            except Exception:
                                pass
                            visible_marks.append((f0_f / freq_scale, label, col))
                        zoom_scaled_env = None if zmin is None else (zmin / freq_scale, zmax / freq_scale)
                        self._draw_frequency_markers(env_ax, visible_marks, zoom_scaled_env)
                    except Exception:
                        pass
                    env_chart = MatplotlibChart(env_fig, expand=True, isolated=True)
                    plt.close(env_fig)
            except Exception:
                env_chart = None

            orbit_chart = None
            try:
                orbit_enabled = False
                if getattr(self, 'orbit_cb', None):
                    orbit_enabled = bool(getattr(self.orbit_cb, 'value', False))
                else:
                    orbit_enabled = bool(getattr(self, 'orbit_plot_enabled', False))
                if orbit_enabled:
                    x_col = getattr(self, 'orbit_x_dd', None).value if getattr(self, 'orbit_x_dd', None) else self.orbit_axis_x_pref
                    y_col = getattr(self, 'orbit_y_dd', None).value if getattr(self, 'orbit_y_dd', None) else self.orbit_axis_y_pref
                    if x_col and y_col and x_col in self.current_df.columns and y_col in self.current_df.columns:
                        try:
                            x_seg = segment_df[x_col].to_numpy()
                            y_seg = segment_df[y_col].to_numpy()
                        except Exception:
                            x_seg = self.current_df[x_col].to_numpy()
                            y_seg = self.current_df[y_col].to_numpy()
                        t_orbit, x_orbit, y_orbit = self._trim_orbit_window(t_segment, x_seg, y_seg)
                        orbit_fig = self._generate_orbit_figure(
                            t_orbit,
                            x_orbit,
                            y_orbit,
                            x_col,
                            y_col,
                            fc,
                            hide_lf,
                            fmax_ui,
                            self.is_dark_mode,
                        )
                        if orbit_fig is not None:
                            orbit_chart = MatplotlibChart(orbit_fig, expand=True, isolated=True)
                            plt.close(orbit_fig)
            except Exception:
                orbit_chart = None

            runup_chart = None
            try:
                if getattr(self, 'runup_3d_cb', None) and getattr(self.runup_3d_cb, 'value', False):
                    zoom_tuple = (zmin, zmax) if zmin is not None else None
                    try:
                        full_t_uniform, full_acc_uniform, _, _ = self._prepare_segment_for_analysis(t, signal, fft_signal_col)
                    except Exception:
                        full_t_uniform, full_acc_uniform = None, None
                    base_t = full_t_uniform if full_t_uniform is not None and full_acc_uniform is not None else t_segment
                    base_acc = full_acc_uniform if full_t_uniform is not None and full_acc_uniform is not None else acc_segment
                    try:
                        runup_mask, _, _ = self._resolve_runup_period(base_t)
                    except Exception:
                        runup_mask = None
                    if (
                        runup_mask is not None
                        and runup_mask.size == base_t.size
                        and np.count_nonzero(runup_mask) >= 2
                    ):
                        runup_t = base_t[runup_mask]
                        runup_acc = base_acc[runup_mask]
                    else:
                        runup_t = base_t
                        runup_acc = base_acc
                    runup_fig = self._generate_runup_3d_figure(
                        runup_t,
                        runup_acc,
                        fc,
                        hide_lf,
                        fmax_ui,
                        zoom_tuple,
                        self.is_dark_mode,
                        base_t,
                        base_acc,
                        self.fft_window_type,
                    )
                    if runup_fig is not None:
                        runup_chart = MatplotlibChart(runup_fig, expand=True, isolated=True)
                        plt.close(runup_fig)
            except Exception:
                runup_chart = None


            # --- Gráficas auxiliares ---

            aux_plots = []

            for cb, color_dd, style_dd in self.aux_controls:

                if cb.value:

                    aux_fig, aux_ax = plt.subplots(figsize=(8, 2))

                    aux_ax.plot(

                        self.current_df[time_col],

                        self.current_df[cb.label],

                        color=color_dd.value,

                        linestyle=style_dd.value,

                        linewidth=2,

                        label=cb.label

                    )

                    aux_ax.set_title(f"{cb.label} vs Tiempo")

                    aux_ax.legend()

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        aux_fig.tight_layout()

                    aux_plots.append(MatplotlibChart(aux_fig, expand=True, isolated=True))
                    plt.close(aux_fig)

            def _build_axis_controls() -> List[ft.Control]:
                controls: List[ft.Control] = []
                if not axis_summaries:
                    return controls
                controls.append(
                    ft.Text(
                        "Severidad por eje (RMS velocidad)",
                        size=14,
                        weight="bold",
                        text_align=ft.TextAlign.LEFT,
                    )
                )
                for entry in axis_summaries:
                    try:
                        value_txt = f"{float(entry.get('rms_mm_s', 0.0)):.3f} mm/s"
                    except Exception:
                        value_txt = "N/D"
                    color_hex = entry.get("color", "#7f8c8d")
                    controls.append(
                        ft.Container(
                            expand=True,
                            content=ft.Column(
                                [
                                    ft.Row(
                                        [
                                            ft.Icon(
                                                ft.Icons.TIMELAPSE,
                                                size=16,
                                                color=color_hex,
                                            ),
                                            ft.Text(
                                                entry.get("name", "Eje"),
                                                weight="bold" if entry.get("is_global") else None,
                                                expand=True,
                                                max_lines=1,
                                                overflow=ft.TextOverflow.ELLIPSIS,
                                                text_align=ft.TextAlign.LEFT,
                                            ),
                                        ],
                                        spacing=8,
                                        alignment="start",
                                    ),
                                    ft.Row(
                                        [
                                            ft.Text(
                                                value_txt,
                                                weight="bold",
                                                text_align=ft.TextAlign.LEFT,
                                            ),
                                            ft.Text(
                                                entry.get("iso_label", "N/D"),
                                                color=color_hex,
                                                expand=True,
                                                max_lines=2,
                                                overflow=ft.TextOverflow.ELLIPSIS,
                                                text_align=ft.TextAlign.RIGHT,
                                            ),
                                        ],
                                        spacing=8,
                                        alignment="spaceBetween",
                                        vertical_alignment="center",
                                    ),
                                ],
                                spacing=6,
                            ),
                            border=ft.border.all(1, color_hex),
                            border_radius=8,
                            padding=ft.padding.symmetric(horizontal=12, vertical=10),
                        )
                    )
                return controls

            axis_summary_controls_exec = _build_axis_controls()
            axis_summary_controls_main = _build_axis_controls()

            # --- Resumen Ejecutivo (mm/s, formal al inicio) ---
            try:
                sev_label, sev_color = primary_label, primary_color
            except Exception:
                sev_label, sev_color = "N/D", "#7f8c8d"
            exec_findings_all = list(findings)
            exec_findings = self._select_main_findings(exec_findings_all)
            if not exec_findings:
                exec_findings = ["Sin anomalías evidentes según reglas actuales."]
            resumen_exec = ft.Container(
                content=ft.Column(
                    [
                        ft.Row(
                            [
                                ft.Icon(ft.Icons.ANALYTICS, color=self._accent_ui()),
                                ft.Text(
                                    "Resumen Ejecutivo",
                                    size=18,
                                    weight="bold",
                                    text_align=ft.TextAlign.LEFT,
                                ),
                            ],
                            spacing=8,
                            alignment="start",
                        ),
                        ft.Container(
                            content=ft.Row(
                                [
                                ft.Icon(ft.Icons.SPEED, color=sev_color),
                                    ft.Text(f"Clasificación ISO: {sev_label}", weight="bold"),
                                ],
                                spacing=8,
                                alignment="start",
                            ),
                            bgcolor=ft.Colors.with_opacity(0.12, sev_color),
                            border_radius=20,
                            padding=ft.padding.symmetric(horizontal=14, vertical=8),
                        ),
                        ft.Text(
                            f"RMS global (mm/s): {primary_rms_mm:.3f}",
                            weight="bold",
                            text_align=ft.TextAlign.LEFT,
                        ),
                        ft.Text(
                            f"Frecuencia dominante: {dom_freq:.2f} Hz",
                            text_align=ft.TextAlign.LEFT,
                        ),
                        self._build_severity_traffic_light(primary_rms_mm),
                        self._build_spectral_balance_widget(frac_low, frac_mid, frac_high),
                        *axis_summary_controls_exec,
                        ft.Text(
                            "Hallazgos clave",
                            weight="bold",
                            text_align=ft.TextAlign.LEFT,
                        ),
                        ft.Column(
                            [
                                ft.Row(
                                    [
                                        ft.Icon(ft.Icons.CHEVRON_RIGHT, color=self._accent_ui(), size=18),
                                        ft.Text(
                                            text,
                                            expand=True,
                                            text_align=ft.TextAlign.LEFT,
                                            max_lines=4,
                                            overflow=ft.TextOverflow.ELLIPSIS,
                                        ),
                                    ],
                                    spacing=6,
                                    alignment="start",
                                )
                                for text in exec_findings
                            ],
                            spacing=4,
                            tight=True,
                        ),
                    ],
                    spacing=12,
                ),
                bgcolor=ft.Colors.with_opacity(0.06, self._accent_ui()),
                border_radius=12,
                padding=ft.padding.all(16),
            )

            # --- Panel resumen + diagnóstico ---

            resumen = ft.Container(
                content=ft.Column(
                    [
                        ft.Row(
                            [
                                ft.Icon(ft.Icons.INSIGHTS, color=self._accent_ui()),
                                ft.Text(
                                    "Resumen del análisis",
                                    size=18,
                                    weight="bold",
                                    text_align=ft.TextAlign.LEFT,
                                ),
                            ],
                            spacing=8,
                            alignment="start",
                        ),
                        ft.Text(
                            f"Periodo analizado: {start_t:.2f}s – {end_t:.2f}s",
                            text_align=ft.TextAlign.LEFT,
                        ),
                        ft.Text(
                            f"Frecuencia dominante: {dom_freq:.2f} Hz",
                            text_align=ft.TextAlign.LEFT,
                        ),
                        ft.Text(
                            f"RMS velocidad global: {primary_rms_mm:.3f} mm/s",
                            text_align=ft.TextAlign.LEFT,
                        ),
                        self._build_severity_traffic_light(primary_rms_mm),
                        self._build_spectral_balance_widget(frac_low, frac_mid, frac_high),
                        *axis_summary_controls_main,
                        ft.Text(
                            "Crest factor (aceleración): "
                            f"{(float(np.max(np.abs(acc_segment))) / (float(self._calculate_rms(acc_segment)) + 1e-12)):.2f}",
                            text_align=ft.TextAlign.LEFT,
                        ),
                        ft.Divider(),
                        ft.Row(
                            [
                                ft.Icon(ft.Icons.FACT_CHECK, color=self._accent_ui()),
                                ft.Text(
                                    "Diagnóstico automático (baseline)",
                                    size=16,
                                    weight="bold",
                                    text_align=ft.TextAlign.LEFT,
                                ),
                            ],
                            spacing=8,
                            alignment="start",
                        ),
                        ft.Column(
                            [
                                ft.Row(
                                    [
                                        ft.Icon(ft.Icons.CHEVRON_RIGHT, color=self._accent_ui(), size=18),
                                        ft.Text(
                                            text,
                                            expand=True,
                                            text_align=ft.TextAlign.LEFT,
                                            max_lines=4,
                                            overflow=ft.TextOverflow.ELLIPSIS,
                                        ),
                                    ],
                                    spacing=6,
                                    alignment="start",
                                )
                                for text in findings
                            ],
                            spacing=4,
                            tight=True,
                        ),
                    ],
                    spacing=12,
                ),
                bgcolor=ft.Colors.with_opacity(0.06, self._accent_ui()),
                border_radius=12,
                padding=ft.padding.all(16),
            )



            # Nota de filtro visual FFT para mayor precisión en la interpretación
            try:
                _fc = float(self.lf_cutoff_field.value) if getattr(self, 'lf_cutoff_field', None) and getattr(self.lf_cutoff_field, 'value', '') else 0.5
            except Exception:
                _fc = 0.5
            try:
                _hide_lf = bool(getattr(self, 'hide_lf_cb', None).value)
            except Exception:
                _hide_lf = True
            _fft_filter_note = f"Filtro visual FFT: oculta < {_fc:.2f} Hz" if _hide_lf else "Filtro visual FFT: sin ocultar"

            # Recalcular explicaciones con helper unificado (evita divergencias)
            exp_lines: List[str] = []
            try:
                exp_lines = self._build_explanations(res, findings)
            except Exception:
                exp_lines = []
            if not exp_lines:
                exp_lines = ["Revisión ISO 10816/20816 – sin información disponible."]

            # --- Contenedor con scroll (en Column, no en Container) ---

            ml_card = self._build_ml_summary_card(res.get("ml"))
            column_controls: List[ft.Control] = [resumen_exec]
            if ml_card is not None:
                column_controls.append(ml_card)

            column_controls.append(
                ft.Container(
                        content=ft.Column(
                            [
                                ft.Row(
                                    [
                                        ft.Icon(ft.Icons.LIGHTBULB, color=self._accent_ui()),
                                        ft.Text(
                                            "Explicación y revisión sugerida",
                                            size=16,
                                            weight="bold",
                                            text_align=ft.TextAlign.LEFT,
                                        ),
                                    ],
                                    spacing=8,
                                    alignment="start",
                                ),
                                ft.Column(
                                    [
                                        ft.Row(
                                            [
                                                ft.Icon(ft.Icons.CHECK_CIRCLE, color=self._accent_ui(), size=18),
                                                ft.Text(
                                                    line,
                                                    expand=True,
                                                    text_align=ft.TextAlign.LEFT,
                                                    max_lines=4,
                                                    overflow=ft.TextOverflow.ELLIPSIS,
                                                ),
                                            ],
                                            spacing=6,
                                            alignment="start",
                                        )
                                        for line in exp_lines
                                    ],
                                    spacing=6,
                                    tight=True,
                                ),
                            ],
                            spacing=12,
                        ),
                        bgcolor=ft.Colors.with_opacity(0.06, self._accent_ui()),
                        border_radius=12,
                        padding=ft.padding.all(16),
                    )
            )

            column_controls.append(ft.Text(_fft_filter_note, text_align=ft.TextAlign.LEFT))
            if chart is not None:
                column_controls.append(chart)

            if runup_chart:
                column_controls.append(runup_chart)
            if orbit_chart:
                column_controls.append(orbit_chart)
            if 'env_chart' in locals() and env_chart:
                column_controls.append(env_chart)
            column_controls.extend(aux_plots)

            return ft.Container(
                expand=True,
                content=ft.Column(
                    controls=column_controls,
                    spacing=20,
                    scroll="auto",   # 👈 scroll vertical aquí (válido en Column)
                    expand=True
                )
            )
        except Exception as e:
            try:
                import traceback
                tb = traceback.format_exc()
                self._log(f"Error en análisis: {e} | {tb}")
            except Exception:
                self._log(f"Error en análisis: {e}")
            return ft.Text(f"Error en análisis: {e}", size=14, color="#e74c3c")

            return ft.Text(f"Error en análisis: {e}", size=14, color="#e74c3c")



    def _build_reports_view(self):
        return self._build_reports_view_impl()

    # ===== Vista de rodamientos =====
    def _build_bearings_view(self):
        # Crear/attach controles si no existen
        if not getattr(self, 'bearing_list_view', None):
            self.bearing_list_view = ft.ListView(expand=True, spacing=4, padding=4)
        if not getattr(self, 'br_model_field_dlg', None):
            self.br_model_field_dlg = ft.TextField(label="Modelo", width=220)
            self.br_n_field_dlg = ft.TextField(label="# Elementos (n)", width=150)
            self.br_d_mm_field_dlg = ft.TextField(label="d (mm)", width=120)
            self.br_D_mm_field_dlg = ft.TextField(label="D (mm)", width=120)
            self.br_theta_deg_field_dlg = ft.TextField(label="Ángulo (°)", width=120, value="0")
        # Refresh list content
        self._refresh_bearing_list_ui()
        # Panel detalle
        detail_col = ft.Column([
            ft.Text("Detalle del rodamiento", size=16, weight="bold"),
            self.br_model_field_dlg,
            ft.Row([self.br_n_field_dlg, self.br_d_mm_field_dlg], spacing=10),
            ft.Row([self.br_D_mm_field_dlg, self.br_theta_deg_field_dlg], spacing=10),
            ft.Row([
                ft.OutlinedButton("Nuevo", icon=ft.Icons.ADD_ROUNDED, on_click=self._bearing_new_click),
                ft.ElevatedButton("Guardar", icon=ft.Icons.SAVE_ROUNDED, on_click=self._bearing_save_click),
                ft.ElevatedButton("Usar en análisis", icon=ft.Icons.CHECK_CIRCLE_ROUNDED, on_click=self._bearing_use_and_go),
                ft.OutlinedButton("Ir a Análisis", icon=ft.Icons.ARROW_FORWARD_ROUNDED, on_click=lambda e: self._select_menu("analysis", force_rebuild=True)),
            ], spacing=10),
        ], spacing=10, expand=True, alignment="start", scroll="auto")

        # Buscador de rodamientos
        if not getattr(self, 'bearing_search', None):
            self.bearing_search = ft.TextField(hint_text="Buscar por modelo o n...", on_change=lambda e: self._refresh_bearing_list_ui(), dense=True)
        # Tabs por marca
        if not getattr(self, 'bearing_tabs', None):
            self.bearing_tabs = ft.Tabs(tabs=[ft.Tab(text=n) for n in self._bearing_brand_names()], selected_index=0, on_change=self._on_bearing_tab_change)
        else:
            self._rebuild_bearing_tabs()
        # Checkbox favoritos sólo
        if not getattr(self, 'bearing_favs_only_cb', None):
            self.bearing_favs_only_cb = ft.Checkbox(label="Mostrar favoritos", value=bool(self.bearing_show_favs_only), on_change=lambda e: self._toggle_bearing_favs_filter())
        list_col = ft.Column([
            ft.Text("Listado de rodamientos", size=16, weight="bold"),
            self.bearing_tabs,
            self.bearing_search,
            self.bearing_favs_only_cb,
            self.bearing_list_view,
        ], spacing=10, expand=True, alignment="start")

        return ft.Column([
            ft.Row([
                ft.Text("Gestor de Rodamientos", size=24, weight="bold"),
                ft.Row([
                    ft.OutlinedButton("Importar CSV", icon=ft.Icons.UPLOAD_FILE_ROUNDED, on_click=self._bearing_open_csv_picker),
                    ft.ElevatedButton("Analizar", icon=ft.Icons.ANALYTICS_ROUNDED, on_click=self._bearing_analyze_click),
                ], spacing=10),
            ], alignment="space_between"),
            ft.Row([
                ft.Container(content=list_col, width=500, height=600, padding=10, bgcolor=ft.Colors.with_opacity(0.03, "white" if self.is_dark_mode else "black"), border_radius=10, alignment=ft.alignment.top_left),
                ft.Container(content=detail_col, height=600, expand=True, padding=10, bgcolor=ft.Colors.with_opacity(0.03, "white" if self.is_dark_mode else "black"), border_radius=10, alignment=ft.alignment.top_left),
            ], spacing=16),
        ], expand=True, scroll="auto")

    # Mantener implementación original de reports separada
    def _build_reports_view_impl(self):

        self.report_search = ft.TextField(

            hint_text="Buscar por nombre...",

            expand=True,

            on_change=lambda e: self._refresh_report_list_scandir()

        )

        self.report_list = ft.ListView(expand=1, spacing=8, padding=10)

        # Filtro de favoritos para reportes
        self.report_favs_only_cb = ft.Checkbox(label="Mostrar favoritos", value=bool(self.report_show_favs_only), on_change=lambda e: self._toggle_reports_fav_filter())



        # Render inicial

        self._refresh_report_list_scandir()



        return ft.Column(

            controls=[

                ft.Text("📑 Reportes Generados", size=24, weight="bold"),

                ft.Row([self.report_search, self.report_favs_only_cb], alignment="spaceBetween"),

                ft.Container(content=self.report_list, expand=True, border_radius=10, padding=10),

            ],

            expand=True

        )



    def _refresh_report_list(self):

        self.report_list.controls.clear()

        if not hasattr(self, "generated_reports") or not self.generated_reports:

            self.report_list.controls.append(ft.Text("Aún no hay reportes generados.", size=14))

        else:

            query = self.report_search.value.lower() if self.report_search.value else ""

            for path in reversed(self.generated_reports):

                name = os.path.basename(path)

                if query in name.lower():

                    self.report_list.controls.append(

                        ft.Container(

                            content=ft.Row(

                                controls=[

                                    ft.Icon(ft.Icons.PICTURE_AS_PDF_ROUNDED, size=30, color="#e74c3c"),

                                    ft.Text(name, expand=True),

                                    ft.IconButton(icon=ft.Icons.FOLDER_OPEN_ROUNDED, on_click=lambda e,p=path: os.startfile(os.path.dirname(p))),

                                    ft.IconButton(icon=ft.Icons.OPEN_IN_NEW_ROUNDED, on_click=lambda e,p=path: os.startfile(p)),

                                ]

                            ),

                            padding=10,

                            border_radius=8,

                            bgcolor=ft.Colors.with_opacity(0.05, self._accent_ui()),

                        )

                    )

        if self.report_list.page:

            self.report_list.update()

    def _refresh_report_list_scandir(self):
        self.report_list.controls.clear()
        try:
            reports_dir = os.path.join(os.getcwd(), "reports")
            os.makedirs(reports_dir, exist_ok=True)
            files = []
            for fn in os.listdir(reports_dir):
                if fn.lower().endswith('.pdf'):
                    p = os.path.join(reports_dir, fn)
                    try:
                        mt = os.path.getmtime(p)
                    except Exception:
                        mt = 0.0
                    files.append((p, mt))
            if not files:
                self.report_list.controls.append(ft.Text("Aún no hay reportes generados.", size=14))
            else:
                query = (self.report_search.value.lower() if getattr(self, 'report_search', None) and self.report_search.value else "")
                if query:
                    files = [(p, mt) for (p, mt) in files if query in os.path.basename(p).lower()]
                # Filtrar por favoritos si está activo
                try:
                    if getattr(self, 'report_favs_only_cb', None) and getattr(self.report_favs_only_cb, 'value', False):
                        favs = getattr(self, 'report_favorites', {}) or {}
                        files = [(p, mt) for (p, mt) in files if bool(favs.get(p, False))]
                except Exception:
                    pass
                from datetime import datetime as _dt
                groups = {}
                for p, mt in files:
                    base = os.path.basename(p)
                    try:
                        date_key = _dt.strptime(base[:10], "%Y-%m-%d").strftime("%Y-%m-%d")
                    except Exception:
                        date_key = _dt.fromtimestamp(mt).strftime("%Y-%m-%d")
                    groups.setdefault(date_key, []).append((p, mt))
                for date_key in sorted(groups.keys(), reverse=True):
                    items = sorted(groups[date_key], key=lambda x: x[1], reverse=True)
                    self.report_list.controls.append(ft.Text(date_key, weight="bold"))
                    for p, _mt in items:
                        name = os.path.basename(p)
                        # Estrella de favoritos
                        is_fav = False
                        try:
                            is_fav = bool(getattr(self, 'report_favorites', {}).get(p, False))
                        except Exception:
                            is_fav = False
                        star_icon = ft.Icons.STAR if is_fav else ft.Icons.STAR_BORDER_ROUNDED
                        star_color = "#f1c40f" if is_fav else "#bdc3c7"
                        self.report_list.controls.append(
                            ft.Container(
                                content=ft.Row(
                                    controls=[
                                        ft.IconButton(icon=star_icon, icon_color=star_color, tooltip="Marcar favorito", on_click=lambda e,pp=p: self._toggle_report_favorite(pp)),
                                        ft.Icon(ft.Icons.PICTURE_AS_PDF_ROUNDED, size=30, color="#e74c3c"),
                                        ft.Text(name, expand=True),
                                        ft.IconButton(icon=ft.Icons.FOLDER_OPEN_ROUNDED, on_click=lambda e,pp=p: os.startfile(os.path.dirname(pp))),
                                        ft.IconButton(icon=ft.Icons.OPEN_IN_NEW_ROUNDED, on_click=lambda e,pp=p: os.startfile(pp)),
                                    ]
                                ),
                                padding=10,
                                border_radius=8,
                                bgcolor=ft.Colors.with_opacity(0.05, self._accent_ui()),
                            )
                        )
        except Exception:
            self.report_list.controls.append(ft.Text("No se pudieron listar los reportes.", size=14))
        if self.report_list.page:
            self.report_list.update()



    def _build_settings_view(self):

        return ft.Column(

            controls=[

                ft.Text("Configuración", size=24, weight="bold"),

                ft.Container(height=20),

                ft.Container(

                    content=ft.Column(

                        spacing=20,

                        controls=[

                            ft.Container(

                                content=ft.Row(

                                    alignment="space_between",

                                    controls=[

                                        ft.Row(

                                            controls=[

                                                ft.Icon(ft.Icons.DARK_MODE_ROUNDED, size=24),

                                                ft.Text("Tema oscuro", size=16),

                                            ],

                                            spacing=15

                                        ),

                                        ft.Switch(

                                            value=self.is_dark_mode,

                                            on_change=self._toggle_theme_switch,

                                            active_color=self._accent_ui(),

                                        ),

                                    ]

                                ),

                                padding=15,

                                border_radius=10,

                                bgcolor=ft.Colors.with_opacity(0.05, self._accent_ui()),

                            ),

                            ft.Container(

                                content=ft.Row(

                                    alignment="space_between",

                                    controls=[

                                        ft.Row(

                                            controls=[

                                                ft.Icon(ft.Icons.ACCESS_TIME_ROUNDED, size=24),

                                                ft.Text("Formato 24 horas", size=16),

                                            ],

                                            spacing=15

                                        ),

                                        ft.Switch(

                                            value=self.clock_24h,

                                            on_change=self._toggle_clock_format_switch,

                                            active_color=self._accent_ui(),

                                        ),

                                    ]

                                ),

                                padding=15,

                                border_radius=10,

                                bgcolor=ft.Colors.with_opacity(0.05, self._accent_ui()),

                            ),

                            ft.Container(

                                content=ft.Row(

                                    alignment="space_between",

                                    controls=[

                                        ft.Row(

                                            controls=[

                                                ft.Icon(ft.Icons.PALETTE_ROUNDED, size=24),

                                                ft.Text("Color de acento", size=16),

                                            ],

                                            spacing=15

                                        ),

                                        ft.Container(height=10),

                                        # Paleta de gradiente clicable (mejor UX que escribir #RRGGBB)
                                        self._build_accent_palette(),

                                    ]

                                ),

                                padding=15,

                                border_radius=10,

                                bgcolor=ft.Colors.with_opacity(0.05, self._accent_ui()),

                            ),

                            ft.Container(

                                content=ft.Row(

                                    alignment="space_between",

                                    controls=[

                                        ft.Row(

                                            controls=[

                                                ft.Icon(ft.Icons.STORAGE_ROUNDED, size=24),

                                                ft.Text("Almacenamiento", size=16),

                                            ],

                                            spacing=15

                                        ),

                                        ft.Container(height=10),

                                        ft.ElevatedButton(

                                            "Limpiar Datos",

                                            icon=ft.Icons.DELETE_OUTLINE_ROUNDED,

                                            on_click=self._clear_storage,

                                            style=ft.ButtonStyle(

                                                bgcolor="#e74c3c",

                                                color="white",

                                                shape=ft.RoundedRectangleBorder(radius=8),

                                            ),

                                        ),

                                    ]

                                ),

                                padding=15,

                                border_radius=10,

                                bgcolor=ft.Colors.with_opacity(0.05, self._accent_ui()),

                            ),

                        ]

                    ),

                    padding=20,

                )

            ],

            expand=True

        )



    def _change_accent_color(self, e):

        try:
            value = getattr(e.control, "value", None)
            if value is not None:
                self._set_accent(value)
                self._log(f"Color de acento cambiado a: {self.accent}")
        except Exception:
            pass



    def _clear_storage(self, e):

        try:

            self.page.client_storage.clear()

            self._log("Datos de almacenamiento limpiados")

            self.page.snack_bar = ft.SnackBar(

                content=ft.Text("Configuración restablecida correctamente"),

                action="OK",

            )

            self.page.snack_bar.open = True

            self.page.update()

        except Exception as ex:

            self._log(f"Error al limpiar almacenamiento: {ex}")



    def _toggle_theme_switch(self, e):

        self.is_dark_mode = e.control.value

        self.page.client_storage.set("is_dark_mode", self.is_dark_mode)

        self._apply_theme()

        self._update_theme_for_all_components()

        self._log(f"Tema cambiado a: {'oscuro' if self.is_dark_mode else 'claro'}")



    def _toggle_clock_format_switch(self, e):

        self.clock_24h = e.control.value

        self.page.client_storage.set("clock_24h", self.clock_24h)

        self.clock_text.value = self._get_current_time()

        self.clock_text.update()

        self._log(f"Formato de hora cambiado a: {'24h' if self.clock_24h else '12h'}")



    def _update_theme_for_all_components(self):

        # Actualizar botones del menú

        for button in self.menu_buttons.values():
            try:
                button.accent = self.accent
            except Exception:
                pass
            button.update_theme(self.is_dark_mode)
            try:
                if getattr(button, "is_active", False):
                    button.set_active(True, safe=True)
            except Exception:
                pass

        

        # Update background colors

        self.menu.bgcolor = "#16213e" if self.is_dark_mode else "#ffffff"

        self.control_panel.bgcolor = "#16213e" if self.is_dark_mode else "#ffffff"

        self.main_content_area.bgcolor = ft.Colors.with_opacity(0.03, "white" if self.is_dark_mode else "black")

        

        # Update header/menu accent elements if present
        try:
            if hasattr(self, "menu_logo_icon") and self.menu_logo_icon is not None:
                self.menu_logo_icon.color = self._accent_ui()
                self.menu_logo_icon.update()
        except Exception:
            pass
        try:
            if hasattr(self, "menu_logo_text") and self.menu_logo_text is not None:
                self.menu_logo_text.color = self._accent_ui()
                self.menu_logo_text.update()
        except Exception:
            pass

        # Update quick panel accent elements
        try:
            if hasattr(self, "btn_upload") and self.btn_upload is not None:
                self.btn_upload.style = ft.ButtonStyle(bgcolor=self._accent_ui(), color="white")
                self.btn_upload.update()
        except Exception:
            pass
        try:
            if hasattr(self, "clock_card") and self.clock_card is not None:
                self.clock_card.bgcolor = ft.Colors.with_opacity(0.1, self._accent_ui())
                self.clock_card.update()
        except Exception:
            pass

        # Rebuild view if needed

        current_view = self.last_view

        self._select_menu(current_view, force_rebuild=True)



    def _on_menu_click(self, e):

        view_key = e.control.data

        self._select_menu(view_key)



    def _select_menu(self, view_key, force_rebuild=False):

        if view_key == self.last_view and not force_rebuild:

            return



        # Desactivar todos los botones primero

        for key, button in self.menu_buttons.items():

            button.set_active(key == view_key)



        self.last_view = view_key



        # Construir la vista correspondiente

        if view_key == "welcome":

            new_view = self._build_welcome_view()

        elif view_key == "files":

            new_view = self._build_files_view()

        elif view_key == "analysis":

            new_view = self._build_analysis_view()

        elif view_key == "reports":

            new_view = self._build_reports_view()

        elif view_key == "settings":

            new_view = self._build_settings_view()

        elif view_key == "bearings":

            new_view = self._build_bearings_view()

        else:

            new_view = self._build_welcome_view()



        self.main_content_area.content = new_view

        if self.main_content_area.page:  # Only update if added to page

            self.main_content_area.update()



        # Actualizar ayuda contextual después de que la vista principal esté lista

        self._update_contextual_help(view_key)



    def _update_contextual_help(self, view_key):

        help_content = {

            "welcome": "Bienvenido al sistema de análisis de vibraciones. Desde aquí puede comenzar un nuevo análisis o ver la documentación.",

            "files": "Gestione sus archivos de datos. Puede cargar múltiples archivos CSV y seleccionarlos para análisis.",

            "analysis": "Realice análisis FFT y diagnóstico de vibraciones. Configure los parámetros y genere gráficos interactivos.",

            "reports": "Genere reportes detallados de sus análisis (próximamente).",

            "settings": "Configure las preferencias de la aplicación, incluyendo tema colores y formato de hora."

        }

        
        help_content["bearings"] = "Gestione rodamientos: seleccione un modelo, edite geometría y úselo en el análisis."

        self.help_panel.controls = [

            ft.Container(

                content=ft.Text(f"📋 Ayuda - {view_key.capitalize()}", size=16, weight="bold"),

                padding=ft.padding.only(bottom=10)

            ),

            ft.Text(help_content.get(view_key, "Información no disponible"), size=13)

        ]

        if self.help_panel.page:  # Only update if added to page

            self.help_panel.update()



    def _on_tab_change(self, e):

        tab_index = self.tabs.selected_index

        if tab_index == 0:

            self.tab_content.content = self.quick_actions

        elif tab_index == 1:

            self.tab_content.content = self.help_panel

        elif tab_index == 2:

            self.tab_content.content = self.log_panel

        if self.tab_content.page:  # Only update if added to page

            self.tab_content.update()



    def _pick_files(self, e):

        self.file_picker.pick_files(

            allow_multiple=True,

            allowed_extensions=["csv", "txt", "xlsx"],

            file_type=ft.FilePickerFileType.CUSTOM

        )



    def _handle_file_pick_result(self, e: ft.FilePickerResultEvent):
        """
        Manejador simplificado que solo obtiene la ruta del archivo y
        delega el proceso de carga y análisis a la función unificada.
        """
        if not e.files or len(e.files) == 0:
            self._log("No se seleccionó ningún archivo.")
            return

        # Solo toma el primer archivo seleccionado y llama a la función de carga principal
        file_path = e.files[0].path
        try:
            self._log(f"Archivo seleccionado: {os.path.basename(file_path)}. Cargando...")
            # Llama a la única función encargada de cargar y preparar el análisis
            self._load_file_data(file_path)
        except Exception as ex:
            self._log(f"Error fatal al iniciar la carga del archivo: {ex}")



    def _refresh_files_list(self):

        self.files_list_view.controls.clear()

        try:
            data_dir = os.path.join(os.getcwd(), "data")
            os.makedirs(data_dir, exist_ok=True)
            files = []
            for fn in os.listdir(data_dir):
                if fn.lower().endswith((".csv", ".txt", ".xlsx")):
                    p = os.path.join(data_dir, fn)
                    try:
                        mt = os.path.getmtime(p)
                    except Exception:
                        mt = 0.0
                    files.append((p, mt))
            # Filtro de búsqueda
            try:
                q = (getattr(self, 'data_search', None).value or '').strip().lower() if getattr(self, 'data_search', None) else ''
            except Exception:
                q = ''
            if q:
                files = [(p, mt) for (p, mt) in files if q in os.path.basename(p).lower()]
            from datetime import datetime as _dt
            groups = {}
            for p, mt in files:
                base = os.path.basename(p)
                try:
                    date_key = _dt.strptime(base[:10], "%Y-%m-%d").strftime("%Y-%m-%d")
                except Exception:
                    date_key = _dt.fromtimestamp(mt).strftime("%Y-%m-%d")
                groups.setdefault(date_key, []).append((p, mt))
            for date_key in sorted(groups.keys(), reverse=True):
                items = sorted(groups[date_key], key=lambda x: x[1], reverse=True)
                # Filtrar por favoritos si está activo
                try:
                    fav_only = bool(getattr(self, 'data_favs_only_cb', None) and getattr(self.data_favs_only_cb, 'value', False))
                except Exception:
                    fav_only = False
                filtered = []
                for p, _mt in items:
                    file_name = os.path.basename(p)
                    # Estado de favorito
                    is_fav = False
                    try:
                        is_fav = bool((self.data_favorites or {}).get(p, False))
                    except Exception:
                        is_fav = False
                    if fav_only and not is_fav:
                        continue
                    filtered.append((p, is_fav))
                if not filtered:
                    continue
                # Agregar cabecera de fecha solo si hay elementos
                self.files_list_view.controls.append(ft.Text(date_key, weight="bold"))
                for p, is_fav in filtered:
                    file_name = os.path.basename(p)
                    star_icon = ft.Icons.STAR if is_fav else ft.Icons.STAR_BORDER_ROUNDED
                    star_color = "#f1c40f" if is_fav else "#bdc3c7"
                    file_card = ft.Container(
                        content=ft.Row(
                            controls=[
                                ft.IconButton(icon=star_icon, icon_color=star_color, tooltip="Favorito", on_click=lambda e, path=p: self._toggle_data_favorite(path)),
                                ft.Icon(ft.Icons.DESCRIPTION_ROUNDED, size=24),
                                ft.Column(
                                    controls=[
                                        ft.Text(file_name, weight="bold", size=14),
                                        ft.Text(p, size=12, color="#7f8c8d"),
                                    ],
                                    expand=True,
                                ),
                                ft.IconButton(
                                    icon=ft.Icons.DELETE_OUTLINE_ROUNDED,
                                    tooltip="Eliminar archivo",
                                    on_click=lambda e, path=p: self._remove_file(path),
                                ),
                                ft.IconButton(
                                    icon=ft.Icons.VISIBILITY_ROUNDED,
                                    tooltip="Seleccionar para análisis",
                                    on_click=lambda e, path=p: self._load_file_data(path),
                                ),
                            ],
                            alignment="space_between",
                        ),
                        padding=15,
                        border_radius=10,
                        bgcolor=ft.Colors.with_opacity(0.05, self._accent_ui()),
                        on_hover=lambda e: self._on_file_hover(e),
                    )
                    self.files_list_view.controls.append(file_card)
        except Exception:
            self.files_list_view.controls.append(ft.Text("No se pudieron listar datos persistidos.", size=14))

        if self.files_list_view.page:  # Only update if added to page
            self.files_list_view.update()

            file_name = file_path.split("/")[-1] if "/" in file_path else file_path.split("\\")[-1]

            

            file_card = ft.Container(

                content=ft.Row(

                    controls=[

                        ft.Icon(ft.Icons.DESCRIPTION_ROUNDED, size=24),

                        ft.Column(

                            controls=[

                                ft.Text(file_name, weight="bold", size=14),

                                ft.Text(file_path, size=12, color="#7f8c8d"),

                            ],

                            expand=True,

                        ),

                        ft.IconButton(

                            icon=ft.Icons.DELETE_OUTLINE_ROUNDED,

                            tooltip="Eliminar archivo",

                            on_click=lambda e, path=file_path: self._remove_file(path),

                        ),

                        ft.IconButton(

                            icon=ft.Icons.VISIBILITY_ROUNDED,

                            tooltip="Seleccionar para análisis",

                            on_click=lambda e, path=file_path: self._load_file_data(path),

                        ),

                    ],

                    alignment="space_between",

                ),

                padding=15,

                border_radius=10,

                bgcolor=ft.Colors.with_opacity(0.05, self._accent_ui()),

                on_hover=lambda e: self._on_file_hover(e),

            )

            self.files_list_view.controls.append(file_card)

        

        if self.files_list_view.page:  # Only update if added to page

            self.files_list_view.update()



    def _on_file_hover(self, e):

        if e.data == "true":

            e.control.bgcolor = ft.Colors.with_opacity(0.1, self._accent_ui())

        else:

            e.control.bgcolor = ft.Colors.with_opacity(0.05, self._accent_ui())

        e.control.update()



    def _remove_file(self, file_path):

        if file_path in self.uploaded_files:

            self.uploaded_files.remove(file_path)

            self._log(f"Archivo eliminado: {file_path}")

            

            # Si el archivo eliminado era el actual, limpiar current_df

            if self.current_df is not None and file_path in self.file_data_storage:

                del self.file_data_storage[file_path]

                if not self.uploaded_files:

                    self.current_df = None

                else:

                    self._load_file_data(self.uploaded_files[0])

        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception:
            pass

        self._refresh_files_list()

    # ==== Favoritos de datos ====
    def _data_favorites_path(self) -> str:
        try:
            data_dir = os.path.join(os.getcwd(), "data")
            os.makedirs(data_dir, exist_ok=True)
            return os.path.join(data_dir, "data_favorites.json")
        except Exception:
            return "data_favorites.json"

    def _load_data_favorites(self) -> Dict[str, bool]:
        try:
            import json
            path = self._data_favorites_path()
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return {str(k): bool(v) for k, v in data.items()}
        except Exception:
            pass
        return {}

    def _save_data_favorites(self):
        try:
            import json
            path = self._data_favorites_path()
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.data_favorites or {}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _toggle_data_favorite(self, path: str):
        try:
            cur = bool((self.data_favorites or {}).get(path, False))
            self.data_favorites[path] = not cur
            self._save_data_favorites()
        except Exception:
            pass
        self._refresh_files_list()

    def _toggle_data_favs_filter(self):
        try:
            self.data_show_favs_only = bool(getattr(self, 'data_favs_only_cb', None) and getattr(self.data_favs_only_cb, 'value', False))
        except Exception:
            self.data_show_favs_only = False
        try:
            self.page.client_storage.set("data_favs_only", self.data_show_favs_only)
        except Exception:
            pass
        self._refresh_files_list()



    def _gather_calibration_settings(self) -> Dict[str, Dict[str, float]]:
        """Lee los campos de calibración disponibles y devuelve un mapa simple por canal."""
        calibs: Dict[str, Dict[str, float]] = {}
        try:
            sens_ctrl = getattr(self, "sens_unit_dd", None)
            sens_value_ctrl = getattr(self, "sensor_sens_field", None)
            if sens_ctrl is None or sens_value_ctrl is None:
                return calibs
            sens_unit = getattr(sens_ctrl, "value", None)
            raw_value = getattr(sens_value_ctrl, "value", None)
            if not sens_unit or raw_value in (None, ""):
                return calibs
            sens_val = float(raw_value)
            if sens_unit == "mV/g":
                calibs["__default__"] = {"mv_per_g": sens_val}
            elif sens_unit == "V/g":
                calibs["__default__"] = {"mv_per_g": sens_val * 1000.0}
        except Exception:
            return {}
        return calibs

    def _standardize_dataset(
        self,
        file_path: str,
        df: Optional[pd.DataFrame] = None,
        calibrations: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Normaliza un dataset a aceleración [m/s²] con tiempo en 0 y muestreo uniforme.

        Devuelve un DataFrame con columnas: t_s, accX_ms2, accY_ms2, accZ_ms2 (cuando existan).
        """
        data_frame = None
        if df is not None:
            data_frame = df.copy()
        else:
            try:
                data_frame = pd.read_csv(file_path)
            except Exception:
                data_frame = pd.read_csv(file_path, sep=';')
        if data_frame is None or data_frame.empty:
            return None

        # Detectar columna de tiempo
        time_col = None
        for col in data_frame.columns:
            col_str = str(col).lower()
            if any(key in col_str for key in ["time", "t_s", "tiempo", "timestamp", "sample"]):
                time_col = col
                break
        if time_col is None:
            data_frame = data_frame.copy()
            data_frame["time"] = np.arange(len(data_frame), dtype=float) / 10000.0
            time_col = "time"

        t_raw = pd.to_numeric(data_frame[time_col], errors="coerce").to_numpy(dtype=float)
        scale_to_seconds = 1.0
        finite_raw = t_raw[np.isfinite(t_raw)]
        span_raw = 0.0
        max_abs_raw = 0.0
        median_dt: Optional[float] = None
        if finite_raw.size:
            finite_sorted = np.sort(finite_raw)
            diffs_raw = np.diff(finite_sorted)
            diffs_raw = diffs_raw[np.isfinite(diffs_raw)]
            if diffs_raw.size:
                median_candidate = float(np.nanmedian(np.abs(diffs_raw)))
                if np.isfinite(median_candidate) and median_candidate > 0:
                    median_dt = median_candidate
            span_raw = float(finite_sorted[-1] - finite_sorted[0]) if finite_sorted.size > 1 else 0.0
            max_abs_raw = float(np.nanmax(np.abs(finite_raw)))

        def _infer_scale(dt: Optional[float], span: float, max_abs: float) -> float:
            if dt is not None:
                if dt >= 5e4:
                    return 1e9  # ns
                if dt >= 5e1:
                    return 1e6  # µs
                if dt >= 5e-1:
                    return 1e3  # ms
                return 1.0
            if span > 0 or max_abs > 0:
                if span >= 5e11 or max_abs >= 5e11:
                    return 1e9
                if span >= 5e8 or max_abs >= 5e8:
                    return 1e6
                if span >= 5e5 or max_abs >= 5e5:
                    return 1e3
            return 1.0

        scale_to_seconds = _infer_scale(median_dt, span_raw, max_abs_raw)

        if scale_to_seconds != 1.0:
            t = t_raw / scale_to_seconds
        else:
            t = t_raw

        finite_t = t[np.isfinite(t)]
        if finite_t.size:
            t0 = float(np.min(finite_t))
            t = t - t0
        else:
            t = np.arange(len(t_raw), dtype=float) / 10000.0

        diffs = np.diff(t[np.isfinite(t)])
        dt = float(np.nanmean(diffs)) if diffs.size else 0.0
        if not (np.isfinite(dt) and dt > 0):
            dt = 1.0 / 10000.0
        t_uniform = np.arange(0, dt * len(t), dt, dtype=float)[: len(t)]

        # Detección de columnas X/Y/Z
        def _is_x(name: str) -> bool:
            return bool(re.search(r"(acc|acel|x|ch1)", name.lower()))

        def _is_y(name: str) -> bool:
            return bool(re.search(r"(acc|acel|y|ch2)", name.lower()))

        def _is_z(name: str) -> bool:
            return bool(re.search(r"(acc|acel|z|ch3)", name.lower()))

        numeric_cols = [
            col
            for col in data_frame.columns
            if col != time_col and pd.api.types.is_numeric_dtype(data_frame[col])
        ]
        col_order = {col: idx for idx, col in enumerate(numeric_cols)}

        def _axis_score(col_name: Any, axis_letter: Optional[str]) -> float:
            name = str(col_name)
            lower = name.lower()
            score = 0.0
            if axis_letter:
                if axis_letter in lower:
                    score += 10.0
                else:
                    other_letters = {"x", "y", "z"} - {axis_letter}
                    if any(letter in lower for letter in other_letters):
                        score -= 4.0
                if axis_letter == "x" and any(token in lower for token in ["ch1", "canal1", "canal 1"]):
                    score += 4.0
                if axis_letter == "y" and any(token in lower for token in ["ch2", "canal2", "canal 2"]):
                    score += 4.0
                if axis_letter == "z" and any(token in lower for token in ["ch3", "canal3", "canal 3"]):
                    score += 4.0
            if re.search(r"(acc|acel)", lower):
                score += 6.0
            if re.search(r"m[/ _]?s\^?2", lower) or "ms2" in lower:
                score += 8.0
            if "_m_s2" in lower or "_ms2" in lower:
                score += 2.0
            if any(token in lower for token in ["vel", "velocity", "rpm"]):
                score -= 6.0
            if any(token in lower for token in ["disp", "despl", "position"]):
                score -= 6.0
            if any(token in lower for token in ["count", "adc", "raw"]):
                score -= 2.0
            score -= 0.001 * float(col_order.get(col_name, 0))
            return score

        remaining_cols: List[Any] = list(numeric_cols)

        def _select_axis(axis_letter: Optional[str]) -> Optional[Any]:
            if not remaining_cols:
                return None
            best_col = None
            best_score = float("-inf")
            for candidate in remaining_cols:
                score = _axis_score(candidate, axis_letter)
                if score > best_score:
                    best_score = score
                    best_col = candidate
            if best_col is not None:
                remaining_cols.remove(best_col)
            return best_col

        col_x = _select_axis("x")
        col_y = _select_axis("y")
        col_z = _select_axis("z")

        if col_x is None:
            col_x = _select_axis(None)
        if col_y is None:
            col_y = _select_axis(None)
        if col_z is None:
            col_z = _select_axis(None)

        def _interp_to_uniform(values: pd.Series) -> np.ndarray:
            y = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
            finite_mask = np.isfinite(t) & np.isfinite(y)
            if np.count_nonzero(finite_mask) < 2:
                return np.zeros_like(t_uniform)
            base_t = t[finite_mask]
            base_y = y[finite_mask]
            order = np.argsort(base_t)
            base_t = base_t[order]
            base_y = base_y[order]
            unique_t, unique_idx = np.unique(base_t, return_index=True)
            base_y = base_y[unique_idx]
            unique_t = unique_t.astype(float)
            return np.interp(t_uniform, unique_t, base_y, left=base_y[0], right=base_y[-1])

        def _get_calibration(name: Optional[str]) -> Optional[Dict[str, float]]:
            if not calibrations:
                return None
            if name is None:
                return calibrations.get("__default__")
            name_str = str(name)
            return (
                calibrations.get(name_str)
                or calibrations.get(name_str.lower())
                or calibrations.get("__default__")
            )

        def _to_ms2(series: Optional[pd.Series], header_name: Optional[str]) -> Optional[np.ndarray]:
            if series is None:
                return None

            series_dtype = getattr(series, "dtype", None)
            dtype_is_int = pd.api.types.is_integer_dtype(series_dtype)
            raw_values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
            y = _interp_to_uniform(series)
            h = str(header_name or "").lower()
            calib = _get_calibration(header_name)

            def deriv(sig: np.ndarray) -> np.ndarray:
                if dt <= 0:
                    return np.zeros_like(sig)
                return np.gradient(sig, dt, edge_order=2 if sig.size > 2 else 1)

            def second_deriv(sig: np.ndarray) -> np.ndarray:
                if dt <= 0:
                    return np.zeros_like(sig)
                first = deriv(sig)
                return deriv(first)

            def convert_counts(sig: np.ndarray) -> np.ndarray:
                if calib:
                    if "counts_per_g" in calib:
                        g_val = sig / float(calib["counts_per_g"])
                        return g_val * 9.80665
                    if "g_per_count" in calib:
                        g_val = sig * float(calib["g_per_count"])
                        return g_val * 9.80665
                    if "full_scale_g" in calib and "full_scale_counts" in calib:
                        ratio = float(calib["full_scale_g"]) / float(calib["full_scale_counts"])
                        g_val = sig * ratio
                        return g_val * 9.80665
                    if "mv_per_g" in calib:
                        # Conteos calibrados mediante sensibilidad en mV/g
                        mv = sig * 1000.0 if "v" in h and "mv" not in h else sig
                        g_val = mv / float(calib["mv_per_g"])
                        return g_val * 9.80665
                # Heurística estándar: ±2 g en 16 bits
                return sig * (2.0 / 32768.0) * 9.80665

            finite_raw = raw_values[np.isfinite(raw_values)]
            counts_like = False
            if finite_raw.size:
                max_abs_raw = float(np.nanmax(np.abs(finite_raw)))
                frac = np.abs(finite_raw - np.round(finite_raw))
                max_frac = float(np.nanmax(frac)) if frac.size else 0.0
                has_counts_calib = bool(
                    calib
                    and (
                        "counts_per_g" in calib
                        or "g_per_count" in calib
                        or "full_scale_g" in calib
                        or "full_scale_counts" in calib
                    )
                )
                if (
                    max_abs_raw > 50.0
                    and max_abs_raw <= 40000.0
                    and max_frac < 1e-6
                    and (dtype_is_int or max_abs_raw >= 512.0 or has_counts_calib)
                ):
                    counts_like = True

            if re.search(r"m[/ _]?s\^?2", h):
                return y

            if any(key in h for key in ["acc", "acel"]):
                if any(key in h for key in ["mm/s2", "mmps2", "mm s^2", "mm_s2"]):
                    return y / 1000.0
                if "mg" in h:
                    return y * 9.80665e-3
                if "gal" in h or "cm/s2" in h:
                    return y * 0.01
                if re.search(r"[^m]g\b", h) or h.endswith("_g") or " acc_g" in h:
                    return y * 9.80665
                if any(key in h for key in ["count", "adc", "volt", "mv", " v"]):
                    if calib:
                        if "mv_per_g" in calib:
                            y_mv = y * 1000.0 if "v" in h and "mv" not in h else y
                            g_val = y_mv / float(calib["mv_per_g"])
                            return g_val * 9.80665
                        if "counts_per_g" in calib or "g_per_count" in calib or "full_scale_g" in calib:
                            return convert_counts(y)
                    return y
                if counts_like:
                    return convert_counts(y)
                if calib and ("counts_per_g" in calib or "g_per_count" in calib or "full_scale_g" in calib):
                    return convert_counts(y)
                return y

            if counts_like:
                return convert_counts(y)

            if any(key in h for key in ["vel", "velocity"]):
                vel = y
                if any(key in h for key in ["mm/s", "mmps"]) and not any(key in h for key in ["m/s2", "mm/s2"]):
                    vel = vel / 1000.0
                return deriv(vel)

            if any(key in h for key in ["disp", "despl", "displacement"]):
                disp = y
                if "µm" in h or "um" in h:
                    disp = disp * 1e-6
                elif re.search(r"\bmm\b", h) and not any(key in h for key in ["mm/s", "mm/s2"]):
                    disp = disp * 1e-3
                return second_deriv(disp)

            if counts_like:
                return convert_counts(y)
            return y

        acc_x = _to_ms2(data_frame[col_x], col_x) if col_x is not None else None
        acc_y = _to_ms2(data_frame[col_y], col_y) if col_y is not None else None
        acc_z = _to_ms2(data_frame[col_z], col_z) if col_z is not None else None

        data: Dict[str, np.ndarray] = {"t_s": t_uniform.astype(float)}
        if acc_x is not None:
            data["accX_ms2"] = np.asarray(acc_x, dtype=float)
        if acc_y is not None:
            data["accY_ms2"] = np.asarray(acc_y, dtype=float)
        if acc_z is not None:
            data["accZ_ms2"] = np.asarray(acc_z, dtype=float)

        if len(data) <= 1:
            return None

        return pd.DataFrame(data)
    
    def reset_analysis_state(self):
        """Reinicia todas las estructuras internas del análisis para evitar acumulaciones."""
        self._reset_runtime_analysis_state(
            clear_dataset=True,
            clear_visuals=True,
            announce=False,
            clear_file_storage=True,
        )
        print("🧽 Estado de análisis completamente reiniciado")
        try:
            self._log("Estado de análisis completamente reiniciado")
        except Exception:
            pass


    def _reset_runtime_analysis_state(
        self,
        *,
        clear_dataset: bool = False,
        clear_visuals: bool = False,
        announce: bool = True,
        clear_file_storage: bool = False,
    ) -> None:
        """Limpia el estado de resultados y espectros antes de un nuevo análisis."""

        self._last_axis_severity = []
        self._last_primary_severity = None
        if hasattr(self, "_last_combined_sources"):
            self._last_combined_sources = []
        self._last_xf = None
        self._last_spec = None
        self._last_tseg = None
        self._last_accseg = None

        if clear_dataset:
            self.current_df = None
            try:
                self._raw_current_df = None
            except Exception:
                pass
            self.signal_unit_map.clear()

            if clear_file_storage and hasattr(self, "file_data_storage") and isinstance(self.file_data_storage, dict):
                self.file_data_storage.clear()

        if clear_dataset or clear_visuals:
            self._fft_zoom_range = None
            self._fft_full_range = None
            self._fft_zoom_syncing = False
            self._fft_display_scale = 1.0
            self._fft_display_unit = "Hz"

        if announce:
            msg = "Estado limpio: reset de resultados anteriores antes del nuevo análisis"
            print(msg)
            try:
                self._log(msg)
            except Exception:
                pass

    def _reset_analysis_state_on_new_file(self):
        """Limpia resultados y caches para evitar arrastre entre archivos."""

        self._reset_runtime_analysis_state(
            clear_dataset=True,
            clear_visuals=True,
            announce=True,
            clear_file_storage=False,
        )

        # Restablecer combinaciones vectoriales
        try:
            self.combine_signals_enabled = False
            if hasattr(self, "combine_signals_cb") and self.combine_signals_cb is not None:
                self.combine_signals_cb.value = False
                if getattr(self.combine_signals_cb, "page", None):
                    self.combine_signals_cb.update()
        except Exception:
            pass
        try:
            for cb in getattr(self, "signal_checkboxes", []) or []:
                cb.value = False
                if getattr(cb, "page", None):
                    cb.update()
        except Exception:
            pass

        # Limpiar contenedores visuales principales
        try:
            if hasattr(self, "chart_container") and self.chart_container is not None:
                self.chart_container.content = ft.Column(
                    [
                        ft.Icon(ft.Icons.INSIGHTS, color=self._accent_ui()),
                        ft.Text("Carga un archivo para generar nuevas gráficas", size=14),
                    ],
                    horizontal_alignment="center",
                    alignment="center",
                    spacing=8,
                )
                if getattr(self.chart_container, "page", None):
                    self.chart_container.update()
        except Exception:
            pass
        try:
            if hasattr(self, "multi_chart_container") and self.multi_chart_container is not None:
                self.multi_chart_container.content = ft.Text(
                    "Selecciona canales para visualizar después de cargar un archivo."
                )
                if getattr(self.multi_chart_container, "page", None):
                    self.multi_chart_container.update()
        except Exception:
            pass


    def _load_file_data(self, file_path):

        df: Optional[pd.DataFrame] = None
        df_std: Optional[pd.DataFrame] = None
        normalized = False
        calibrations: Dict[str, Dict[str, float]] = {}

        try:
            cached_df: Optional[pd.DataFrame] = None
            if hasattr(self, "file_data_storage") and isinstance(self.file_data_storage, dict):
                cached_df = self.file_data_storage.get(file_path)
        except Exception:
            cached_df = None

        try:
            if isinstance(cached_df, pd.DataFrame) and not cached_df.empty:
                # Reutilizar dataset normalizado previamente almacenado
                self._reset_analysis_state_on_new_file()
                df = cached_df.copy(deep=True)
                df_std = df.copy(deep=True)
                normalized = True
            else:
                # Carga completa desde disco
                self._reset_analysis_state_on_new_file()

                calibrations = self._gather_calibration_settings()
                if calibrations:
                    self._log("Calibraciones ingresadas en la UI ignoradas para preservar RMS originales.")
                    calibrations = {}
                try:
                    print(f"[DEBUG] Calibraciones activas: {calibrations}")
                except Exception:
                    pass

                data_dir = os.path.join(os.getcwd(), "data")
                os.makedirs(data_dir, exist_ok=True)
                abs_data_dir = os.path.abspath(data_dir)
                storage_path = os.path.abspath(file_path)
                try:
                    if not storage_path.startswith(abs_data_dir):
                        base_name = os.path.basename(file_path)
                        target_path = os.path.join(data_dir, base_name)
                        root_name, ext = os.path.splitext(base_name)
                        counter = 1
                        while os.path.exists(target_path):
                            target_path = os.path.join(data_dir, f"{root_name}_{counter}{ext}")
                            counter += 1
                        shutil.copy2(file_path, target_path)
                        storage_path = os.path.abspath(target_path)
                        self._log(f"Archivo copiado al repositorio local: {os.path.basename(target_path)}")
                except Exception as copy_exc:
                    self._log(f"No se pudo guardar copia local: {copy_exc}")
                file_path = storage_path

                if file_path.endswith('.csv'):
                    df_raw = pd.read_csv(file_path)
                elif file_path.endswith('.xlsx'):
                    df_raw = pd.read_excel(file_path)
                else:
                    df_raw = pd.read_csv(file_path)

                df = df_raw
                df_std = self._standardize_dataset(file_path, df=df_raw, calibrations=calibrations)

            if df_std is not None and not df_std.empty:
                df = df_std
                self.signal_unit_map = {col: "acc_ms2" for col in df.columns if col != "t_s"}
                normalized = True
                try:
                    self.input_signal_unit = "acc_ms2"
                    if getattr(self, "input_signal_unit_dd", None):
                        if getattr(self.input_signal_unit_dd, "value", None) != "acc_ms2":
                            self.input_signal_unit_dd.value = "acc_ms2"
                        if getattr(self.input_signal_unit_dd, "page", None):
                            self.input_signal_unit_dd.update()
                except Exception:
                    pass
            else:
                self.signal_unit_map = {}

            # Auto-rescalar ejes si las velocidades resultan irreales (>500 mm/s)
            try:
                if df is not None and "t_s" in df.columns:
                    t_vals = pd.to_numeric(df["t_s"], errors="coerce").to_numpy(dtype=float)
                    if t_vals.size > 2:
                        axes = [col for col in df.columns if col != "t_s"]
                        rescaled_axes: List[str] = []
                        for col in axes:
                            series = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
                            if series.size < 2:
                                continue
                            try:
                                res_axis = analyze_vibration(t_vals, series)
                                rms_val = float(res_axis.get("severity", {}).get("rms_mm_s", 0.0))
                            except Exception:
                                rms_val = 0.0
                            if rms_val > 500.0:
                                df[col] = series / 1000.0
                                rescaled_axes.append(col)
                        if rescaled_axes:
                            for col in rescaled_axes:
                                self.signal_unit_map[col] = "acc_ms2"
                            self._log(
                                "RMS excesivo detectado; señal reescalada ÷1000 en: "
                                + ", ".join(rescaled_axes)
                            )
                            for col in rescaled_axes:
                                try:
                                    series_post = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
                                    rescaled_res = analyze_vibration(t_vals, series_post)
                                    self._log(
                                        f"RMS ajustado {col}: {float(rescaled_res.get('severity', {}).get('rms_mm_s', 0.0)):.3f} mm/s"
                                    )
                                except Exception:
                                    pass
            except Exception:
                pass

        except Exception as e:
            self._log(f"Error al leer archivo {file_path}: {str(e)}")
            self.page.snack_bar = ft.SnackBar(
                content=ft.Text(f"Error al cargar archivo: {str(e)}"),
                bgcolor="#e74c3c",
            )
            self.page.snack_bar.open = True
            self.page.update()
            return

        if df is None:
            self._log(f"No se pudo cargar archivo: {file_path}")
            return
            
        # ... (el resto de la función sigue igual) ...
        # ... (Asignar self.current_df, self._raw_current_df, etc.) ...
        
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            self._log(f"Archivo inválido: {file_path} no tiene columnas numéricas")
            return

        try:
            if file_path in self.uploaded_files:
                self.uploaded_files.remove(file_path)
            self.uploaded_files.append(file_path)
        except Exception:
            self.uploaded_files = [file_path]

        self.file_data_storage[file_path] = df.copy(deep=True)
        self.current_df = df
        self._raw_current_df = df.copy(deep=True)

        status_msg = "Datos cargados y normalizados" if normalized else "Datos cargados"
        self._log(f"{status_msg}: {file_path} ({len(df)} filas)")

        try:
            self._refresh_files_list()
        except Exception:
            pass

        # Cambiar a la vista de análisis para que el usuario pueda trabajar
        self._select_menu("analysis", force_rebuild=True)


    def _set_pdf_progress(self, visible: bool, message: Optional[str] = None) -> None:
        """Muestra u oculta la superposición de progreso al exportar PDF."""
        try:
            if message is not None and hasattr(self, "pdf_progress_text"):
                self.pdf_progress_text.value = message
                if self.pdf_progress_text.page:
                    self.pdf_progress_text.update()
            if not hasattr(self, "pdf_progress_overlay"):
                return
            self.pdf_progress_overlay.visible = visible
            if self.pdf_progress_overlay.page:
                self.pdf_progress_overlay.update()
            else:
                self.page.update()
        except Exception:
            pass


    def _log(self, message):

        timestamp = datetime.now().strftime("%H:%M:%S" if self.clock_24h else "%I:%M:%S %p")

        log_entry = ft.Text(f"[{timestamp}] {message}", size=12)

        self.log_panel.controls.append(log_entry)

        if self.log_panel.page:  # Only update if added to page

            self.log_panel.update()

        print(f"[LOG] {message}")



    def _toggle_theme(self, e):

        self.is_dark_mode = not self.is_dark_mode

        self.page.client_storage.set("is_dark_mode", self.is_dark_mode)

        self._apply_theme()

        self._update_theme_for_all_components()

        self._log(f"Tema cambiado a: {'oscuro' if self.is_dark_mode else 'claro'}")



    def _toggle_clock_format(self, e):

        self.clock_24h = not self.clock_24h

        self.page.client_storage.set("clock_24h", self.clock_24h)

        self.clock_text.value = self._get_current_time()

        self.clock_text.update()

        self._log(f"Formato de hora cambiado a: {'24h' if self.clock_24h else '12h'}")



    async def _start_clock_timer(self):

        while True:

            self.clock_text.value = self._get_current_time()

            if self.clock_text.page:

                self.clock_text.update()

            await asyncio.sleep(1)



    def _toggle_panel(self, e):

        self.is_panel_expanded = not self.is_panel_expanded

        

        # Actualizar icono del botón

        e.control.icon = (

            ft.Icons.CHEVRON_LEFT_ROUNDED 

            if self.is_panel_expanded 

            else ft.Icons.CHEVRON_RIGHT_ROUNDED

        )

        

        # Actualizar el panel

        self.control_panel.width = 350 if self.is_panel_expanded else 65

        self.control_panel.padding = ft.padding.all(20 if self.is_panel_expanded else 10)

        

        # Actualizar visibilidad del contenido

        panel_header = self.control_panel.content.controls[0]

        panel_header.controls[0].visible = self.is_panel_expanded  # Título

        

        # Actualizar resto del contenido

        content_container = self.control_panel.content.controls[-1]

        content_container.visible = self.is_panel_expanded

        

        # Actualizar el panel

        self.control_panel.update()



    def _toggle_config_panel(self, e):

        self.config_expanded = not self.config_expanded

        self.config_container.visible = self.config_expanded

        e.control.icon = (

            ft.Icons.ARROW_DROP_DOWN_CIRCLE if self.config_expanded else ft.Icons.ARROW_RIGHT

        )

        if self.page:

            self.page.update()


    def _on_time_color_change(self, e=None):

        selected = getattr(getattr(e, "control", None), "value", None) if e else None
        fallback = self.time_plot_color or "#00bcd4"
        self.time_plot_color = self._remember_color_pref("time_plot_color", selected, fallback)
        ctrl = getattr(self, "time_color_dd", None)
        if ctrl:
            ctrl.value = self.time_plot_color
            if ctrl.page:
                ctrl.update()
        self._update_analysis()

    def _on_input_signal_unit_change(self, e=None):
        unit = getattr(getattr(e, "control", None), "value", None) if e else None
        self.input_signal_unit = unit if unit else getattr(self, "input_signal_unit", "acc_ms2")
        try:
            self.page.client_storage.set("input_signal_unit", self.input_signal_unit)
        except Exception:
            pass
        self._update_analysis()
        try:
            self._update_multi_chart()
        except Exception:
            pass

    def _on_fft_color_change(self, e=None):

        selected = getattr(getattr(e, "control", None), "value", None) if e else None
        fallback = self.fft_plot_color or self._accent_ui()
        self.fft_plot_color = self._remember_color_pref("fft_plot_color", selected, fallback)
        ctrl = getattr(self, "fft_color_dd", None)
        if ctrl:
            ctrl.value = self.fft_plot_color
            if ctrl.page:
                ctrl.update()
        self._update_analysis()


    def _on_fft_window_change(self, e=None):

        try:
            selected = getattr(getattr(e, "control", None), "value", None) if e else None
        except Exception:
            selected = None
        resolved = _resolve_fft_window(selected if selected is not None else getattr(self, "fft_window_type", "hann"))
        self.fft_window_type = resolved
        try:
            self.page.client_storage.set("fft_window_type", resolved)
        except Exception:
            pass
        self._update_analysis()


    def _on_combine_signals_toggle(self, e=None):

        try:
            self.combine_signals_enabled = bool(getattr(self.combine_signals_cb, "value", False))
            self.page.client_storage.set("combine_signals_enabled", self.combine_signals_enabled)
        except Exception:
            pass
        self._update_analysis()
        try:
            self._update_multi_chart()
        except Exception:
            pass

    def _on_runup_3d_toggle(self, e=None):
        try:
            enabled = bool(getattr(self, 'runup_3d_cb', None).value)
        except Exception:
            enabled = False
        self.runup_3d_enabled = enabled
        try:
            self.page.client_storage.set("runup_3d_enabled", self.runup_3d_enabled)
        except Exception:
            pass
        self._update_analysis()


    def _refresh_orbit_inputs(self):
        ctrl_x = getattr(self, 'orbit_x_dd', None)
        ctrl_y = getattr(self, 'orbit_y_dd', None)
        ctrl_period = getattr(self, 'orbit_period_field', None)
        try:
            enabled = bool(getattr(self, 'orbit_cb', None).value)
        except Exception:
            enabled = bool(getattr(self, 'orbit_plot_enabled', False))
        try:
            options_available = bool(getattr(ctrl_x, 'options', []) or getattr(ctrl_y, 'options', []))
            if isinstance(getattr(ctrl_x, 'options', None), list):
                options_available = options_available and len(ctrl_x.options) > 0
            if isinstance(getattr(ctrl_y, 'options', None), list):
                options_available = options_available and len(ctrl_y.options) > 0
        except Exception:
            options_available = False
        disabled = (not enabled) or (not options_available)
        for ctrl in (ctrl_x, ctrl_y):
            if ctrl is None:
                continue
            try:
                ctrl.disabled = disabled
                if ctrl.page:
                    ctrl.update()
            except Exception:
                continue
        if ctrl_period is not None:
            try:
                ctrl_period.disabled = not enabled
                if ctrl_period.page:
                    ctrl_period.update()
            except Exception:
                pass

    def _on_orbit_toggle(self, e=None):
        try:
            enabled = bool(getattr(self, 'orbit_cb', None).value)
        except Exception:
            enabled = False
        self.orbit_plot_enabled = enabled
        try:
            self.page.client_storage.set("orbit_plot_enabled", self.orbit_plot_enabled)
        except Exception:
            pass
        self._refresh_orbit_inputs()
        self._update_analysis()

    def _on_orbit_axis_change(self, e=None):
        ctrl = getattr(e, 'control', None) if e else None
        value = getattr(ctrl, 'value', None) if ctrl else None
        if ctrl is getattr(self, 'orbit_x_dd', None):
            self._remember_orbit_axis('x', value if value else None)
        elif ctrl is getattr(self, 'orbit_y_dd', None):
            self._remember_orbit_axis('y', value if value else None)
        else:
            return
        if self.orbit_cb and not getattr(self.orbit_cb, 'value', False):
            return
        self._update_analysis()

    def _on_orbit_period_change(self, e=None):
        raw = ""
        try:
            raw = str(getattr(self.orbit_period_field, 'value', "")).strip()
        except Exception:
            raw = ""
        if raw:
            raw = raw.replace(",", ".")
        try:
            parsed = float(raw) if raw else None
        except Exception:
            parsed = None
        if parsed is not None and (not np.isfinite(parsed) or parsed <= 0):
            parsed = None
        self.orbit_period_seconds = parsed
        try:
            if parsed is None:
                self.page.client_storage.set("orbit_period_seconds", "")
            else:
                self.page.client_storage.set("orbit_period_seconds", parsed)
        except Exception:
            pass
        if self.orbit_cb and not getattr(self.orbit_cb, 'value', False):
            return
        self._update_analysis()



    def _update_multi_chart(self, e=None, normalize=True):

        """

        Genera gráfica combinada de FFTs seleccionadas.

        - normalize=True: escala cada señal entre 0–1 para ver todas.

        """

        try:

            time_col = self.time_dropdown.value

            t = self.current_df[time_col].to_numpy()

            selected_signals = [cb.label for cb in self.signal_checkboxes if cb.value]



            if not selected_signals:

                chart = ft.Text("⚠️ No hay señales seleccionadas")

            else:

                plt.style.use('dark_background' if self.is_dark_mode else 'seaborn-v0_8-whitegrid')

                fig, ax = plt.subplots(figsize=(12, 5))



                # Vista en dBV real opcional (aplica a todas las curvas)
                try:
                    use_dbv = bool(getattr(self, 'db_scale_cb', None) and getattr(self.db_scale_cb, 'value', False))
                except Exception:
                    use_dbv = False
                # Calibración para dBV
                try:
                    sens_unit = getattr(self, 'sens_unit_dd', None).value if getattr(self, 'sens_unit_dd', None) else 'mV/g'
                except Exception:
                    sens_unit = 'mV/g'
                try:
                    sens_val = float(getattr(self, 'sensor_sens_field', None).value) if getattr(self, 'sensor_sens_field', None) else 100.0
                except Exception:
                    sens_val = 100.0
                try:
                    gain_vv = float(getattr(self, 'gain_field', None).value) if getattr(self, 'gain_field', None) else 1.0
                except Exception:
                    gain_vv = 1.0

                # Filtros de frecuencia visuales
                try:
                    fmin_ui = float(self.lf_cutoff_field.value) if getattr(self, 'lf_cutoff_field', None) and getattr(self.lf_cutoff_field, 'value', '') else 0.0
                except Exception:
                    fmin_ui = 0.0
                try:
                    fmax_ui = float(self.hf_limit_field.value) if getattr(self, 'hf_limit_field', None) and getattr(self.hf_limit_field, 'value', '') else None
                except Exception:
                    fmax_ui = None

                freq_scale = getattr(self, "_fft_display_scale", 1.0) or 1.0
                if not np.isfinite(freq_scale) or freq_scale <= 0:
                    freq_scale = 1.0
                freq_unit = getattr(self, "_fft_display_unit", "Hz") or "Hz"

                for sig in selected_signals:

                    y_raw = self.current_df[sig].to_numpy()
                    t_sig, acc_sig, _, _ = self._prepare_segment_for_analysis(t, y_raw, sig)
                    if acc_sig.size < 2 or t_sig.size != acc_sig.size:
                        continue
                    try:
                        pre_dec = float(fmax_ui) if fmax_ui and fmax_ui > 0 else None
                    except Exception:
                        pre_dec = None
                    res_sig = analyze_vibration(
                        t_sig,
                        acc_sig,
                        pre_decimate_to_fmax_hz=pre_dec,
                        fft_window=self.fft_window_type,
                    )
                    xf = res_sig.get('fft', {}).get('f_hz')
                    mag_vel_mm = res_sig.get('fft', {}).get('vel_spec_mm_s')
                    mag_acc = res_sig.get('fft', {}).get('acc_spec_ms2')
                    if xf is None or mag_vel_mm is None:
                        continue
                    xf = np.asarray(xf, dtype=float)
                    mag_vel_mm = np.asarray(mag_vel_mm, dtype=float)
                    mag_acc = np.asarray(mag_acc if mag_acc is not None else np.zeros_like(mag_vel_mm), dtype=float)

                    mask = xf >= max(0.0, fmin_ui)
                    if fmax_ui and fmax_ui > 0:
                        mask = mask & (xf <= float(fmax_ui))

                    if use_dbv:
                        if sens_unit == 'mV/g':
                            sens_v_per_g = sens_val * 1e-3
                            V_amp = (mag_acc / 9.80665) * sens_v_per_g * gain_vv
                        elif sens_unit == 'V/g':
                            V_amp = (mag_acc / 9.80665) * sens_val * gain_vv
                        elif sens_unit == 'mV/(mm/s)':
                            V_amp = mag_vel_mm * (sens_val * 1e-3) * gain_vv
                        elif sens_unit == 'V/(mm/s)':
                            V_amp = mag_vel_mm * sens_val * gain_vv
                        else:
                            V_amp = np.zeros_like(mag_vel_mm)
                        eps = 1e-12
                        yplot = 20.0 * np.log10(np.maximum(np.asarray(V_amp, dtype=float), eps) / 1.0)
                        ax.plot((xf[mask] / freq_scale), yplot[mask], linewidth=2, label=sig)
                    else:
                        yplot = mag_vel_mm.copy()
                        if normalize and yplot.max() > 0:
                            yplot = yplot / yplot.max()
                        ax.plot((xf[mask] / freq_scale), yplot[mask], linewidth=2, label=sig)

                ax.set_title("FFT combinada de señales")

                ax.set_xlabel(f"Frecuencia ({freq_unit})")

                if use_dbv:
                    ax.set_ylabel("Nivel [dBV]")
                    # Rango Y en dBV si se definió
                    try:
                        ymin = float(self.db_ymin_field.value) if getattr(self, 'db_ymin_field', None) and getattr(self.db_ymin_field, 'value', '') != '' else None
                    except Exception:
                        ymin = None
                    try:
                        ymax = float(self.db_ymax_field.value) if getattr(self, 'db_ymax_field', None) and getattr(self.db_ymax_field, 'value', '') != '' else None
                    except Exception:
                        ymax = None
                    if ymin is not None or ymax is not None:
                        cur = ax.get_ylim()
                        ax.set_ylim(ymin if ymin is not None else cur[0], ymax if ymax is not None else cur[1])
                else:
                    ax.set_ylabel("Velocidad [mm/s]" if not normalize else "Amplitud normalizada")

                try:
                    if fmax_ui and fmax_ui > 0:
                        ax.set_xlim(left=0.0, right=float(fmax_ui) / freq_scale)
                    if fmin_ui and fmin_ui > 0:
                        cur = ax.get_xlim()
                        ax.set_xlim(left=float(fmin_ui), right=cur[1])
                except Exception:
                    pass
                ax.legend(ncol=2, fontsize=8)



                chart = MatplotlibChart(fig, expand=True, isolated=True)

                plt.close(fig)

            self.multi_chart_container.content = chart

            if self.multi_chart_container.page:

                self.multi_chart_container.update()

        except Exception as ex:

            self._log(f"Error en gráfica combinada: {ex}")

            self.multi_chart_container.content = ft.Text(f"Error en gráfica combinada: {ex}")

            if self.multi_chart_container.page:

                self.multi_chart_container.update()



# =========================

#   Apartado: Diagnóstico

# =========================

import flet as ft



def diagnostico_view(page: ft.Page, on_generate=None):

    opciones = [

        "Vibraciones generales", "FFT señal completa", "FFT por ventana", "Valores RMS",

        "Valor pico", "Valor pico-pico", "Espectro de frecuencias", "Velocidad crítica",

        "Resonancia", "Distorsión de carcasa", "Armónicos", "IPS (pulgadas/seg)",

        "Comparación de espectros", "Tendencias históricas", "Temperatura relacionada",

        "Desbalanceo", "Desalineación", "Rodamientos", "Excentricidad",

        "Falla catastrófica", "Engranes", "Holguras", "Fuerzas axiales",

        "Modos propios", "Filtros banda", "Cepstrum", "Envelope",

        "Top-N picos", "Order tracking", "Overspeed",

    ]



    chips = [ft.FilterChip(label=op, selected=False) for op in opciones]



    def set_all(val: bool):

        for c in chips:

            c.selected = val

            c.update()



    def generar(_):

        seleccionadas = [c.label for c in chips if c.selected]

        if on_generate:

            on_generate(seleccionadas)

        else:

            page.snack_bar = ft.SnackBar(ft.Text(f"Elegidas: {len(seleccionadas)}"))

            page.snack_bar.open = True

            page.update()



    # --- Dividir en columnas ---

    per_col = 8

    def chunk(lst, n):

        for i in range(0, len(lst), n):

            yield lst[i:i+n]



    columnas = []

    for grupo in chunk(chips, per_col):

        columnas.append(

            ft.Container(

                width=260,

                padding=10,

                content=ft.Column(

                    controls=grupo,

                    scroll="auto",

                )

            )

        )



    # --- Scroll horizontal de columnas ---

    opciones_scroller = ft.Container(

        height=320,

        content=ft.Row(

            controls=columnas,

            spacing=12,

            scroll="auto",

            vertical_alignment=ft.CrossAxisAlignment.START

        )

    )



    acciones = ft.Row(

        controls=[

            ft.TextButton("Seleccionar todo", on_click=lambda e: set_all(True)),

            ft.TextButton("Limpiar", on_click=lambda e: set_all(False)),

            ft.ElevatedButton(

                "Generar gráficas",

                icon=ft.Icons.ANALYTICS,

                on_click=generar,

            ),

        ],

        wrap=True,

        alignment=ft.MainAxisAlignment.END,

    )



    return ft.Column(

        controls=[

            ft.Text("Opciones de diagnóstico", size=20, weight="bold"),

            opciones_scroller,

            acciones,

        ],

        expand=True,

        spacing=12,

    )



# =========================

#   Punto de entrada

# =========================

def main(page: ft.Page):

    print("App iniciada")

    app = MainApp(page)

    page.add(app.content)



if __name__ == "__main__":

    ft.app(target=main)
