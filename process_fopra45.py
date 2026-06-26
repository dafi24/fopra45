#!/usr/bin/env python3
"""FoPra 45 data processing.

Execute from the repository root with

    python process_fopra45.py

The script creates analysis_output/ with calibration plots, processed spectra,
temperature traces, quantum-well energy/width plots, results.csv, and summary.txt.
Only numpy and matplotlib are required.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "CiaraDavidNiccolo"
OUT = ROOT / "analysis_output"
OUT.mkdir(exist_ok=True)

HC = 1239.50  # eV nm
EG_GAAS = 1.5192  # eV, GaAs gap at 4.2 K
VAR_A, VAR_B = 5.405e-4, 204.0
EPS_GAAS, RYDBERG_EV = 12.58, 13.605693
HBAR2_2M0 = 0.0380998212  # eV nm^2

# Corrected handwritten/lab-note setup values.
GRATING_LINES_MM = 150.0
FOCAL_MM = 200.0
SLIT_MM = 0.025
ORDER = 1

KNOWN_LINES = np.array([710.0, 730.0, 733.0, 755.0, 760.0, 845.0])
MEASURED_SEEDS = np.array([705.70, 723.66, 727.24, 747.97, 752.56, 836.87])
ALGAAS_WIN = (630.0, 690.0)
QW_WIN = (700.0, 815.0)
GAAS_WIN = (805.0, 835.0)
MQW_TEMP_WIN = (760.0, 840.0)


def read_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read two-column spectrometer CSV and sort by raw x-axis."""
    a = np.genfromtxt(path, delimiter=",", dtype=float)
    a = a[np.isfinite(a).all(axis=1)]
    order = np.argsort(a[:, 0])
    return a[order, 0], a[order, 1]


def exposure(path: Path) -> float:
    """Integration time: explicit in bottom/background names, otherwise 5 s."""
    name = path.name.lower()
    if "30sec" in name:
        return 30.0
    if "10sec" in name:
        return 10.0
    return 5.0


def smooth(y: np.ndarray, n: int = 7) -> np.ndarray:
    if n <= 1:
        return y.copy()
    return np.convolve(y, np.ones(n) / n, mode="same")


def despike(y: np.ndarray, n: int = 5) -> np.ndarray:
    """Median filter suppresses one-bin cosmic-ray spikes mentioned in notes."""
    h = n // 2
    p = np.pad(y, (h, h), mode="edge")
    return np.array([np.median(p[i:i+n]) for i in range(len(y))])


def vertex(x: np.ndarray, y: np.ndarray, i: int) -> float:
    """Parabolic peak center from three points."""
    if i <= 0 or i >= len(x) - 1:
        return float(x[i])
    a, b, _ = np.polyfit(x[i-1:i+2], y[i-1:i+2], 2)
    xv = -b / (2 * a) if a < 0 else x[i]
    return float(xv if x[i-1] <= xv <= x[i+1] else x[i])


def fwhm(x: np.ndarray, y: np.ndarray, i: int) -> float:
    """Line width above local baseline, used as measured spectral resolution."""
    base = np.percentile(y, 10)
    half = base + 0.5 * (y[i] - base)
    l = i
    while l > 0 and y[l] > half:
        l -= 1
    r = i
    while r < len(y) - 1 and y[r] > half:
        r += 1
    if l == i or r == i:
        return float("nan")
    xl = np.interp(half, [y[l], y[l+1]], [x[l], x[l+1]])
    xr = np.interp(half, [y[r], y[r-1]], [x[r], x[r-1]])
    return abs(float(xr - xl))


def peak(x: np.ndarray, y: np.ndarray, win: tuple[float, float]) -> tuple[float, float]:
    """Strongest peak center and FWHM in a wavelength window."""
    m = (x >= win[0]) & (x <= win[1])
    if m.sum() < 5:
        return float("nan"), float("nan")
    xx, yy = x[m], smooth(y[m])
    i = int(np.argmax(yy))
    return vertex(xx, yy, i), fwhm(xx, yy, i)


def peaks(x: np.ndarray, y: np.ndarray, win: tuple[float, float], n: int = 6, dmin: float = 3.0) -> list[tuple[float, float]]:
    """Well-separated local maxima: used for the QW luminescence lines."""
    m = (x >= win[0]) & (x <= win[1])
    xx, yy = x[m], smooth(y[m])
    if len(xx) < 5:
        return []
    noise = 1.4826 * np.median(np.abs(yy - np.median(yy)))
    thr = np.percentile(yy, 20) + max(3 * noise, 0.05 * (yy.max() - np.percentile(yy, 20)))
    cand = np.where((yy[1:-1] > yy[:-2]) & (yy[1:-1] >= yy[2:]) & (yy[1:-1] > thr))[0] + 1
    cand = cand[np.argsort(yy[cand])[::-1]]
    out: list[tuple[float, float]] = []
    for i in cand:
        xp = vertex(xx, yy, int(i))
        if all(abs(xp - p[0]) >= dmin for p in out):
            out.append((xp, fwhm(xx, yy, int(i))))
        if len(out) == n:
            break
    return sorted(out)


def make_calibration() -> tuple[np.ndarray, np.ndarray]:
    """Fit raw spectrometer axis to He-Ne plasma literature lines."""
    x, y = read_xy(DATA / "LaserCalibration(5s).csv")
    measured = []
    for seed in MEASURED_SEEDS:
        m = (x > seed - 3) & (x < seed + 3)
        xx, yy = x[m], smooth(y[m], 5)
        measured.append(vertex(xx, yy, int(np.argmax(yy))) if len(xx) else seed)
    measured = np.array(measured)
    coeff = np.polyfit(measured, KNOWN_LINES, 1)
    residual = np.polyval(coeff, measured) - KNOWN_LINES

    xc = np.polyval(coeff, x)
    plt.figure(figsize=(9, 4))
    plt.plot(xc, y / 5.0, lw=1)
    for lam in KNOWN_LINES:
        plt.axvline(lam, ls="--", alpha=0.35)
    plt.xlabel("calibrated wavelength / nm")
    plt.ylabel("counts / s")
    plt.title("He-Ne plasma calibration spectrum")
    plt.tight_layout(); plt.savefig(OUT / "01_calibration_spectrum.png", dpi=300); plt.close()

    plt.figure(figsize=(5, 4))
    grid = np.linspace(measured.min() - 5, measured.max() + 5, 200)
    plt.plot(measured, KNOWN_LINES, "o", label="assigned peaks")
    plt.plot(grid, np.polyval(coeff, grid), label=f"lambda={coeff[0]:.6f}x+{coeff[1]:.3f}")
    plt.xlabel("raw axis / nm"); plt.ylabel("known wavelength / nm")
    plt.title("Calibration curve"); plt.legend(); plt.tight_layout()
    plt.savefig(OUT / "02_calibration_curve.png", dpi=300); plt.close()

    plt.figure(figsize=(5, 3))
    plt.axhline(0, color="k", lw=0.8); plt.plot(KNOWN_LINES, residual, "o-")
    plt.xlabel("known wavelength / nm"); plt.ylabel("residual / nm")
    plt.title("Calibration residuals"); plt.tight_layout()
    plt.savefig(OUT / "03_calibration_residuals.png", dpi=300); plt.close()
    return coeff, residual


def background(sec: float) -> tuple[np.ndarray, np.ndarray] | None:
    files = sorted((DATA / "noise_background").glob(f"{int(sec)}sec_*.csv"))
    if not files:
        return None
    x0, y0 = read_xy(files[0])
    rates = [y0 / sec]
    for f in files[1:]:
        x, y = read_xy(f)
        rates.append(np.interp(x0, x, y / sec))
    return x0, np.mean(rates, axis=0)


def corrected(path: Path, coeff: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calibration -> counts/s -> background subtraction -> despiking."""
    x, y = read_xy(path)
    sec = exposure(path)
    rate = y / sec
    bg = background(sec)
    if bg is not None:
        rate -= np.interp(x, bg[0], bg[1])
    return np.polyval(coeff, x), despike(rate)


def numbered(folder: Path) -> list[Path]:
    return sorted([p for p in folder.glob("*.csv") if p.stem.isdigit()], key=lambda p: int(p.stem))


def varshni(T: float, Eg0: float) -> float:
    return Eg0 - VAR_A * T * T / (T + VAR_B)


def temp_from_E(E: float, Eg0: float) -> float:
    """Invert Eg(T)=Eg(0)-a*T^2/(T+b) by bisection."""
    if not np.isfinite(E):
        return float("nan")
    lo, hi = 0.0, 400.0
    if E >= varshni(lo, Eg0):
        return 0.0
    if E <= varshni(hi, Eg0):
        return hi
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if varshni(mid, Eg0) > E:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def series(name: str, coeff: np.ndarray) -> list[dict[str, float | str]]:
    """Process cooldown/warmup spectra and calculate MQW temperatures."""
    files = numbered(DATA / name)
    rows = []
    for f in files:
        x, y = corrected(f, coeff)
        lam, w = peak(x, y, MQW_TEMP_WIN)
        rows.append({"series": name, "index": int(f.stem), "file": str(f.relative_to(ROOT)),
                     "lambda_nm": lam, "energy_eV": HC / lam, "fwhm_nm": w})
    if rows:
        cold_E = np.nanmax([r["energy_eV"] for r in rows])
        Eg0_eff = cold_E + VAR_A * 4.2**2 / (4.2 + VAR_B)
        for r in rows:
            r["temperature_K"] = temp_from_E(float(r["energy_eV"]), Eg0_eff)
            r["Eg0_eff_eV"] = Eg0_eff

        plt.figure(figsize=(9, 5))
        for j, f in enumerate(files):
            x, y = corrected(f, coeff)
            scale = np.nanmax(np.abs(y)) or 1.0
            plt.plot(x, y / scale + j, lw=0.8)
        plt.xlabel("wavelength / nm"); plt.ylabel("normalized counts/s + offset")
        plt.title(f"{name}: processed spectra"); plt.tight_layout()
        plt.savefig(OUT / f"04_{name}_spectra.png", dpi=300); plt.close()
    return rows


def finite_even(L: float, mt: float, mB: float, V: float) -> float:
    """Lowest even finite-well state from Eq. (16)."""
    if L <= 0 or V <= 0:
        return float("nan")
    lo = 1e-12
    hi = min(V * (1 - 1e-9), math.pi**2 * HBAR2_2M0 / (mt * L * L) * (1 - 1e-9))
    def f(E: float) -> float:
        arg = math.sqrt(E * mt / HBAR2_2M0) * L / 2
        return math.sqrt(mB * E / (mt * (V - E))) * math.tan(arg) - 1
    if f(lo) > 0 or f(hi) < 0:
        return float("nan")
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def exciton_mev(L: float, x: float) -> float:
    """Heavy-hole 2D exciton correction, visually digitized from Fig. 10."""
    Ltab = np.array([2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40], float)
    e015 = np.array([5.9, 7.5, 8.1, 8.2, 8.0, 7.6, 7.3, 6.8, 6.4, 5.7, 5.2])
    e030 = np.array([6.6, 8.6, 9.4, 9.5, 9.2, 8.7, 8.3, 7.5, 6.8, 5.8, 5.3])
    w = np.clip((x - 0.15) / 0.15, 0, 1)
    return float(np.interp(L, Ltab, (1 - w) * e015 + w * e030))


def invert_width(widths: np.ndarray, curve: np.ndarray, target: float) -> float:
    ok = np.isfinite(curve)
    if ok.sum() < 2 or target < np.nanmin(curve[ok]) or target > np.nanmax(curve[ok]):
        return float("nan")
    order = np.argsort(curve[ok])
    return float(np.interp(target, curve[ok][order], widths[ok][order]))


def resolution_theory(lam_nm: float = 760.0) -> float:
    """Theoretical spectrometer resolution with corrected g, f and slit width."""
    lam = lam_nm * 1e-6
    g = 1 / GRATING_LINES_MM
    fac = math.sqrt(max(4 * g * g / (ORDER**2 * lam * lam) - 1, 0))
    return lam * fac * SLIT_MM / (2 * FOCAL_MM) * 1e6


def low_temperature(coeff: np.ndarray) -> list[dict[str, float | str]]:
    """Use bottom spectra for Al content, offsets, QW widths and exciton values."""
    files = sorted((DATA / "cooldown").glob("bottom*sec_*.csv")) or numbered(DATA / "cooldown")[-3:]
    x0, y0 = corrected(files[0], coeff)
    ys = [y0]
    for f in files[1:]:
        x, y = corrected(f, coeff)
        ys.append(np.interp(x0, x, y))
    x, y = x0, np.mean(ys, axis=0)

    al_lam, _ = peak(x, y, ALGAAS_WIN)
    gaas_lam, _ = peak(x, y, GAAS_WIN)
    qw = peaks(x, y, QW_WIN, n=6)

    E_al = HC / al_lam
    x_al = float(np.clip((E_al - EG_GAAS) / 1.247, 0, 0.6))
    dEg = 1.247 * x_al
    dEc, dEv = 0.65 * dEg, 0.35 * dEg

    me_w, mh_w = 0.067, 0.34
    me_b, mh_b = 0.067 + 0.084 * x_al, 0.34 + 0.175 * x_al
    L = np.linspace(1.5, 40, 600)
    ee = np.array([finite_even(l, me_w, me_b, dEc) for l in L])
    eh = np.array([finite_even(l, mh_w, mh_b, dEv) for l in L])
    finite_curve = ee + eh - np.array([exciton_mev(l, x_al) for l in L]) / 1000
    infinite_curve = math.pi**2 * HBAR2_2M0 * (1 / me_w + 1 / mh_w) / L**2

    rows: list[dict[str, float | str]] = [
        {"quantity": "AlGaAs peak wavelength", "value": al_lam, "unit": "nm"},
        {"quantity": "AlGaAs peak energy", "value": E_al, "unit": "eV"},
        {"quantity": "aluminium fraction x", "value": x_al, "unit": ""},
        {"quantity": "bandgap difference Delta Eg", "value": dEg, "unit": "eV"},
        {"quantity": "conduction-band offset Delta E_L", "value": dEc, "unit": "eV"},
        {"quantity": "valence-band offset Delta E_V", "value": dEv, "unit": "eV"},
    ]
    if np.isfinite(gaas_lam):
        E_gaas = HC / gaas_lam
        mu = me_w * mh_w / (me_w + mh_w)
        rows += [
            {"quantity": "GaAs substrate wavelength", "value": gaas_lam, "unit": "nm"},
            {"quantity": "GaAs substrate energy", "value": E_gaas, "unit": "eV"},
            {"quantity": "measured 3D exciton binding", "value": 1000 * (EG_GAAS - E_gaas), "unit": "meV"},
            {"quantity": "theoretical 3D exciton binding", "value": 1000 * RYDBERG_EV * mu / EPS_GAAS**2, "unit": "meV"},
        ]
    for i, (lam, w) in enumerate(qw, 1):
        E = HC / lam
        excess = E - EG_GAAS
        rows += [
            {"quantity": f"QW{i} wavelength", "value": lam, "unit": "nm"},
            {"quantity": f"QW{i} energy", "value": E, "unit": "eV"},
            {"quantity": f"QW{i} finite-well width", "value": invert_width(L, finite_curve, excess), "unit": "nm"},
            {"quantity": f"QW{i} infinite-well width", "value": invert_width(L, infinite_curve, excess), "unit": "nm"},
            {"quantity": f"QW{i} FWHM", "value": w, "unit": "nm"},
        ]

    cx, cy = read_xy(DATA / "LaserCalibration(5s).csv")
    cx = np.polyval(coeff, cx)
    widths = [peak(cx, cy, (lam - 3, lam + 3))[1] for lam in KNOWN_LINES]
    rows += [
        {"quantity": "mean measured spectral resolution", "value": float(np.nanmean(widths)), "unit": "nm"},
        {"quantity": "theoretical spectral resolution at 760 nm", "value": resolution_theory(), "unit": "nm"},
    ]

    plt.figure(figsize=(9, 5))
    plt.plot(x, y, lw=1)
    for lam, label in [(al_lam, "AlGaAs"), (gaas_lam, "GaAs")]:
        if np.isfinite(lam):
            plt.axvline(lam, ls="--", alpha=0.5); plt.text(lam, plt.ylim()[1]*0.92, label, rotation=90, va="top")
    for lam, _ in qw:
        plt.axvline(lam, ls=":", alpha=0.4)
    plt.xlabel("wavelength / nm"); plt.ylabel("background-subtracted counts / s")
    plt.title("Cold/bottom average spectrum with identified peaks")
    plt.tight_layout(); plt.savefig(OUT / "06_bottom_average_peaks.png", dpi=300); plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(L, EG_GAAS + finite_curve, label="finite well + hh exciton correction")
    plt.plot(L, EG_GAAS + infinite_curve, label="infinite well")
    for i, (lam, _) in enumerate(qw, 1):
        plt.axhline(HC / lam, ls="--", alpha=0.35); plt.text(L[-1], HC / lam, f" QW{i}", va="center")
    plt.xlabel("GaAs well width / nm"); plt.ylabel("transition energy / eV")
    plt.title("Quantum-well width estimate")
    plt.legend(); plt.tight_layout(); plt.savefig(OUT / "07_well_width_model.png", dpi=300); plt.close()
    return rows


def write_tables(cal_coeff: np.ndarray, cal_resid: np.ndarray, values: list[dict[str, float | str]], temps: list[dict[str, float | str]]) -> None:
    with (OUT / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["quantity", "value", "unit"]); w.writeheader()
        w.writerow({"quantity": "calibration slope", "value": cal_coeff[0], "unit": "nm/raw-nm"})
        w.writerow({"quantity": "calibration intercept", "value": cal_coeff[1], "unit": "nm"})
        w.writerow({"quantity": "calibration RMS residual", "value": float(np.sqrt(np.mean(cal_resid**2))), "unit": "nm"})
        w.writerows(values)
    with (OUT / "temperature_series.csv").open("w", newline="") as f:
        fields = ["series", "index", "file", "lambda_nm", "energy_eV", "fwhm_nm", "temperature_K", "Eg0_eff_eV"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in temps:
            w.writerow({k: r.get(k, "") for k in fields})
    with (OUT / "summary.txt").open("w") as f:
        f.write("FoPra 45 analysis summary\n=========================\n\n")
        f.write(f"Calibration: lambda = {cal_coeff[0]:.8f} * raw + {cal_coeff[1]:.5f} nm\n")
        f.write(f"Calibration RMS residual: {np.sqrt(np.mean(cal_resid**2)):.3f} nm\n\n")
        for r in values:
            v = r["value"]
            f.write(f"{r['quantity']}: {v:.6g} {r['unit']}\n" if isinstance(v, float) else f"{r['quantity']}: {v} {r['unit']}\n")


def main() -> None:
    coeff, resid = make_calibration()
    temps = series("cooldown", coeff) + series("warmup", coeff)
    if temps:
        plt.figure(figsize=(7, 4))
        for name in sorted({r["series"] for r in temps}):
            rr = [r for r in temps if r["series"] == name]
            plt.plot([r["index"] for r in rr], [r["temperature_K"] for r in rr], "o-", label=name)
        plt.xlabel("spectrum number"); plt.ylabel("temperature / K")
        plt.title("Temperature from MQW luminescence"); plt.legend(); plt.tight_layout()
        plt.savefig(OUT / "05_temperature_series.png", dpi=300); plt.close()
    values = low_temperature(coeff)
    write_tables(coeff, resid, values, temps)
    print(f"Done. See {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
