from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

HC_EV_NM = 1239.50  # eV nm


def _as_list(value, n, default=None):
    if value is None:
        return [default] * n
    if isinstance(value, (str, Path)) or not hasattr(value, "__iter__"):
        return [value] * n
    value = list(value)
    if len(value) != n:
        raise ValueError(f"Expected {n} values, got {len(value)}.")
    return value


def _apply_calibration(x_raw, calibration=None, x_shift_nm=0.0):
    """
    calibration can be:
      None                 -> x_raw + x_shift_nm
      callable             -> calibration(x_raw) + x_shift_nm
      polynomial coeffs    -> np.polyval(calibration, x_raw) + x_shift_nm
    """
    x_raw = np.asarray(x_raw, dtype=float)

    if calibration is None:
        return x_raw + x_shift_nm
    if callable(calibration):
        return calibration(x_raw) + x_shift_nm

    return np.polyval(np.asarray(calibration, dtype=float), x_raw) + x_shift_nm


def _simple_peak_table(x, y, n_peaks=8, peak_sigma=4):
    """Small dependency-free local-maximum peak finder for quick inspection."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    median = np.nanmedian(y)
    mad = np.nanmedian(np.abs(y - median))
    noise = 1.4826 * mad if mad > 0 else np.nanstd(y)
    threshold = median + peak_sigma * noise

    idx = np.where(
        (y[1:-1] > y[:-2]) &
        (y[1:-1] >= y[2:]) &
        (y[1:-1] > threshold)
    )[0] + 1

    if len(idx) == 0:
        return pd.DataFrame(columns=["x_nm", "energy_eV", "counts"])

    idx = idx[np.argsort(y[idx])[::-1]][:n_peaks]

    peaks = pd.DataFrame({
        "x_nm": x[idx],
        "energy_eV": HC_EV_NM / x[idx],
        "counts": y[idx],
    })

    return peaks.sort_values("x_nm").reset_index(drop=True)


def plot_and_inspect_spectra(
    paths,
    labels=None,
    integration_times=None,
    calibration=None,
    x_shift_nm=0.0,
    smooth_window=1,
    n_peaks=8,
    peak_sigma=4,
    xlim=None,
    ylim=None,
    energy_axis=True,
    annotate_peaks=True,
):
    """
    Plot and inspect one or more FoPra-style spectrum CSV files.

    Parameters
    ----------
    paths : str/path or list[str/path]
        CSV file(s), expected as two columns: x, counts.
    labels : list[str], optional
        Plot labels. Defaults to file names.
    integration_times : float or list[float], optional
        Exposure times in seconds. If given, counts are plotted as counts/s.
    calibration : None, callable, or polynomial coefficients
        Use np.polyfit(measured_peak_positions, known_wavelengths, deg=1 or 2)
        and pass the resulting coefficients here.
    x_shift_nm : float
        Quick constant wavelength shift, e.g. 12 for a rough first look.
        Prefer a fitted calibration for report-quality analysis.
    smooth_window : int
        Rolling median window. Use 1 for raw data, 3 or 5 to suppress spikes.
    n_peaks : int
        Number of strongest local peaks to print.
    peak_sigma : float
        Peak threshold in robust-noise units.
    """

    if isinstance(paths, (str, Path)):
        paths = [paths]
    else:
        paths = list(paths)

    n = len(paths)
    labels = _as_list(labels, n)
    integration_times = _as_list(integration_times, n)

    fig, ax = plt.subplots(figsize=(10, 5))

    spectra = {}
    peak_tables = {}

    for path, label, t_exp in zip(paths, labels, integration_times):
        path = Path(path)
        label = label or path.name

        df = pd.read_csv(
            path,
            header=None,
            names=["x_raw", "counts"],
            sep=r"\s*,\s*",
            engine="python",
        )

        df = df.apply(pd.to_numeric, errors="coerce").dropna()
        df = df.sort_values("x_raw").reset_index(drop=True)

        df["x_nm"] = _apply_calibration(
            df["x_raw"].to_numpy(),
            calibration=calibration,
            x_shift_nm=x_shift_nm,
        )

        y = df["counts"].to_numpy(dtype=float)

        if t_exp is not None:
            y = y / float(t_exp)
            y_label = "counts / s"
        else:
            y_label = "counts"

        if smooth_window and smooth_window > 1:
            window = int(smooth_window)
            if window % 2 == 0:
                window += 1

            y_plot = (
                pd.Series(y)
                .rolling(window=window, center=True, min_periods=1)
                .median()
                .to_numpy()
            )
        else:
            y_plot = y

        df["y_plot"] = y_plot
        spectra[label] = df

        ax.plot(df["x_nm"], y_plot, label=label)

        peaks = _simple_peak_table(
            df["x_nm"].to_numpy(),
            y_plot,
            n_peaks=n_peaks,
            peak_sigma=peak_sigma,
        )

        peak_tables[label] = peaks

        print(f"\n{label}")
        print(f"  points: {len(df)}")
        print(f"  x range: {df['x_nm'].min():.2f} to {df['x_nm'].max():.2f} nm")
        print(f"  counts range: {np.nanmin(y):.2f} to {np.nanmax(y):.2f}")

        if len(peaks):
            print("  strongest local peaks:")
            print(peaks.round({"x_nm": 2, "energy_eV": 4, "counts": 1}).to_string(index=False))
        else:
            print("  no strong local peaks found; lower peak_sigma if needed")

        if annotate_peaks and len(peaks):
            for _, row in peaks.head(min(5, len(peaks))).iterrows():
                ax.axvline(row["x_nm"], linestyle="--", alpha=0.25)

    ax.set_xlabel("wavelength / nm")
    ax.set_ylabel(y_label)
    ax.set_title("Spectrum inspection")
    ax.grid(True, alpha=0.3)

    if n > 1:
        ax.legend()

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    if energy_axis:
        def nm_to_ev(lam):
            return HC_EV_NM / np.asarray(lam)

        def ev_to_nm(ev):
            return HC_EV_NM / np.asarray(ev)

        secax = ax.secondary_xaxis("top", functions=(nm_to_ev, ev_to_nm))
        secax.set_xlabel("energy / eV")

    fig.tight_layout()

    out_dir = Path("plotsOutput")
    out_dir.mkdir(exist_ok=True)

    plot_name = "_".join(str(label) for label in labels if label)
    plot_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", plot_name).strip("_")

    fig.savefig(out_dir / f"{plot_name}.png", dpi=300, bbox_inches="tight")
    plt.show()

    return spectra, peak_tables, fig, ax





# known_lines_nm = np.array([710, 730, 733, 755, 760, 845, 850])
known_lines_nm = np.array([
    710, 730, 733, 755, 760, 845
])

measured_lines_nm = np.array([
    705.70, 723.66, 727.24, 747.97, 752.56, 836.87
])

calibration = np.polyfit(measured_lines_nm, known_lines_nm, deg=1)

fit_values = np.polyval(calibration, measured_lines_nm)
residuals = fit_values - known_lines_nm

for measured, known, fitted, residual in zip(
    measured_lines_nm, known_lines_nm, fit_values, residuals
):
    print(f"measured {measured:7.2f} nm -> fitted {fitted:7.2f} nm, "
          f"known {known:7.2f} nm, residual {residual:+.2f} nm")




spectra, peaks, fig, ax = plot_and_inspect_spectra(
    "CiaraDavidNiccolo/warmup/30.csv",
    calibration=calibration,
    smooth_window=3,
    labels=["sample spectrum warmup file 30 calibrated"],
    integration_times=[5],
)