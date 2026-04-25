# GOES Unified Architecture

> **Status**: APPROVED - P0-G1 complete  
> **Approved**: 2026-04-25  
> **Depends on**: None  
> **Consumed by**: P0-G2, P0-A2, P0-G3, P0-G6, P0-D2, P0-G4

---

## 1. Purpose

The SWMI pipeline uses GOES data for two distinct scientific inputs:

- **Magnetometer** data provides a geosynchronous proxy for magnetopause
  compression and near-Earth magnetic disturbance.
- **X-ray flux** provides solar flare precursor information that can lead
  geoeffective solar wind signatures by hours to days.

Both products must be retrieved through a single GOES module,
`src/swmi/api/goes.py`, with deterministic satellite routing, canonical
schemas, and schema validation before every Parquet write.

---

## 2. Class Hierarchy

```python
class BaseRetriever(ABC):
    config: dict
    output_root: Path
    product: str

    @abstractmethod
    def retrieve(self, year: int, month: int) -> pd.DataFrame:
        ...

    def detect_era(self, satellite: str | int) -> Literal["legacy", "modern"]:
        ...

    def month_bounds(self, year: int, month: int) -> tuple[pd.Timestamp, pd.Timestamp]:
        ...

    def expected_minute_index(self, year: int, month: int) -> pd.DatetimeIndex:
        ...

    def validate_and_write(
        self,
        df: pd.DataFrame,
        schema_name: str,
        output_path: Path,
        *,
        unique_subset: list[str],
    ) -> Path:
        ...
```

`BaseRetriever` owns behavior shared by GOES products: YAML config loading,
UTC month boundaries, 1-minute canonical index construction, satellite era
detection, output directory creation, and calls to `validate_output_schema()`.

```python
class GOESMagRetriever(BaseRetriever):
    product = "mag"

    def retrieve(self, year: int, month: int) -> pd.DataFrame:
        ...
```

`GOESMagRetriever` retrieves or loads all configured magnetometer satellites
for a month, normalizes each satellite to the canonical magnetometer schema,
and calls `merge_goes_satellites()` for deterministic primary/backup fusion.

```python
class GOESXrayRetriever(BaseRetriever):
    product = "xray"

    def retrieve(self, year: int, month: int) -> pd.DataFrame:
        ...
```

`GOESXrayRetriever` retrieves or loads XRS flux for all configured satellites
for a month, applies era-specific quality filtering, and writes canonical
Earth-observed flux. It must not apply an `au_factor`.

---

## 3. Era Detection

Era is determined by satellite number, not by calendar year alone:

| Era | Satellite numbers | Parser family | Notes |
|---|---:|---|---|
| Legacy | GOES-13, GOES-14, GOES-15 | NCEI legacy NetCDF | Used in 2015-2016 priority chain; overlaps modern era through 2020 |
| Modern | GOES-16, GOES-17, GOES-18, GOES-19 | NGDC/modern netCDF4 | Operational for 2017+ study years |

The implementation should parse either `"GOES-16"`, `"goes16"`, `"g16"`, or
integer `16` into a satellite number. Satellite numbers `<= 15` route to the
legacy parser; satellite numbers `>= 16` route to the modern parser.

Calendar year is used only to select the deterministic priority order from
`configs/data_retrieval.yaml`:

```yaml
goes:
  satellite_priority:
    2015: ["GOES-15", "GOES-13"]
    2016: ["GOES-15", "GOES-13"]
    2017: ["GOES-16", "GOES-15"]
    2018: ["GOES-16", "GOES-15"]
    2019: ["GOES-16", "GOES-17"]
    2020: ["GOES-16", "GOES-17"]
    2021: ["GOES-16", "GOES-17"]
    2022: ["GOES-16", "GOES-18"]
    2023: ["GOES-16", "GOES-18"]
```

`P0-B3` will expand this config with explicit operational periods. Retrieval
must check those periods once they exist, but priority order remains the
authoritative merge order.

---

## 4. Canonical Magnetometer Schema

Monthly GOES magnetometer output is written to:

```text
data/raw/goes/goes_mag_YYYYMM.parquet
```

Required columns:

| Column | Type | Meaning |
|---|---|---|
| `timestamp` | UTC datetime64, 1-minute cadence | Canonical minute timestamp |
| `goes_bz_gsm` | float64 | GSM Bz magnetic field component in nT |
| `goes_source_satellite` | string | Satellite selected for that timestamp, e.g. `GOES-16` |
| `goes_mag_missing_flag` | int8 | `1` when no configured satellite has valid Bz at that minute |

Optional diagnostic columns may be retained when useful:

| Column | Type | Meaning |
|---|---|---|
| `goes_bz_gsm_raw` | float64 | Unmodified source value before canonical filtering |
| `goes_quality_flag` | int/string | Source-specific magnetometer quality flag |
| `goes_era` | string | `legacy` or `modern` |

Canonical output must contain one row per expected minute from month start
inclusive to next month start exclusive. Missing minutes are represented as
`NaN` in `goes_bz_gsm` with `goes_mag_missing_flag = 1`, not by dropping rows.

---

## 5. Magnetometer Merge Strategy

`merge_goes_satellites()` accepts a list or mapping of satellite DataFrames and
the configured priority order for the target year.

Algorithm:

1. Build the complete UTC 1-minute index for the requested month.
2. Reindex every satellite DataFrame to that index.
3. Visit satellites in the exact order from `goes.satellite_priority[year]`.
4. For each timestamp, choose the first satellite with a finite, quality-passing
   `goes_bz_gsm`.
5. Record the selected satellite in `goes_source_satellite`.
6. Set `goes_mag_missing_flag = 1` only if all configured satellites are missing
   or invalid at that timestamp.
7. Assert that output timestamps are unique and monotonic.

This strategy is deterministic and does not depend on filesystem ordering,
download order, row order, or pandas concatenation side effects.

---

## 6. Canonical X-Ray Schema

Monthly GOES X-ray output is written to:

```text
data/raw/goes/goes_xray_YYYYMM.parquet
```

Required columns:

| Column | Type | Meaning |
|---|---|---|
| `timestamp` | UTC datetime64, 1-minute cadence | Canonical minute timestamp |
| `xrsa_flux` | float64 | Short-channel X-ray flux, Earth-observed W/m^2 |
| `xrsb_flux` | float64 | Long-channel X-ray flux, Earth-observed W/m^2 |
| `xray_quality_flags` | string/int | Product-specific quality summary retained for auditing |
| `xray_source_satellite` | string | Satellite selected for that timestamp |
| `xray_missing_flag` | int8 | `1` when no configured satellite has valid XRS data |

No distance normalization is applied. In particular, the pipeline must not use
or derive an `au_factor`; Earth-observed irradiance is the intended input.

---

## 7. X-Ray Quality Filtering

Quality filtering is era-specific and happens before merge and normalization.

Legacy XRS:

- Use the legacy 4-flag quality system.
- Keep only records where the source product marks flux as good.
- Reject non-positive flux before log transforms.

Modern XRS:

- Reject `electron_correction_flag == 1`.
- Reject `electron_correction_flag == 4`.
- Reject contamination bits `8, 16, 32, 64, 128, 256`.
- Reject non-positive flux before log transforms.

The retriever writes cleaned but unnormalized flux. Cross-satellite
normalization belongs in `normalize_goes_xray()` in
`src/swmi/preprocessing/cleaners.py`.

---

## 8. X-Ray Normalization And Feature Integration

`normalize_goes_xray()` consumes canonical X-ray Parquet and applies:

1. Era-specific quality filtering if raw diagnostic flags remain present.
2. `log10` transform of positive flux.
3. Per-satellite quiet-Sun baseline subtraction.
4. NOAA scale factors for legacy-to-modern continuity.
5. Validation plot output under `results/figures/`.

Feature engineering must treat X-ray as a solar precursor. `add_xray_features()`
should use event-driven accumulators, not rolling windows:

- `goes_xray_long_log`
- `goes_xray_long_dlog_dt`
- `goes_xray_time_since_last_c_flare`
- `goes_xray_time_since_last_m_flare`
- `goes_xray_time_since_last_x_flare`
- `goes_xray_cumulative_m_class_24h`
- `goes_xray_max_flux_24h`

---

## 9. Validation Requirements

Before every Parquet write:

- Call `validate_output_schema()` with the appropriate GOES schema name.
- Enforce unique `timestamp` rows.
- Enforce monotonic UTC timestamps.
- Enforce the configured `master_cadence` of `1min`.
- Preserve missingness with explicit missing flags.

Completeness reporting should include:

- Total expected minutes.
- Valid minutes by source satellite.
- Minutes filled by backup satellites.
- Minutes missing after all configured satellites are considered.

---

## 10. Non-Goals

- Do not implement MMS.
- Do not implement DMSP as part of the GOES chain.
- Do not modify `newell_coupling.py` physics.
- Do not compute SuperMAG dB/dt from `.mag` or `.geo` coordinates.
- Do not apply `au_factor` to X-ray flux.