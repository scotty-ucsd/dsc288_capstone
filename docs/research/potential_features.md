Summary Statistics
Total Feature Count (Approximate)

Raw Features:

    Sun (GOES-R): 12 raw features
    L1 (OMNI): ~25 raw features
    GEO (GOES): ~20 raw features (more with multi-sat + particles)
    LEO (Swarm): ~15 raw features per satellite
    Ground (SuperMAG): ~7 per station + 4 indices = ~100+ with network

Engineered Features:

    Category 1 (Solar Wind Preconditioning): ~30 features
    Category 2 (GEO Magnetospheric State): ~35 features
    Category 3 (Solar Wind Components): ~50 features
    Category 4 (Interaction Terms): ~30 features
    Category 5 (Temporal Context): ~50 features

Total: ~400-500+ features depending on:

    Number of ground stations included
    Whether multi-satellite data used
    Time window variations
    Particle data availability


Sun: GOES-R
Raw Features

    xray_flux_short (0.05-0.4 nm, W/m²)
    xray_flux_long (0.1-0.8 nm, W/m²)
    xray_class (A/B/C/M/X classification)
    euv_flux_0304 (30.4 nm, He II line)
    euv_flux_1216 (121.6 nm, Lyman-alpha)
    solar_wind_speed_proxy (from coronal hole models if available)
    flare_occurrence (boolean, past 24h)
    flare_magnitude (peak X-ray flux)
    cme_occurrence (boolean, past 48h from coronagraph)
    active_region_count
    active_region_complexity (Hale classification)
    sunspot_number

Engineered Features

    xray_rate_of_change = d(xray_flux)/dt
    xray_max_24h = max(xray_flux, t-24h to t)
    flare_time_since_last = t - t_last_flare
    flare_count_24h = count(flares in past 24h)
    euv_variability = std(euv_flux, t-3h to t)
    solar_active = boolean(xray_flux > threshold OR flare_24h > 0)
    cme_transit_expected = boolean(CME launched 1-3 days ago)
    euv_trend_24h = linear_slope(euv_flux, t-24h to t)

Formatted XRS Features
Sun
Source: GOES-R XRS (X-Ray Sensor)
Raw Features

    xrs_flux_short (W/m², 0.05-0.4 nm band, XRS-A)
    xrs_flux_long (W/m², 0.1-0.8 nm band, XRS-B)
    xrs_class_short (classification: A/B/C/M/X for short band)
    xrs_class_long (classification: A/B/C/M/X for long band)
    xrs_timestamp (time of observation)
    xrs_quality_flag (data quality indicator)

Engineered Features

Solar Activity State:

    xrs_ratio = xrs_flux_short / xrs_flux_long (coronal temperature proxy)
    xrs_flux_avg_7d = mean(xrs_flux_long, t-7d to t) - background activity level
    xrs_flux_avg_24h = mean(xrs_flux_long, t-24h to t)
    xrs_flux_std_7d = std(xrs_flux_long, t-7d to t) - variability measure
    xrs_above_baseline = boolean(xrs_flux_long > 1.5 × xrs_flux_avg_7d)
    xrs_elevated_state = boolean(xrs_flux_avg_24h > C-class threshold)

Flare Activity:

    dXRS_dt = (xrs_flux_long[t] - xrs_flux_long[t-1min]) / 1min - rate of change
    dXRS_dt_max_1h = max(dXRS_dt, t-1h to t)
    flare_detected = boolean(dXRS_dt > threshold AND xrs_flux > baseline)
    flare_count_24h = count(flare events in past 24h)
    flare_count_7d = count(flare events in past 7 days)
    time_since_last_flare = t - t_last_flare
    last_flare_magnitude = max(xrs_flux) during last flare
    last_flare_class = classification of most recent flare

Temporal Patterns:

    xrs_trend_24h = linear_slope(xrs_flux_long, t-24h to t)
    xrs_trend_7d = linear_slope(xrs_flux_long, t-7d to t)
    xrs_increasing = boolean(xrs_trend_24h > 0)
    xrs_volatility_24h = std(dXRS_dt, t-24h to t)

Coronal Context (if combined with EUV):

    active_sun_indicator = boolean(xrs_ratio > threshold OR flare_count_24h > 0)
    quiet_sun_period = boolean(xrs_flux_avg_7d < A-class AND flare_count_7d = 0)
    elevated_background = xrs_flux_avg_24h / xrs_flux_avg_7d

L1: OMNI
Raw Features

Magnetic Field (IMF):

    Bx_gsm (nT, GSM coordinates)
    By_gsm (nT, GSM coordinates)
    Bz_gsm (nT, GSM coordinates)
    B_total (nT, total field magnitude)
    B_theta (degrees, clock angle)
    B_phi (degrees, cone angle)

Plasma:

    proton_density (n, cm⁻³)
    proton_temperature (T, K)
    proton_speed (v, km/s)
    flow_pressure (nPa)
    alpha_proton_ratio
    flow_azimuth (degrees)
    flow_elevation (degrees)

Derived OMNI Products:

    electric_field_ey (mV/m, dawn-dusk)
    plasma_beta
    alfven_mach_number
    magnetosonic_mach_number

Indices (if included in OMNI):

    kp_index
    dst_index (if preliminary available)
    ae_index (if preliminary available)

Engineered Features

Category 1: Solar Wind Preconditioning

    newell_coupling_current = v^(4/3) × B_T^(2/3) × sin^8(θ_c/2) where θ_c = IMF clock angle
    newell_integral_48h = ∫(newell_coupling) dt from t-48h to t
    newell_integral_24h = ∫(newell_coupling) dt from t-24h to t
    newell_integral_12h = ∫(newell_coupling) dt from t-12h to t
    newell_integral_6h = ∫(newell_coupling) dt from t-6h to t
    newell_slope_48h = linear_slope(newell_coupling, t-48h to t)
    newell_slope_24h = linear_slope(newell_coupling, t-24h to t)
    newell_slope_12h = linear_slope(newell_coupling, t-12h to t)
    newell_avg_48h = mean(newell_coupling, t-48h to t)
    newell_avg_24h = mean(newell_coupling, t-24h to t)
    newell_avg_12h = mean(newell_coupling, t-12h to t)
    newell_avg_6h = mean(newell_coupling, t-6h to t)
    newell_max_24h = max(newell_coupling, t-24h to t)
    newell_std_24h = std(newell_coupling, t-24h to t)

Category 3: Solar Wind Components

    Bz_min_3h = min(Bz_gsm, t-3h to t)
    Bz_min_6h = min(Bz_gsm, t-6h to t)
    Bz_min_12h = min(Bz_gsm, t-12h to t)
    Bz_avg_3h = mean(Bz_gsm, t-3h to t)
    Bz_std_3h = std(Bz_gsm, t-3h to t)
    Bz_southward_duration = cumulative_time(Bz < -2 nT in past 6h)
    Bz_southward_fraction_6h = fraction_time(Bz < 0 in past 6h)
    By_avg_3h = mean(By_gsm, t-3h to t)
    B_total_avg_3h = mean(B_total, t-3h to t)
    B_total_max_3h = max(B_total, t-3h to t)
    dBz_dt_1h = (Bz[t] - Bz[t-1h]) / 1h
    dBz_dt_30m = (Bz[t] - Bz[t-30m]) / 30m
    dBz_dt_15m = (Bz[t] - Bz[t-15m]) / 15m
    dB_total_dt_1h = (B_total[t] - B_total[t-1h]) / 1h

Velocity & Pressure:

    v_avg_3h = mean(proton_speed, t-3h to t)
    v_max_3h = max(proton_speed, t-3h to t)
    v_std_3h = std(proton_speed, t-3h to t)
    dv_dt_1h = (v[t] - v[t-1h]) / 1h
    dv_dt_30m = (v[t] - v[t-30m]) / 30m
    ram_pressure_current = n × v² × m_p
    ram_pressure_avg_3h = mean(ram_pressure, t-3h to t)
    dP_dt_1h = (ram_pressure[t] - ram_pressure[t-1h]) / 1h
    dP_dt_30m = (ram_pressure[t] - ram_pressure[t-30m]) / 30m
    pressure_pulse = boolean(dP_dt_30m > threshold)

Density & Temperature:

    n_avg_3h = mean(proton_density, t-3h to t)
    n_std_3h = std(proton_density, t-3h to t)
    T_avg_3h = mean(proton_temperature, t-3h to t)
    dn_dt_1h = (n[t] - n[t-1h]) / 1h

Coupling Functions:

    epsilon_parameter = (v^(4/3) × B_T^(2/3) × sin^8(θ/2)) × l_0²
    epsilon_integral_6h = ∫(epsilon) dt from t-6h to t
    rectified_voltage = v × B_T × sin²(θ/2) × l_0
    wygant_coupling = v × Bz (if Bz < 0, else 0)
    borovsky_coupling = (v × B_T × sin²(θ/2))^2 / μ₀
    clock_angle = atan2(By, Bz)
    cone_angle = atan2(sqrt(By² + Bz²), Bx)

Trigger Indicators:

    dNewell_dt_1h = (newell[t] - newell[t-1h]) / 1h
    dNewell_dt_30m = (newell[t] - newell[t-30m]) / 30m
    dNewell_dt_15m = (newell[t] - newell[t-15m]) / 15m
    trigger_strength_1h = max(abs(dNewell_dt) in past 1h)
    Bz_sudden_turning = boolean(|dBz_dt_15m| > threshold)
    southward_turning_strength = dBz_dt if Bz crosses from + to -

Variability Measures:

    IMF_variability_1h = std([Bx, By, Bz], t-1h to t)
    velocity_variability_3h = std(v, t-3h to t)
    cone_angle_variability_3h = std(cone_angle, t-3h to t)


GEO: GOES
Raw Features

Magnetometer (GOES-16/17/18):

    Hp_geo (nT, parallel component in spacecraft coord)
    He_geo (nT, perpendicular-east component)
    Hn_geo (nT, perpendicular-north component)
    Ht_total_geo (nT, total field magnitude)

Transformed to Geophysical:

    Bx_gsm_geo (nT, GSM X at GEO)
    By_gsm_geo (nT, GSM Y at GEO)
    Bz_gsm_geo (nT, GSM Z at GEO)
    B_total_geo (nT, total field at GEO)

Energetic Particles (if GOES-R SEISS available):

    electron_flux_40keV (electrons/cm²/s/sr/keV)
    electron_flux_75keV
    electron_flux_150keV
    electron_flux_275keV
    electron_flux_475keV
    proton_flux_1MeV (protons/cm²/s/sr/MeV)
    proton_flux_10MeV
    proton_flux_100MeV

Multi-Satellite (if available):

    Bz_goes_east (GOES-East, ~75°W)
    Bz_goes_west (GOES-West, ~135°W)
    magnetic_local_time_goes_east (hours)
    magnetic_local_time_goes_west (hours)

Engineered Features

Category 2: GEO Magnetospheric State

Tail Stretching Indicators:

    Bz_baseline_quiet = median(Bz_geo) during known quiet times (model parameter)
    Bz_depression_fraction = (Bz_baseline - Bz_geo[t]) / Bz_baseline
    Bz_depression_absolute = Bz_baseline - Bz_geo[t]
    Bx_enhancement = Bx_geo[t] - Bx_baseline
    tail_stretch_index = Bx_geo / Bz_geo (higher = more tail-like)
    field_tilt_angle = atan2(Bx_geo, Bz_geo)
    stretch_duration = cumulative_time(Bz_depression > 0.3 in past 2h)

Rate of Stretching:

    dBz_geo_dt_30m = (Bz_geo[t] - Bz_geo[t-30m]) / 30m
    dBz_geo_dt_15m = (Bz_geo[t] - Bz_geo[t-15m]) / 15m
    dBz_geo_dt_5m = (Bz_geo[t] - Bz_geo[t-5m]) / 5m
    dBx_geo_dt_30m = (Bx_geo[t] - Bx_geo[t-30m]) / 30m
    d2Bz_geo_dt2 = second derivative of Bz (acceleration of stretching)

Minimum Bz (Onset Predictor):

    Bz_min_2h = min(Bz_geo, t-2h to t)
    Bz_min_1h = min(Bz_geo, t-1h to t)
    time_since_Bz_min = t - time_of_min(Bz_geo in past 2h)
    at_Bz_minimum = boolean(Bz_geo ≈ Bz_min_2h within tolerance)

Field Variability:

    Bz_std_1h = std(Bz_geo, t-1h to t)
    B_total_std_1h = std(B_total_geo, t-1h to t)
    Bz_range_1h = max(Bz_geo, t-1h to t) - min(Bz_geo, t-1h to t)

Dipolarization Detection (Post-Onset):

    Bz_increase_5m = max(0, Bz_geo[t] - Bz_geo[t-5m])
    rapid_Bz_increase = boolean(Bz_increase_5m > 10 nT)
    dipolarization_rate = max(dBz_geo_dt in past 15m)

Multi-Satellite Features (if available):

    Bz_east_west_difference = abs(Bz_goes_east - Bz_goes_west)
    Bz_east_west_ratio = Bz_goes_east / Bz_goes_west
    MLT_sector_east = magnetic_local_time_goes_east (binned: pre-midnight/midnight/post-midnight)
    MLT_sector_west = magnetic_local_time_goes_west
    midnight_sector_stretched = boolean(Bz_depression_goes_east > 0.4 AND MLT_east in [22-02])

Particle Features (if available):

    electron_flux_ratio_275_75 = flux_275keV / flux_75keV (spectral hardness)
    total_electron_flux_sum = sum(electron fluxes across energies)
    dFlux_dt_150keV = (flux_150keV[t] - flux_150keV[t-30m]) / 30m
    particle_injection_indicator = boolean(dFlux_dt_150keV > threshold × flux_150keV[t-30m])
    flux_dropout = boolean(flux_150keV < 0.1 × flux_150keV_24h_avg) - lobe indicator
    energetic_electron_present = boolean(flux_275keV > threshold)


LEO: Swarm
Raw Features

Magnetic Field (High Resolution, ~1 Hz):

    B_NEC_north (nT, North component in NEC frame)
    B_NEC_east (nT, East component)
    B_NEC_center (nT, Center/radial component)
    B_magnitude (nT, total field)
    latitude (degrees)
    longitude (degrees)
    magnetic_latitude (degrees, quasi-dipole)
    magnetic_local_time (hours)
    altitude (km)

Field-Aligned Currents (FAC product):

    FAC_density (μA/m², field-aligned current density)
    FAC_quality_flag

Plasma (Swarm Langmuir Probes):

    electron_density (cm⁻³)
    electron_temperature (K)

Multi-Satellite (Swarm A/B/C constellation):

    B_swarmA_north
    B_swarmB_north
    B_swarmC_north
    separation_AB (km, cross-track)
    separation_AC (km, along-track)

Engineered Features

Auroral Zone Indicators:

    in_auroral_zone = boolean(60° < |mag_lat| < 75°)
    in_polar_cap = boolean(|mag_lat| > 75°)
    auroral_oval_crossing = boolean(satellite crossing 65-70° mag_lat)
    MLT_sector = binned MLT (dawn/noon/dusk/midnight)

Magnetic Perturbations:

    dB_horizontal = sqrt(dB_north² + dB_east²) - horizontal perturbation from model
    dB_vertical = dB_center - vertical perturbation
    dB_total = |B_observed - B_model| (model = IGRF/Swarm L2 baseline)
    dB_dt_along_track = (B[t] - B[t-1s]) / dt (time derivative along orbit)
    dB_ds_spatial = spatial gradient along orbit track

Field-Aligned Currents:

    FAC_max_auroral = max(|FAC_density|) during auroral zone crossing
    FAC_total_auroral = ∫|FAC_density| ds during crossing
    FAC_region1_indicator = FAC pattern consistent with Region 1 currents
    FAC_region2_indicator = FAC pattern consistent with Region 2 currents
    FAC_variability = std(FAC_density) during crossing

Conjugate Observations (Multi-Sat):

    dB_AB_difference = |B_swarmA - B_swarmB| (cross-track gradient)
    dB_AC_difference = |B_swarmA - B_swarmC| (along-track gradient)
    current_sheet_thickness_proxy = separation_AB / dB_AB_difference

Ground Mapping (Approximate):

    ground_conjugate_lat = map satellite position to ground along field line
    ground_conjugate_lon
    ground_perturbation_proxy = dB_horizontal × mapping_factor

Statistical Features:

    dB_max_pass = max(dB_horizontal) during orbital pass over region
    dB_mean_auroral_zone = mean(dB) during 60-75° mag_lat crossing
    perturbation_duration = time_duration(dB_horizontal > threshold)
    high_latitude_activity_index = integral of dB² over high-lat pass

Ground: SuperMAG
Raw Features

Station Measurements (per station):

    H_component (nT, horizontal north component)
    D_component (nT, horizontal east component)
    Z_component (nT, vertical down component)
    geographic_latitude (degrees)
    geographic_longitude (degrees)
    magnetic_latitude (degrees)
    magnetic_local_time (hours)

SuperMAG Indices:

    SMR_index (nT, polar cap index - measures transpolar current)
    SME_index (nT, equivalent to AE - auroral electrojet)
    SML_index (nT, equivalent to AL - westward electrojet, negative)
    SMU_index (nT, equivalent to AU - eastward electrojet, positive)

Station Network:

    n_stations_available (count of reporting stations)
    station_coverage_score (quality metric)

Engineered Features

Single Station Features:

    dH_dt = (H[t] - H[t-1min]) / 1min (GIC proxy!)
    dD_dt = (D[t] - D[t-1min]) / 1min
    dH_dt_max_10min = max(|dH_dt|) in past 10 minutes
    dH_horizontal = sqrt(dH² + dD²)
    dB_dt_total = sqrt((dH/dt)² + (dD/dt)² + (dZ/dt)²)

Regional Aggregation:

    dB_dt_max_auroral_zone = max(dB_dt) across all stations in 60-75° mag_lat
    dB_dt_mean_auroral_zone = mean(dB_dt) across auroral stations
    dB_dt_std_auroral_zone = std(dB_dt) across auroral stations
    n_stations_active = count(stations with dB_dt > threshold)
    auroral_extent_proxy = max_lat(active) - min_lat(active)

SuperMAG Index Features:

    SME_current = current SME value
    SME_max_3h = max(SME, t-3h to t)
    SME_avg_3h = mean(SME, t-3h to t)
    dSME_dt_1h = (SME[t] - SME[t-1h]) / 1h
    dSME_dt_15m = (SME[t] - SME[t-15m]) / 15m
    SML_current = current SML (negative bay magnitude)
    SML_min_3h = min(SML, t-3h to t) - most negative value
    SMU_current = current SMU
    SMU_max_3h = max(SMU, t-3h to t)
    SMR_current = polar cap index current value
    SMR_avg_1h = mean(SMR, t-1h to t)

Substorm Identification from Indices:

    SME_rapid_increase = boolean(dSME_dt_15m > threshold)
    westward_surge_indicator = boolean(SML decreasing rapidly)
    bay_magnitude = |SML_current - SML_quiet_baseline|
    onset_candidate = boolean(SME jumps >100 nT in 10 min)

Local Time Distribution:

    peak_activity_MLT = MLT of station with max dB_dt
    midnight_sector_active = boolean(max activity in 22-02 MLT)
    activity_MLT_spread = MLT_range of active stations

**Spatial Patterns:**

    east_west_gradient = difference in dB_dt between dusk and dawn stations
    north_south_gradient = difference in activity between high-lat and mid-lat
    activity_centroid_lat = weighted average latitude of active stations
    activity_centroid_MLT = weighted average MLT of active stations

Temporal Evolution:

    dB_dt_acceleration = d²H/dt² (second derivative)
    perturbation_growth_rate = (dB_dt_current - dB_dt_30min_ago) / 30min
    sustained_activity_duration = time_duration(dB_dt > 300 nT/min)
    activity_onset_time = t - time_when(dB_dt first exceeded threshold)

Historical Context (Per Station):

    H_baseline_24h = mean(H, t-24h to t) during quiet periods
    dH_from_baseline = H[t] - H_baseline_24h
    H_storm_deviation = (H[t] - H_quiet_baseline) / H_std_typical

Network-Wide Statistics:

    fraction_stations_active = n_active / n_total
    activity_coherence = correlation(dB_dt across nearby stations)
    simultaneous_onset_score = correlation of dB_dt_peaks across stations
    azimuthal_coherence = correlation along same magnetic latitude

GIC-Relevant Features:

    dB_dt_99th_percentile = 99th percentile of dB_dt across all stations
    extreme_dB_dt_count = count(stations with dB_dt > 1000 nT/min)
    max_dB_dt_station_id = ID of station with maximum dB_dt
    max_dB_dt_location_lat = latitude of max dB_dt
    max_dB_dt_location_lon = longitude of max dB_dt

Frequency Content (if high-cadence data):

    Pi2_power = spectral power in 40-150 second band (substorm indicator)
    Pc5_power = spectral power in 150-600 second band (ULF waves)
    high_freq_content = power above 0.1 Hz (impulsive activity)


Category 4: Interaction Terms (Cross-Dataset Features)
Solar Wind × Magnetosphere State

Loading × State:

    precondition_trigger_newell = newell_integral_48h × dNewell_dt_1h
    precondition_trigger_epsilon = epsilon_integral_6h × dNewell_dt_30m
    loading_rate_product = newell_slope_48h × dNewell_dt_1h
    energy_trigger_composite = newell_integral_24h × dBz_L1_dt_30m × (1 + Bz_depression_geo)

Driver × GEO Response:

    newell_times_stretch = newell_current × tail_stretch_index
    Bz_L1_times_Bz_depression_GEO = Bz_gsm_L1 × Bz_depression_fraction
    pressure_times_stretch = ram_pressure × tail_stretch_index
    velocity_times_depression = v_sw × Bz_depression_absolute

Threshold Interactions:

    primed_magnetosphere = boolean(newell_integral_48h > threshold_1 AND Bz_depression > 0.3)
    triggered_primed_system = boolean(primed_magnetosphere AND dNewell_dt_1h > threshold_2)
    critical_loading = boolean(newell_integral_24h > threshold AND Bz_min_2h_GEO < threshold)

Solar Wind × Ground Response

Driver × Ground Activity:

    newell_times_SME = newell_current × SME_current
    Bz_times_dB_dt_ground = |Bz_gsm_L1| × dB_dt_max_auroral_zone (if Bz < 0)
    coupling_times_activity = epsilon_parameter × SME_current
    pressure_pulse_times_ground = dP_dt_30m × dB_dt_mean_auroral_zone

Preconditioning × Response:

    stored_energy_times_ground_activity = newell_integral_24h × SME_max_3h
    loading_times_current_activity = newell_slope_24h × SME_current

GEO × Ground Coherence

Magnetospheric-Ionospheric Coupling:

    GEO_ground_correlation = correlation(dBz_geo_dt, dB_dt_auroral_mean) over past hour
    dipolarization_ground_response = Bz_increase_5m_GEO × dB_dt_max_ground (if simultaneous)
    stretch_ground_quiet = tail_stretch_index × (1 - SME_normalized) - growth phase indicator
    GEO_dipolarization_with_ground_onset = boolean(rapid_Bz_increase_GEO AND SME_rapid_increase)

Local Time Matching:

    GEO_MLT_matches_peak_activity = boolean(|MLT_GEO - peak_activity_MLT_ground| < 2 hours)
    midnight_sector_alignment = boolean(midnight_sector_stretched_GEO AND midnight_sector_active_ground)

Multi-Altitude Consistency

LEO × Ground:

    Swarm_SuperMAG_correlation = correlation(dB_horizontal_Swarm, dB_dt_conjugate_station)
    FAC_ground_consistency = FAC_max_auroral × dB_dt_mean_auroral_zone
    overhead_consistency_score = similarity(Swarm_perturbation, ground_perturbation) when overhead

GEO × LEO:

    magnetosphere_ionosphere_coupling = Bz_depression_GEO × FAC_max_auroral
    particle_precipitation_proxy = electron_flux_GEO × FAC_total_auroral

Timing Relationships

Propagation Delays:

    L1_to_GEO_appropriate_delay = feature_L1[t - τ_prop] × feature_GEO[t] where τ_prop ≈ 30-60 min
    newell_delayed_40min = newell_coupling[t-40min] - accounts for solar wind transit
    GEO_to_ground_delay = feature_GEO[t] × feature_ground[t + 0-10min] - nearly simultaneous
    ⋆⋆⋆ulative_delayed_coupling = ∫(newell_coupling[t-τ] × decay_function(τ)) dτ

Cross-Dataset Trigger Detection:

    L1_trigger_GEO_primed = boolean(dNewell_dt_1h > threshold AND at_Bz_minimum_GEO)
    GEO_onset_ground_response = boolean(rapid_Bz_increase_GEO[t] AND SME_rapid_increase[t+5min])


Category 5: Temporal Context
Substorm History

Recent Activity:

    time_since_last_substorm = t - t_last_substorm (from catalog or detected)
    substorm_count_24h = count(substorms in past 24h)
    substorm_count_12h = count(substorms in past 12h)
    substorm_count_6h = count(substorms in past 6h)
    last_substorm_intensity = max_dB_dt from last substorm
    time_since_last_major_substorm = t - t_last_substorm where max_dB_dt > 1000 nT/min

Clustering Indicators:

    in_substorm_cluster = boolean(substorm_count_6h >= 3)
    cluster_intensity_trend = linear_slope(substorm_intensities in cluster)
    inter_substorm_interval = t_last_substorm - t_previous_substorm
    interval_shortening = boolean(current_interval < previous_interval)

Refractory Period:

    magnetosphere_recovery_state = time_since_last_substorm / typical_recovery_time
    insufficient_recovery = boolean(time_since_last_substorm < 30 minutes)
    partial_recovery = boolean(30 min < time_since_last_substorm < 2 hours)
    full_recovery = boolean(time_since_last_substorm > 3 hours)

Cumulative Activity:

    cumulative_SME_24h = ∫(SME) dt over past 24h
    ⋆⋆⋆ulative_dB_dt_24h = ∫(dB_dt_max) dt over past 24h
    total_energy_dissipated_24h = ∫(epsilon_parameter) dt over past 24h

Storm Phase Context

Geomagnetic Storm State:

    dst_current = Dst index at time t (if available)
    dst_min_24h = min(Dst, t-24h to t)
    dst_rate = dDst/dt
    storm_phase = classification (quiet/initial/main/recovery) based on Dst
    in_storm_main_phase = boolean(dst_current < -50 nT AND dst_rate < 0)
    in_storm_recovery = boolean(dst_current < -50 nT AND dst_rate > 0)
    storm_intensity = |dst_min_24h| / 100 nT (normalized)

Storm Progress:

    time_since_storm_onset = t - time_when(Dst first < -50 nT)
    time_since_dst_minimum = t - time_of(dst_min_24h)
    recovery_progress = (dst_current - dst_min) / (0 - dst_min)

Kp Context:

    kp_current = current Kp index
    kp_max_24h = max(Kp, t-24h to t)
    kp_avg_24h = mean(Kp, t-24h to t)
    elevated_kp_duration = time_duration(Kp >= 5 in past 24h)
    kp_rising = boolean(kp_current > kp_3h_ago)

Temporal Cycles

Universal Time Dependence:

    UT_hour = hour of day (0-23)
    UT_hour_sin = sin(2π × UT_hour / 24) - cyclic encoding
    UT_hour_cos = cos(2π × UT_hour / 24) - cyclic encoding
    Russell-McPherron_effect = dipole_tilt_angle × coupling_efficiency (UT-dependent)
    equinox_season = boolean(near spring or fall equinox) - semiannual variation
    dipole_tilt_angle = angle between Earth's dipole and GSM Z-axis

Seasonal Variations:

    day_of_year = 1-365/366
    season_sin = sin(2π × day_of_year / 365)
    season_cos = cos(2π × day_of_year / 365)
    northern_winter = boolean(Nov-Feb)
    southern_winter = boolean(May-Aug)
    equinox_period = boolean(Mar-Apr or Sep-Oct)
    solar_illumination_factor = function of season and UT

Solar Cycle:

    F107_index = solar 10.7 cm radio flux (proxy for solar activity)
    F107_81day_avg = 81-day running average of F10.7
    solar_cycle_phase = position in 11-year cycle (if long training period)
    sunspot_number_smoothed = smoothed sunspot number

Trend Analysis

Short-Term Trends:

    newell_trend_6h = is newell_coupling increasing/decreasing in past 6h?
    SME_trend_3h = slope of SME over past 3h
    Bz_trend_3h = slope of Bz_gsm over past 3h
    activity_accelerating = boolean(d²(SME)/dt² > 0)

Medium-Term Context:

    active_period_24h = boolean(newell_avg_24h > threshold)
    sustained_driving_12h = boolean(newell_avg_12h > threshold)
    quiet_period_24h = boolean(newell_avg_24h < threshold AND SME_avg_24h < threshold)

Change Points:

    regime_change_6h = boolean(significant change in mean/variance of parameters)
    sudden_enhancement = boolean(newell jumped >50% in past 1h)
    activity_transition = detected transition from quiet to active

Forecasting Windows

Look-Ahead Context (for target definition):

    target_dB_dt_max_next_30m = max(dB_dt) in window [t, t+30min]
    target_dB_dt_max_next_1h = max(dB_dt) in window [t, t+1h]
    target_dB_dt_max_next_2h = max(dB_dt) in window [t, t+2h]
    target_substorm_occurrence_30m = boolean(substorm onset in [t, t+30min])
    target_substorm_occurrence_1h = boolean(substorm onset in [t, t+1h])
    target_substorm_intensity_class = class of strongest substorm in [t, t+2h]

Lead Time Features:

    time_to_next_substorm = t_next_substorm - t (for regression)
    imminence_score = 1 / (time_to_next_substorm + 1) - higher when onset near

Data Quality & Availability

Completeness:

    L1_data_quality = fraction of expected L1 measurements present
    GEO_data_quality = data quality flag from GOES
    ground_station_coverage = n_stations_reporting / n_stations_total
    Swarm_coverage_auroral = boolean(Swarm passed through auroral zone in past hour)

Uncertainty Indicators:

    L1_position_uncertainty = is L1 monitor in good position? (libration point stability)
    multi_spacecraft_agreement = std([measurements from multiple spacecraft])
    bow_shock_crossing = boolean(L1 outside magnetosphere temporarily)

Operational Context

Forecast History (if in operational setting):

    previous_forecast_1h = model's forecast from 1h ago
    previous_forecast_error_1h = actual - previous_forecast_1h
    forecast_persistence = current_conditions assumed to persist
    model_confidence_previous = model's confidence score from previous run

Alert State:

    current_alert_level = any active space weather alerts
    watches_in_effect = watches issued for near-term activity
    time_in_current_alert = duration of current alert state

# Potential Model Training
## Temporal Cross-Validation
```
# Don't use random train/test split!
# Use time-based split to avoid data leakage

from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
```
## Target Variable Options
```
# Option 1: Regression - predict magnitude
target = 'dB_dt_max_next_2h'  # continuous value

# Option 2: Classification - predict intensity class
target = 'substorm_intensity_class'  # 0-4

# Option 3: Binary - predict occurrence
target = 'substorm_occurs_next_1h'  # 0 or 1

# Option 4: Multi-output - predict multiple
targets = ['dB_dt_max_next_1h', 
           'dB_dt_max_next_2h',
           'onset_location_MLT']
```

## Handling Class Imbalance




