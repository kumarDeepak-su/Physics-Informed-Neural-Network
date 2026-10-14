# Poseidon 3D (Browse Basin, NW Shelf Australia): data provenance

Started 2026-09-16. Purpose: field-data section of the TGRS revision TGRS-2026-04997
(reviewer comment 6), replacing F3 (timing-limited) and IBDP (single-well, no power).

## Seismic
- Source: AWS Registry of Open Data, bucket `s3://tgs-opendata-poseidon/` (us-west-2), anonymous read.
  Listed stores: `near_stack.mdio/`, `mid_stack.mdio/`, `far_stack.mdio/`, `full_stack_agc.mdio/`.
  The full stack carries AGC (stated by TGS/TerraNubis); the angle stacks do not. Angle stacks
  created by ConocoPhillips; original seismic: Geoscience Australia.
- Licence: CC BY 4.0 (AWS registry listing); TerraNubis states CC BY-SA 3.0 AU for the OpendTect package
  and "Using the data and value-added products in this project in publications is permitted".
- Format: MDIO 1.0.0a1 = zarr v2 group (`.zmetadata` consolidated), created 2024-12-11T16:50:58Z,
  arrays: seismic (float32, chunks 128x128x128, blosc-zstd), headers (inline, crossline,
  scalar_apply_coords, cdp-x, cdp-y), cdp-x/cdp-y (float64 m), inline, crossline, time (ms), trace_mask.
- near_stack geometry (read 2026-09-16): inline 983..4419 step 1 (3437), crossline 504..5556 step 1 (5053),
  time 0..6000 ms at 4 ms (1501 samples), trace_mask 100 % live (17,367,161 traces).
- Bin -> coordinate (header ints, scalar -100 => divide by 100 for metres; CRS per TGS: GDA94 / MGA zone 51):
  x_raw = 1205.226958*il - 957.555533*xl + 40911129.891
  y_raw = 1436.333333*il + 803.484399*xl + 842942630.308
  (least squares on 9 header samples, residual <= 0.5 raw units = 5 mm);
  |d/dil| = 18.75 m, |d/dxl| = 12.5 m, consistent with the published 18.75 x 12.5 m bins.
- Nothing has been modified; only read.

## Wells (to be fetched from NOPIMS, Geoscience Australia)
Kronos 1, Pharos 1, Poseidon 1, Poseidon 2, Poseidon North 1, Proteus 1 (+ ST1, ST2), Torosa 1.
Required per well: LAS with sonic + density, checkshot/VSP time-depth, deviation survey, WCR.
NOTE: the TerraNubis project's time-depth models are constant-velocity (2500 m/s) placeholders, never use them.

## Reports
Poseidon 3D Processing Report and Interpretation Report (ConocoPhillips) - to be obtained (TerraNubis / NOPIMS).

## Facts from the ConocoPhillips / CGGVeritas reports (downloaded 2026-09-16 from dgbes.com, SHA-256 in reports/SHA256SUMS.txt)
- Processing report (CGGVeritas project 501p1fi, Perth): acquisition 2010, 10 streamers, 12.5 m groups,
  2 ms field / 4 ms processing, record length 7000 ms, datum WGS84, bins 12.5 x 18.75 m after trace drop.
  Final sequence: deterministic zero-phasing, **polarity reversed so that a trough = increase in impedance
  (SEG reverse)**, SRME + Radon demultiple, Kirchhoff PSTM (dip 60 deg, 4 km half aperture), phase-only Q (Q=135),
  final angle stacks **full 6-42, near 6-18, mid 18-30, far 30-42 deg**, then post-stack: amplitude-only Q,
  time-variant scaling, diffracted-multiple and random-noise attenuation -> "Final AVA stacks".
  The fast-track full stack additionally had an exponential gain (5 dB/s) and a double-window AGC (1000/400 ms);
  the S3 store `full_stack_agc.mdio` is the AGC product and must not be used for amplitude inversion.
- Interpretation report: "4 wells located within the area of the Poseidon 3D: Poseidon 1, Poseidon 2, Kronos 1,
  Torosa 1" (2012); synthetics from sonic + density + checkshot/VSP "where available"; water bottom 420-1090 m;
  Top Jamieson = strong impedance decrease, basin-wide marker; Near Top Plover = impedance increase.

## Well positions (NOPTA Petroleum_Wells FeatureServer, queried 2026-09-16; lat/lon -> EPSG:28351 -> bins)
Map registration check: survey centre bin (2701,3030) -> lon 122.192, lat -13.634 (TerraNubis map centre 122.2, -13.63).
Inside the survey (13 boreholes, 10 distinct surface locations): Boreas 1, Crown 1 (+ST1), Kronos 1, Pharos 1,
Poseidon 1, Poseidon 2, Poseidon North 1, Proteus 1 (+ST1, ST2), Torosa 1, Zephyros 1. Outside: Torosa 2-6.
Bins: Kronos 1 (2410, 2670); Poseidon 1 (3180, 2337); Poseidon 2 (2856, 2284); Poseidon North 1 (3525, 2400);
Proteus 1 (2766, 1493); Pharos 1 (3255, 1392); Boreas 1 (2992, 2203); Zephyros 1 (1943, 2856); Crown 1 (4068, 1275);
Torosa 1 (1194, 2513). File: wells/wells_positions.json.

## NOPIMS (Geoscience Australia / NOPTA) well data items, listed 2026-09-16
- Public OData: https://services.neats.nopta.gov.au/odata/v1/public/nopims/well/PublicNopimsWells (no login); item
  lists in wells/nopims_reports_for_selection.json and wells/nopims_items_table.csv (104 items for the 13 boreholes).
- File delivery is via https://dnxxuwuw8tglo.cloudfront.net/Wells/<ITEM>.zip with an index <ITEM>.txt that carries the
  zip MD5; the CDN answers real browsers only (scripted clients get an AWS WAF challenge, HTTP 202/403), so the zips
  were NOT fetched by script. Index read in Chrome for Poseidon 1 WCR-Basic P00685602.zip (2,563,458,780 bytes,
  MD5 5281795BB7C2482C4570F51006752346): contains LWD LAS (GR-RES-DEN-POR, sonicVISION runs 3-4), wireline suite 1/2
  DLIS + LAS (LDS-CNL-GR density/neutron, Sonic Scanner P&S S2R3), CMR, mudlog data and the WCR Volume 1 PDF.
- The same well files are mirrored in the public Google Drive folder that TerraNubis lists as the
  "External site supported by ConocoPhillips" (goo.gl/xMzKY9 -> drive folder 0B7brcf-eGK8CRUhfRW9rSG91bW8 ->
  "Poseidon Seismic and well data NW Shelf Australia" 0B7brcf-eGK8Cbk9ueHA0QUU4Zjg -> "All wells data logs and reports"
  1s7y9mFMZN9YgsqXSw2WnyOYrNwqUoEbc). Files taken from there are hashed in wells/drive_download_record.json; a
  later cross-check against the NOPIMS zips (browser download by the author) is recommended before submission.

## Well files actually obtained (2026-09-16, 21:40-23:00), all from the public ConocoPhillips-supported Google Drive
Per-batch records with SHA-256: wells/selection_batch*_record.json, wells/selection_*_curl_record.json
(gdown for batches 1-3, curl for 4-9; both read the public share, no login). Folders: wells/<Well_Name>/{VSP,wireline,LWD,survey,reports}.
- Pharos 1: Schlumberger VSP processing set (TimeDepthListing.xlsx, logs_depth/logs_time/synthetics LAS, corridor stacks,
  QC report), Sonic Scanner and suite-2 field LAS, definitive survey xlsx.
- Kronos 1: VSP set (Logs_depth/time, Time_to_Depth/Velocity/Well_Seismic reports .xls, corridor stacks, VSP report),
  wireline runs 1-3 LAS (LDS density run 1, Sonic Scanner run 2), VSI GR-to-surface LAS.
- Poseidon 1: wireline suites 1-2 LAS (LDS density, Sonic Scanner P&S), VSP corridor stacks (SEG-Y, no digital listing);
  time-depth parsed from WCR Vol 2 Appendix 8 Table 33 -> reports/Poseidon1_VSP_Table33_time_depth.csv (146 levels).
- Poseidon 2: VSP set (logs_depth/time/synthetics, Drift/Velocity/Well_Seismic .xls, corridor stack, VSP report),
  Sonic Scanner LAS, suite-2 LDS-CNL-GR merged LAS, AIT-ECS merged LAS, VSI4 VSP data xls.
- Poseidon North 1: LWD VISION/sonicVISION LAS (no wireline density run listed), wireline S1R1 speed-corrected LAS,
  Sonic Scanner L0 LAS, definitive survey; NO VSP in the mirror or in the basic WCR (RT-MSL 22 m, water depth 486 m).
- Proteus 1 (ST2 is the logged hole): LWD final LAS x3, sonic LWD LAS, ZVSP set (A1-A3 listings .xls, logs_depth_MD/TVDSS,
  logs_time, synthetics, corridor stack listing), QSLT PnS processed LAS, surveys xlsx, VSP processing report.
- Torosa 1 (Woodside 2006): wireline HRLA-TLD-MCFL-CNL LAS, LWD DGR-EWR-NUKE-BAT and DDS LAS, surveys, and the
  Schlumberger VSP set TZV_DEPTH_CHECKSHOT.XLS + TZV_DEPTH_CS_CALIBRATED_LOGS.LAS (TVD & TWT from MSL, calibrated DT,
  density, AI; density above ~3577 m MD is modelled, measured RHOZ only 3577-4666 m), synthetics, zero-offset VSP SEG-Y.
- Spliced_occam/: Occam Technology spliced, 0.5 m decimated LAS for Boreas 1, Kronos 1, Pharos 1, Poseidon 1, Poseidon 2,
  Poseidon North 1, Proteus 1, Torosa 1 (CC-BY per their README; deviation data embedded). Cross-check only.
- Velocity/: NOPIMS item D00013425 (Poseidon 3D MSS velocity, 6.27 GB zip) contains per-inline 2D migration velocity
  files (.westernb); download stopped at 1.5 GB to free bandwidth; the index text is kept. Fetch later if an LFM needs it.
- Not obtainable without a browser: Boreas 1, Crown 1, Zephyros 1 NOPIMS packages (Boreas has only the Occam spliced LAS here).

## Location
Moved from `2D_ADMM_PINN_MANUSCRIPT/Poseidon3D/` to `2D_ADMM_PINN_MANUSCRIPT/DATA_POST_STACK/Poseidon3D/` on
2026-09-16 23:32 (plain rename, no file changed; block record and download records keep relative paths).
Note: wells/info/ held 75 CloudFront "403 ERROR" pages saved in place of the NOPIMS index files (scripted
fetch blocked); they were deleted on 2026-09-16 23:35. Only urls.txt (the index URLs) remains.

## Cross-check against NOPIMS (2026-10-12)
All 75 of 75 mirror well files used in the paper match the NOPIMS package member listings in name and byte size
(items D00015055/56/57/58, P00636376, P00707278, P00688589, P00707041, D00014520, D00017344, P00579942, P00596291);
see provenance/NOPIMS_CROSSCHECK_2026-10-12.md. The zips were not downloaded; the check is by name and size, not by hash.
