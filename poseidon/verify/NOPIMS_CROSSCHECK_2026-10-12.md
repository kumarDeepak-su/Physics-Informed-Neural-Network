# Cross-check of the mirror well files against the NOPIMS package listings

Date: 2026-10-12. Method: the NOPIMS content-delivery index of each well package
(https://dnxxuwuw8tglo.cloudfront.net/Wells/<ITEM>.txt, a 7-Zip listing with the byte size of every member and the
MD5 of the zip) was opened in a normal Chrome session (the CDN serves these only to browsers) and every file used from
the ConocoPhillips-supported Google Drive mirror (wells/<Well>/..., 75 files) was looked up by name; its byte size on
disk was compared with the listed member size. The zips themselves (0.4 to 8.2 GB each) were not downloaded, so this
is a name-and-size check, not a content hash.

Result: 75 of 75 files match in name and byte size; 0 size mismatches; 0 not located.

NOPIMS items consulted (zip MD5 from the index where captured):

- D00014520: 74314F62EF5B9CB0A310EEF60EAF45AB (16 of our files listed)
- D00015055: 9752ABE06E4BA498A66271A92F76DFDC (1 of our files listed)
- D00015058: 806793ED2B171AA13262252A4E7CF17E (15 of our files listed)
- D00017344: 234207C77D1612B3F7831CCAB1FF283F (1 of our files listed)
- P00579942: 144AA3F0AAB026AFB5E4FAFE2F7E21A0 (2 of our files listed)
- P00596291: E87CDF37D5747204733DA7B464FEDD25 (14 of our files listed)
- P00636376: MD5 not recorded (4 of our files listed)
- P00688589: MD5 not recorded (4 of our files listed)
- P00707041: MD5 not recorded (9 of our files listed)
- P00707278: MD5 not recorded (11 of our files listed)

| Well | Mirror file | Bytes on disk | NOPIMS item(s) with the same size | Status |
|---|---|---|---|---|
| Kronos 1 | VSP/CP_Kronos1_Drift_and_Sonic_Adjustment_Report.xls | 106496 | P00707278 | match |
| Kronos 1 | VSP/CP_Kronos1_Logs_depth.las | 5418581 | P00707278 | match |
| Kronos 1 | VSP/CP_Kronos1_Logs_time.las | 851161 | P00707278 | match |
| Kronos 1 | VSP/CP_Kronos1_Synthetics.las | 851592 | P00707278 | match |
| Kronos 1 | VSP/CP_Kronos1_Time_to_Depth_Report.xls | 89088 | P00707278 | match |
| Kronos 1 | VSP/CP_Kronos1_VSP_Report.pdf | 4938462 | P00707278 | match |
| Kronos 1 | VSP/CP_Kronos1_Velocity_Report.xls | 92160 | P00707278 | match |
| Kronos 1 | VSP/CP_Kronos1_Well_Seismic_Report.xls | 120320 | P00707278 | match |
| Kronos 1 | VSP/CP_Kronos1_corstk.sgy | 166040 | P00707278 | match |
| Kronos 1 | VSP/CP_Kronos1_corstkf.sgy | 166040 | P00707278 | match |
| Kronos 1 | VSP/VSI_003_Stack.txt | 28749 | P00707278 | match |
| Kronos 1 | wireline/GR to Surface VSIT_327LUC.las | 1674832 | P00636376 | match |
| Kronos 1 | wireline/Run 1 CMR-LDS-CNT-GR Main Pass _268PUC.las | 1143099 | P00636376 | match |
| Kronos 1 | wireline/Run 2 OBMI-SS-PPC-GR Main Pass 112PUC.las | 1614659 | P00636376 | match |
| Kronos 1 | wireline/Run 3 AIT-ECS-HNGS-GR Main Pass 096PUC.las | 2000729 | P00636376 | match |
| Pharos 1 | VSP/COP-Pharos-1_QC_VSP_Corridor_Stack.sgy | 206040 | D00015058 | match |
| Pharos 1 | VSP/COP-Pharos-1_QC_VSP_Corridor_Stack.txt | 140147 | D00015058 | match |
| Pharos 1 | VSP/COP-Pharos-1_QC_VSP_TimeDepthListing.xlsx | 25923 | D00015058 | match |
| Pharos 1 | VSP/COP_Pharos-1_QC_VSP_GR_log.las | 993247 | D00015058 | match |
| Pharos 1 | VSP/COP_Pharos-1_QC_VSP_Report.pdf | 13020234 | D00015058 | match |
| Pharos 1 | VSP/COP_Pharos-1_VSP_Processing_Report_short.pdf | 15545190 | D00015058 | match |
| Pharos 1 | VSP/COP_Pharos-1_corstk.sgy | 206040 | D00015058 | match |
| Pharos 1 | VSP/COP_Pharos-1_logs_depth.las | 5317220 | D00015058 | match |
| Pharos 1 | VSP/COP_Pharos-1_logs_time.las | 837760 | D00015058 | match |
| Pharos 1 | VSP/COP_Pharos-1_synthetics.las | 1344196 | D00015058 | match |
| Pharos 1 | survey/Pharos-1_Definitive Survey Report_TD 5220m.xlsx | 43146 | D00015055 | match |
| Pharos 1 | wireline/COP_Pharos-1_S1R2_NGIT_PPC_MAST_HNGS_MainPass.las | 12243947 | D00015058 | match |
| Pharos 1 | wireline/COP_Pharos-1_S2R1_Main_Pass.las | 6455686 | D00015058 | match |
| Pharos 1 | wireline/COP_Pharos-1_S2R2_Main_Pass.las | 4296073 | D00015058 | match |
| Pharos 1 | wireline/COP_Pharos-1_S2R2_SonicScanner.las | 154898 | D00015058 | match |
| Pharos 1 | wireline/COP_Pharos-1_SonicScanner_L0_PnS_SLB.las | 616187 | D00015058 | match |
| Poseidon 2 | VSP/CP_Poseidon2_Drift_and_Sonic_Adjustment_Report.xls | 100864 | P00707041 | match |
| Poseidon 2 | VSP/CP_Poseidon2_VSP_Report.pdf | 3974288 | P00707041 | match |
| Poseidon 2 | VSP/CP_Poseidon2_Velocity_Report.xls | 90624 | P00707041 | match |
| Poseidon 2 | VSP/CP_Poseidon2_Well_Seismic_Report.xls | 111104 | P00707041 | match |
| Poseidon 2 | VSP/CP_Poseidon2_corstk.sgy | 206040 | P00707041 | match |
| Poseidon 2 | VSP/CP_Poseidon2_corstk.txt | 49043 | P00707041 | match |
| Poseidon 2 | VSP/CP_Poseidon2_logs_depth.las | 4964848 | P00707041 | match |
| Poseidon 2 | VSP/CP_Poseidon2_logs_time.las | 812688 | P00707041 | match |
| Poseidon 2 | VSP/CP_Poseidon2_synthetics.las | 821046 | P00707041 | match |
| Poseidon 2 | wireline/CoP Poseidon-2 SonicScanner 6in DT P&S.LAS | 406706 | P00688589 | match |
| Poseidon 2 | wireline/ConocoPhillips_Poseidon-2_Suite2_Run6_LDS-CNL-GR_Merged_ML.las | 488164 | P00688589 | match |
| Poseidon 2 | wireline/ConocoPhillips_Poseidon-2_Suite2_Run7_AIT-ECS-LDL-CNL-GR_Merged_ML.las | 1839484 | P00688589 | match |
| Poseidon 2 | wireline/ConocoPhillips_Poseidon-2_Suite2_Run8_VSP_VSI4-GR_Data.xls | 32256 | P00688589 | match |
| Proteus 1 | LWD/ConocoPhillips_Proteus1_Final.las | 5013435 | D00014520 | match |
| Proteus 1 | LWD/ConocoPhillips_Proteus1_ST1_Fina.las | 9678273 | D00014520 | match |
| Proteus 1 | LWD/ConocoPhillips_Proteus1_ST2_Final.las | 10348738 | D00014520 | match |
| Proteus 1 | VSP/COP_Proteus-1ST2_A1_Well_Seismic_Report_Final.xls | 98816 | D00014520 | match |
| Proteus 1 | VSP/COP_Proteus-1ST2_A2_Drift_Report_Final.xls | 93184 | D00014520 | match |
| Proteus 1 | VSP/COP_Proteus-1ST2_A3_Velocity_Report_Final.xls | 89088 | D00014520 | match |
| Proteus 1 | VSP/COP_Proteus-1ST2_VSP_Processing_Report.pdf | 13136316 | D00014520 | match |
| Proteus 1 | VSP/COP_Proteus-1ST2_corstk.txt | 50902 | D00014520 | match |
| Proteus 1 | VSP/COP_Proteus-1ST2_logs_depth_MD.las | 7058095 | D00014520 | match |
| Proteus 1 | VSP/COP_Proteus-1ST2_logs_depth_TVDSS.las | 5377523 | D00014520 | match |
| Proteus 1 | VSP/COP_Proteus-1ST2_logs_time.las | 793306 | D00014520 | match |
| Proteus 1 | VSP/COP_Proteus-1ST2_synthetics.las | 1344445 | D00014520 | match |
| Proteus 1 | VSP/COP_Proteus-1ST2_synthetics_well_wavelets.ascii | 19195 | D00014520 | match |
| Proteus 1 | survey/Proteus Surveys.xlsx | 88747 | D00017344 | match |
| Proteus 1 | survey/Proteus-1 ST2_Definitive Survey Report_TD 5250m.xlsx | 48546 | D00014520 | match |
| Proteus 1 | wireline/COP_Proteus-1ST2_CH_QSLT_PnS_Processed_PTS.las | 141821 | D00014520 | match |
| Proteus 1 | wireline/COP_Proteus-1ST2_OH_QSLT_PnS_Processed_PTS.las | 159410 | D00014520 | match |
| Torosa 1 | LWD/Torosa-1_REC_DDS_550-4685mMD.las | 5571224 | P00579942, P00596291 | match |
| Torosa 1 | LWD/Torosa-1_REC_DGR-EWR-NUKE-BAT_550-4685mMD.las | 10246168 | P00596291 | match |
| Torosa 1 | VSP/TZV_DEPTH_CHECKSHOT.PDF | 126214 | P00596291 | match |
| Torosa 1 | VSP/TZV_DEPTH_CHECKSHOT.XLS | 49664 | P00596291 | match |
| Torosa 1 | VSP/TZV_DEPTH_CS_CALIBRATED_LOGS.LAS | 3446624 | P00596291 | match |
| Torosa 1 | VSP/TZV_DEPTH_GEOMETRY.XLS | 27136 | P00596291 | match |
| Torosa 1 | VSP/TZV_TIME_INDX.XLS | 260096 | P00596291 | match |
| Torosa 1 | VSP/TZV_TIME_SYNTHETICS_1.LAS | 510646 | P00596291 | match |
| Torosa 1 | VSP/VSPZO_COMPUTED_DCU_ZPH_90.SEGY | 2432400 | P00596291 | match |
| Torosa 1 | VSP/VSPZO_COMPUTED_INF.ASC | 3695 | P00596291 | match |
| Torosa 1 | survey/ENO407785_Torosa_1_Surveys_4685mMD.txt | 15103 | P00579942 | match |
| Torosa 1 | survey/Inclination_Survey.txt | 2616 | P00596291 | match |
| Torosa 1 | survey/Torosa-1_Surveys_4685mMD.txt | 11791 | P00596291 | match |
| Torosa 1 | wireline/Torosa-1_Nan Hai IV_Suite 1 Run 1_ Main Log_HRLA_TLD_MCFL_CNL_4680m-3526m.las | 3123731 | P00596291 | match |
