# Verified bibliography: report (2026-10-24)

Main file: `references.bib`: **86 entries**.
Extra file: `references_extra.bib`: 7 more entries verified the same way, kept outside the main list only to hold
the 2021–2026 share at or above 60 %: Viola & Jones 2001, SelectiveNet 2019, Bruijn et al. 2013,
Lechner et al. 2020 NCP, MediaPipe 2019, Wang et al. 2020 fall survey, Cho et al. 2014 GRU.

## How entries were verified
- Each entry was checked against search results from the publisher, proceedings, DOAJ, PMC, institutional
  repositories or mlanthology; the confirming URLs are in each entry's `note`. DOI resolution was not checked
  directly: run `python tools/check_refs.py references.bib` locally.
- Fields that could not be confirmed (pages, issue, DOI) were left out. Entries with open questions carry a
  `todo` field.
- The Ukrainian quota is not met yet (search was cut short); to be completed.

## Counts per topic (main file)
| Topic | Entries | 2021–2026 |
|---|---|---|
| 1 Detectors and pose estimation | 9 | 6 |
| 2 Vision-based fall detection and datasets | 14 | 10 |
| 3 Biomechanics (XCoM, MoS, inverted pendulum) | 3 | 1 |
| 4 Temporal models (LTC, CfC, LSTM, Neural ODE) | 4 | 2 |
| 5 VLM and video anomaly detection | 16 | 14 |
| 6 Cascades, routing, deferral, selective prediction | 6 | 4 |
| 7 Calibration and conformal / risk control | 7 | 5 |
| 8 Change detection | 7 | 2 |
| 9 Dempster–Shafer and evidential fusion | 10 | 2 |
| 10 Ukrainian professional journals | 10 | 6 |
| **Total** | **86** | **52 (60.5%)** |

- **2021–2026 share: 60.5%** (52 of 86). The target was 60% or more: met.
- **Ukrainian share: 11.6%** (10 of 86). The target was 18–25 entries (20–25%): **NOT met**.
- Ukrainian entries:
  - Radio Electronics, Computer Science, Control ×2 (2022, 2023)
  - Cybersecurity: Education, Science, Technique ×2 (2019, 2020)
  - Information Technologies and Computer Engineering (VNTU, 2024)
  - Visnyk VNTU (2025)
  - Adaptive Systems of Automatic Control (KPI, 2021)
  - Advanced Information Systems (NTU KhPI, 2020)
  - Naukovyi Visnyk NHU (2016)
  - Cybernetics and Systems Analysis (2024)
- No NURE-authored paper could be fully verified.

## Entries with partial metadata or minor caveats
- `alam2022vision`: the journal (Computers in Biology and Medicine) and the year are confirmed. Volume 146 and article 105626 could not be confirmed, so they are omitted.
- `kwolek2014human`: the DOI (10.1016/j.cmpb.2014.09.005, from memory) was not confirmed, so it is omitted.
- `cao2021openpose`: the published issue is 2021, 43(1), pp. 172–186. It first appeared online in 2019.
- `platt2000probabilistic`: the book's imprint is 2000. The chapter is often cited as 1999.
- `charfi2013optimized` (Le2i) and `auvinet2010multiple` (Multicam): the metadata comes from citing papers and reference lists, not from the publisher. Some author first names were filled in from memory: Imen, Johel, Julien, Rached for Charfi et al.; Edouard, Caroline, Jean, Alain, Jacqueline for Auvinet et al.
- `murphy2000combining`: the first name "Catherine K." and issue 1 were filled in from memory. The other fields are confirmed.
- `curtze2024notes`: the authors' first names were filled in from memory. The surnames, volume, article number and DOI are confirmed.
- `dempster1967upper`: the DOI 10.1214/aoms/1177698950 is inferred from the Project Euclid identifier. It follows the same pattern as the confirmed Lorden DOI.
- `wu2020not` (XD-Violence): the DOI is confirmed. The LNCS volume and pages are omitted.
- `lee2023training`: the ICASSP 2023 version is confirmed only through the authors' publication pages. The longer NN-CUSUM version is still an arXiv preprint.
- arXiv-only entries (no peer-reviewed version found):
  - RTMPose
  - BlazePose
  - Holmes-VAD
  - Gemma 3 tech report
  - Qwen2.5-VL tech report
  - SeePhys
  - OmniFall (an author page lists it at ECCV 2026; not verified)
- `tymoshyn2021analysis`: the English title is as indexed in DOAJ. The Ukrainian original title and the transliteration of the authors' names were not verified.
- `svatiuk2020zastosuvannia`: the Ukrainian title is used, but the authors are given in DOAJ Latin transliteration.
- `davydenko2023development`: the title keeps the published misspelling "inconcistancy".
- `berezhnyi2020information`, `kovalenko2016clustering`, `onufriienko2024recognition`: page ranges were not confirmed and are omitted.

## Candidates dropped because they could not be verified
- Danchenko O. B., Broyda Yu. I., «Метод визначення типу фізичної вправи за послідовністю поз людини» (Управління розвитком складних систем, No. 56). The title and authors are confirmed. The year (2023 or 2024) and the DOI are not.
- Skarga-Bandurova I. S., "Modeling decisions with Dempster-Shafer belief structures" (EEJET). The year, issue and DOI were not confirmed.
- "Information technology of ranking of expert estimates on the basis of the theory of evidence" (EEJET 2012, DOI 10.15587/1729-4061.2012.5346). The authors were not confirmed.
- Shapovalova, Moskalenko, Baranichenko, "Increasing the Receptive Field of Neurons in CNNs" (Cybernetics and Systems Analysis 59(2), 2023). It appears only in an issue listing. The DOI and pages were not found.
- Onufriienko & Taranenko, filtering and wavelet papers in Cybernetics and Systems Analysis (2022–2023). They are off topic and their metadata is incomplete.
- Bodyanskiy, Shafronenko et al., "Credibilistic robust online fuzzy clustering in data stream mining tasks" (RECSC 2023). The DOI, issue and pages were not found before the search budget ran out.
- The ensemble of shallow CNNs for gender classification in video streams (Advanced Information Systems, 2019). The authors and DOI were not confirmed.
- Zherebukh & Farmaha, «Використання нейронних мереж для визначення об'єктів на зображенні» (Комп'ютерні системи проектування, 6(1), 2024). The DOI was not confirmed.
- Chen et al., "Fall Detection Based on Key Points of Human-Skeleton Using OpenPose" (Symmetry 2020). The authors were not confirmed.
- Inturi et al. LSTM keypoints paper in the Arabian Journal for Science and Engineering. No such record was found. The Sensors 2023 paper by the same group was used instead.
- Galvão et al., "A multimodal approach using deep learning for fall detection" (ESWA). The authors and volume were not confirmed.
- Mashtalir et al., papers in Cybernetics and Systems Analysis 2020–2023 on video. None were found. Only a 2019 article and a 2020 Springer book exist, and their metadata was not fully collected.
- Ultralytics YOLO11: its software citation was not searched (budget). YOLO11 is not in the file.
- The suspicious references listed in the task were not included.
