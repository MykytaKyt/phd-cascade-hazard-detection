# Baseline results on UCF-Crime and XD-Violence (primary sources)

Verified 2026-10-26 through web-search snippets only (arXiv, CVF and AAAI pages could not be fetched directly).
Metrics: UCF-Crime uses frame-level ROC AUC (%), XD-Violence uses frame-level AP (%). Each number is the one the
method's **own paper** reports. A cell is left empty when that paper's number was not confirmed.

Confidence:
- **confirmed**: a snippet from the paper itself (CVF/AAAI/arXiv text or abstract) shows the number, and other papers' tables report the same value.
- **single snippet**: only one snippet from the primary source, or the primary source confirmed only indirectly.

| Method | Key | Setting | Backbone | UCF-Crime AUC | XD AP | Confirming URL(s) | Confidence |
|---|---|---|---|---|---|---|---|
| Sultani et al. 2018 (MIL ranking, with sparsity + smoothness constraints) | sultani2018real | weakly supervised (video-level labels, MIL) | C3D (FC6 features) | 75.41 | — (XD-Violence did not exist yet) | https://openaccess.thecvf.com/content_cvpr_2018/papers/Sultani_Real-World_Anomaly_Detection_CVPR_2018_paper.pdf ; https://arxiv.org/pdf/1801.04264v3 | confirmed |
| RTFM (Tian et al. 2021) | tian2021weakly | weakly supervised | I3D RGB (C3D variant: 83.28 AUC) | 84.30 | 77.81 | https://openaccess.thecvf.com/content/ICCV2021/papers/Tian_Weakly-Supervised_Video_Anomaly_Detection_With_Robust_Temporal_Feature_Magnitude_Learning_ICCV_2021_paper.pdf ; https://arxiv.org/pdf/2101.10030 ; https://github.com/tianyu0207/RTFM/issues/35 | confirmed |
| VadCLIP (Wu et al. 2024) | wu2024vadclip | weakly supervised | frozen CLIP ViT-B/16 (image + text encoders) | 88.02 | 84.51 | https://ojs.aaai.org/index.php/AAAI/article/view/28423 ; https://ink.library.smu.edu.sg/sis_research/9873 ; https://arxiv.org/abs/2308.11681 | confirmed (see note 1) |
| LAVAD (Zanella et al. 2024) | zanella2024harnessing | training-free | BLIP-2 captioner + Llama-2-13b-chat + ImageBind encoders | 80.28 | 62.01 (XD AUC 85.36) | https://openaccess.thecvf.com/content/CVPR2024/papers/Zanella_Harnessing_Large_Language_Models_for_Training-free_Video_Anomaly_Detection_CVPR_2024_paper.pdf ; https://openaccess.thecvf.com/content/CVPR2024/supplemental/Zanella_Harnessing_Large_Language_CVPR_2024_supplemental.pdf ; https://arxiv.org/html/2404.01014v1 ; https://github.com/lucazanella/lavad | confirmed (UCF); single snippet (XD) |
| AnomalyCLIP (Zhou et al., ICLR 2024) | zhou2024anomalyclip | zero-shot **image** anomaly detection | CLIP with object-agnostic prompts | — | — | https://proceedings.iclr.cc/paper_files/paper/2024/file/d7b50b8ac2c781a12f26155f48310d8d-Paper-Conference.pdf ; https://github.com/zqhang/AnomalyCLIP | confirmed: no video number exists (see note 2) |
| Holmes-VAD (Zhang et al. 2024) | zhang2024holmesvad | fine-tuned multimodal LLM (LoRA) on VAD-Instruct50k: single-frame temporal labels plus instruction data, so not strictly weakly supervised | frozen LanguageBind video encoder (CLIP ViT-L/14) + temporal sampler + 7B LLM (Video-LLaVA/Vicuna initialisation per a later paper) | 89.51 | 90.67 | https://arxiv.org/pdf/2406.12235 ; https://arxiv.org/html/2406.12235v2 ; https://github.com/pipixin321/HolmesVAD | single snippet (see note 3) |
| Open-vocabulary VAD (Wu et al. 2024) | wu2024open | open-vocabulary (trained on base categories; detects and classifies seen and unseen anomalies) | frozen CLIP (see note 4) | 86.40 | 66.53 | https://arxiv.org/html/2311.07042v3 ; https://arxiv.org/pdf/2311.07042 ; https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_Open-Vocabulary_Video_Anomaly_Detection_CVPR_2024_paper.pdf | confirmed |

## Notes

1. **VadCLIP 88.02 vs 82.08.** The abstract and the comparison tables give 88.02 AUC on UCF-Crime and 84.51 AP on XD-Violence.
   One sentence in the results text of the arXiv version (2308.11681) reads "84.51% AP and 82.08% AUC", which looks like a typo.
   Cite 88.02, the value in the abstract and tables and the one later papers use.
2. **AnomalyCLIP (Zhou et al.)** evaluates on 17 image datasets: 7 industrial (MVTec AD, VisA, MPDD, BTAD, SDD, DAGM,
   DTD-Synthetic) and 10 medical (ISIC, CVC-ClinicDB, CVC-ColonDB, Kvasir, Endo, TN3K, HeadCT, BrainMRI, Br35H, COVID-19).
   It reports **no UCF-Crime or other video result**, so leave its cells empty. A different paper with the same name,
   *AnomalyCLIP* by Zanella et al. ("Delving into CLIP latent space for Video Anomaly Recognition", arXiv:2310.02835),
   does evaluate on UCF-Crime, ShanghaiTech and XD-Violence. Its UCF-Crime number was not confirmed here. Do not mix the two papers up.
3. **Holmes-VAD.** The arXiv abstract/results report 90.67 AP (XD-Violence) and 89.51 AUC (UCF-Crime). Some later papers list
   lower values for Holmes-VAD (e.g. 84.61 AUC / 84.96 AP) from other evaluation settings. Use the paper's own 89.51 / 90.67.
   The paper is an arXiv preprint (no peer-reviewed venue found).
4. **Open-vocabulary VAD.** The paper reports 86.40 AUC (UCF-Crime), 66.53 AP (XD-Violence) and 62.94 AUC (UBnormal), and states
   that it beats DMU by 1.26 AUC, 2.63 AP and 3.03 AUC. These numbers come from the open-vocabulary protocol
   (UCF-Crime base classes: Abuse, Assault, Burglary, Road Accident, Robbery, Stealing). They are not directly comparable
   with fully weakly supervised numbers. The CLIP backbone variant was not confirmed in snippets for this paper.
   VadCLIP, by the same group, uses ViT-B/16.
5. **UCF-Crime test split.** Confirmed from Sultani et al. 2018 (CVPR PDF and arXiv): "the training set consisting of
   800 normal and 810 anomalous videos ... and the testing set including the remaining 150 normal and 140 anomalous
   videos". So the test split is **290 videos (140 anomalous + 150 normal)**, and frame-level temporal annotations exist only for the test videos.
   The paper's own table also gives the baselines: binary classifier 50.0, Hasan et al. 50.6, Lu et al. 65.51,
   proposed without constraints 74.44, with constraints 75.41.
