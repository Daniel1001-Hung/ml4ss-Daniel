# Week 2 · Multimodal Data and Measurement

[Open the Week 2 notebook in Colab](https://colab.research.google.com/github/sunshineluyao/ml4ss/blob/main/modules/01-measurement/01_week2_data_descriptor_release.ipynb)

**Outcome.** Update your Week 1 proposal and release an actual readable dataset v0.1 under the instructor-confirmed course Hugging Face organization by the individual Week 2 reflection, **Sunday November 8, 2026, 23:00 China Standard Time**. A release plan or local preview alone is insufficient. Your reflection remains one PDF of at most two A4 pages, Times New Roman 12 pt, with one practiced figure.

**Individual and collaborative work.** Weekly reflections are individual submissions. Canvas tools randomly generate the temporary classroom groups separately each day. Discuss, inspect and review in that day's group, then explain your own judgment and revision. GitHub/Hugging Face organizations are publication infrastructure, rather than reflection teams. Personal sign-in preserves authorship and contribution credit.

**Why this sequence.** Theory defines what the measure means. Practice checks a real source and one output. Research confronts the interpretation with published evidence. Innovation makes a justified change inspectable through data, documentation and independent loading. This prepares Week 3 explanation/prediction and the November 18 proposal/dataset workshop.

## Shared multimodal preview before the project

The public teaching example pairs **eight Cleveland Museum of Art catalogue records and CC0 artwork images** through artwork IDs. It is a convenience sample from one API retrieval, not a representative art dataset or a time series. Its artwork dates describe the works; image-capture times are not supplied in the selected fields.

An **embedding** is a numerical representation. The original public CLIP ViT-B/32 model encoded the images, catalogue text and three queries. The notebook's default uses those **saved encodings** and recomputes image/catalogue similarities and their weighted combination. It does not run a live model, train one or require a GPU. Exact source hashes, original model revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`, preprocessing and vectors are in [multimodal_demo.json](multimodal_demo.json). The [recomputation script](prepare_multimodal_demo.py) regenerates the example from public originals.

Before running, predict which information will help. Change **DEMO_QUERY** among the printed landscape, portrait and quantum-computer requests. Compare **DEMO_MODE** `image`, `catalogue` and `both`, and **ALPHA** between zero and one. Equal weighting is a teaching choice, not a fitted or validated optimum. Inspect the actual candidates and original catalogue links. The absent quantum-computer photograph illustrates why a first rank does not establish relevance.

An **AI agent** selects tools and next steps toward a task. A **vision–language model** processes images/text; **retrieval** selects candidates; **deployment** makes the system available as a repeatable service with monitoring and permission controls. The executed `inspect_collection` preview follows a fixed read-only policy: retrieve, read the catalogue, draft an evidence record. Its sentence uses a deterministic template, not an LLM. A publish request returns `HUMAN_APPROVAL_REQUIRED` and performs no external action. Local retrieval timing excludes model encoding, network and service overhead. This is not a robot-control or production-agent benchmark.

The frontier illustrations are π0.5, CoRL 2025 [35], and RiOSWorld, NeurIPS 2025 [36]. They connect data coverage with robot generalization and computer-use evaluation. Keep the two Nature papers as the shared measurement readings. **Your assessed outcome remains your proposal and dataset v0.1, with one project Figure 1.** The gallery is not an extra figure or deployment assignment. The established museum/model example does not replace the 2025–2026 project publication criterion.

## What the data-release terms mean

Start with this question: can another person find your files, interpret a row and know the permitted uses?

| Name | Category | Practical meaning |
|---|---|---|
| FAIR | Guiding principles | Findable, Accessible, Interoperable, Reusable; use identifiers, access routes, shared definitions and provenance. Authorized access can be FAIR. |
| CARE | Indigenous data-governance principles | Collective Benefit, Authority to Control, Responsibility, Ethics; preserve the Indigenous origin and actual community authority. |
| Dataset Card | Human-readable documentation | Explain purpose, records, preparation, limitations and permitted uses. |
| Croissant 1.1 | Machine-readable metadata format | Describe actual files, records and field meanings in JSON-LD. |
| Croissant RAI | Responsible AI metadata extension | Record limitations, bias, sensitive information, use and social impact. |
| Scientific Data | Peer-reviewed data journal | Publish reusable dataset descriptions and technical validation. |
| NeurIPS 2026 E&D | Research-conference track and submission requirements | State the evaluative claim, assumptions and limits; dataset contributions need hosted data and core/minimal RAI metadata. |
| Hugging Face | Hosting platform | Share versioned files, cards and access routes; hosting itself does not certify scientific quality. |

The current NeurIPS track, formerly Datasets & Benchmarks, studies evaluation and data resources alongside model research. It provides a concrete research route for a well-supported measurement, data or evaluation contribution. Parser success and complete metadata alone do not establish validity or conference readiness. Public readability, permission to redistribute and permission for a particular application must be checked separately.

## Choose support

| Route | What is supplied | What you contribute |
|---|---|---|
| A Guided | Filled defaults and three working CPU starters | Your evidenced critique, verified improvement and own release |
| B Curated | Ten publication-linked choices below | A feasible question, audit and adapted measurement/release |
| C Independent | Common passport and tabular/manifest adapter | An eligible source, actual access and same checks |

All routes share the criteria. Switching support or completely changing your topic is allowed. If no eligible usable source emerges in 15 minutes, begin with A. This applies autonomy support with structure [13], alongside criterion-based peer feedback [14]. The studies do not directly test this course's GitHub/HF workflow or establish its learning gains.

## Notebook blocks and exact settings

1. **Choose:** save a Drive copy. Choose ROUTE and STARTER (corelli, finance, heritage, custom). Initial independent reasoning is Human-Only.
2. **Inspect:** read the source passport, actual sample rows, unit, scope, time and quality table. Network failure uses a clearly labelled 20-row dated excerpt rather than fabricated rows. Its counts differ from the full source.
3. **Measure:** edit TOPIC_OVERRIDE, KEYWORDS_OVERRIDE and MAX_ROWS. Literal terms separated by `|` filter the chosen source, while `ALL` keeps rows. Topic text records your question; it does not retrieve a new dataset. Inspect Figure 1 and numerical checks, then replace the teaching responses with your own critique/change/check/value.
4. **Document:** inspect the Dataset Card, explicit field dictionary, source hash/provenance, quality report and Croissant 1.1 with eight RAI/provenance entries. A parser/data reload is not a construct-validity or hosted-access test.
5. **Release:** keep LIVE_PUBLISH off for the demonstration. After real scientific, rights, metadata and peer reviews, set the confirmed course organization, author, unique project slug and version. Explicitly enable publication to stage a candidate. The hidden token prompt verifies organization membership and never saves the token. Check the hosted descriptor with the official validator and record the actual result before finalizing the version tag. A later update uses review and a new version, rather than repeating first-repository creation.
6. **Revise:** ask a peer to load the final hosted commit in a separate session, inspect rows/IDs/metadata, and record their feedback and your response. Update your own proposal.

The optional appendix uses a fully specified synthetic 200-point calibration DGP for three 3D views of noise, units and known-offset alignment. Those views are separate from the empirical starter releases and are not additional reflection figures.

## Critique one specific issue

Use **source evidence → proposed change → validation → scientific/reuse value**. Choose construct/rule, observation unit, sampling/omissions, quality/validity, time/space/modal alignment, rights/authority, or reproducibility. More data may help, and justified annotations, measurements or validated processing may also help. Reuploading, format conversion or volume alone does not establish originality. The filled measurement rules are teaching prototypes, not independent student contributions.

For a benchmark, additionally specify target/task, split unit, leakage control, baseline, metric/uncertainty and a reproducible protocol. The main tutorial does not train a predictive model or identify a causal effect.

## Dataset menu / 数据集菜单

Choose a 2025–2026 Scientific Data or NeurIPS dataset-track publication **and** its actual open data release. Read one selected descriptor closely; the ten choices are not ten required readings. A source article's license does not automatically license its data. Rights and any community authority still apply.

### Distant Listening Corpus / Corelli [1]

Arts, media and culture / 艺术媒体与文化

149 observed movement metadata rows. A notated beat is not a second of audio.

Data: https://github.com/DCMLab/corelli

Reuse: CC BY-NC-SA 4.0. Preserve the Corelli source attribution [15].

Critique and improve: Audit whether annotation density represents your construct. Compare a reviewed sample under a shared codebook.

Feasible entry: Built-in CPU starter; no score rendering or audio model required.

### ICH-NER [2]

Arts, media and culture / 艺术媒体与文化

Chinese intangible-cultural-heritage entity labels. Inspect actual sentence boundaries, label counts and splits.

Data: https://doi.org/10.6084/m9.figshare.28658555

Reuse: Deposited data record CC BY 4.0. Do not additionally copy unreleased source webpages.

Critique and improve: Critique cultural concepts becoming labels. Add authorized terms or a manually checked label audit.

Feasible entry: Read the small train/dev text files; no language-model training required.

### Glo-CoH [3]

Arts, media and culture / 艺术媒体与文化

1,250 heritage-component attribute records; components differ from UNESCO properties.

Data: https://doi.org/10.25375/uct.28547267

Reuse: CC BY 4.0. The source record also supplies geometry.

Critique and improve: Compare component/property/area coverage. An exposure proxy does not establish experienced loss.

Feasible entry: Built-in 99 KB XLSX starter. Geometry is optional.

### Development finance, climate and SDG annotations [4]

Finance, economics, law and social science / 金融经济法律与社会科学

20,985 project rows in the climate workbook, commitment years 2000–2021. Amounts use constant 2021 USD.

Data: https://doi.org/10.6084/m9.figshare.30015124

Reuse: The data record is CC0. Retain the paper, AidData lineage and original IDs.

Critique and improve: Check missing amounts, compare count/funding-weighted coverage, audit title matching or join public country-year indicators with match-loss checks.

Feasible entry: Built-in CPU starter; reads a roughly 9 MB workbook and filters a small subset.

### CongressData and Correlates of State Policy [5]

Finance, economics, law and social science / 金融经济法律与社会科学

Member-years 1789–2024 and state-years 1900–2020. Retain variable-specific coverage and sources.

Data: https://doi.org/10.6084/m9.figshare.28146914

Reuse: CC BY 4.0 in the data record. Preserve upstream variable attribution.

Critique and improve: Compare stable-ID and name joins, inspect temporal missingness, or add a justified public policy indicator.

Feasible entry: RDA files require an R/Python reader such as pyreadr; inspect a few variables/years first.

### LawShift [6]

Finance, economics, law and social science / 金融经济法律与社会科学

Legal cases and statute changes across 31 fine-grained change types. Labels depend on the subtask.

Data: https://huggingface.co/datasets/triangularPeach/LawShift

Reuse: Dataset card Apache 2.0. Check actual task files and upstream conditions.

Critique and improve: Audit statute version versus case time and distinguish observed judgments from constructed task outcomes.

Feasible entry: Release a reviewed, version-aware small annotation subset; no model fine-tuning required.

### Deep4Dist [7]

Environment, geospatial and Earth science / 环境空间与地球科学

Aerial orthophotos, not satellite imagery. RGB/NIR, object height and forest disturbance masks in Rhineland-Palatinate.

Data: https://doi.org/10.5281/zenodo.14884818

Reuse: Data CC BY 4.0. Full train and validation archives exceed 20 GB together.

Critique and improve: Check spatial split leakage, scale, mask definitions and coverage. Add an evidenced audit or permitted complementary source.

Feasible entry: Begin with metadata and a few tiles. Do not require a full archive or U-Net run.

### TraffiDent [8]

Environment, geospatial and Earth science / 环境空间与地球科学

Traffic nodes, flow, speed, occupancy, incidents and road/policy attributes.

Data: https://xaitraffic.github.io/

Reuse: Data CC BY-NC 4.0, code MIT. Linked Kaggle access may need sign-in.

Critique and improve: Check time zones, frequency and event-sensor matches. The paper abstract and project page state different year windows; use actual downloaded files to define yours.

Feasible entry: Use a small time/location subset. Switch to a built-in starter if access blocks progress.

### PAC Bench [9]

Robotics and interdisciplinary governance / 机器人与跨学科治理

Robot-task prerequisites in images: properties, affordances and constraints; labels are not execution evidence.

Data: https://huggingface.co/datasets/lens-lab/pacbench

Reuse: Card declares MIT. Retain applicable upstream image terms and attribution separately.

Critique and improve: Audit whether an affordance implies feasibility for a specified robot, and add reviewed prerequisite labels.

Feasible entry: Use a few images and an annotation table. Correct identity: lens-lab/pacbench, not the unrelated PAC-Bench/PAC-Bench privacy dataset.

### MNISQ [10]

Quantum computing and resource governance / 量子计算与资源治理

Generated quantum circuits encoding MNIST/Fashion-MNIST/Kuzushiji-MNIST at multiple fidelities; not observed quantum hardware.

Data: https://doi.org/10.5281/zenodo.19656638

Reuse: Data CC BY 4.0. Code repository has CC BY-SA 4.0. Full archive is roughly 35.6 GB.

Critique and improve: Audit gate count, depth and fidelity. Cost/access claims require additional real price/hardware evidence.

Feasible entry: Advanced route only: inspect preselected small QASM examples; neither full download nor quantum hardware is required.

## Shared research comparison

Use Bettencourt & Marchio [11] for an operational street-access measure and external/community comparisons. Use Sticher et al. [12] to examine text and satellite data-generating mechanisms, complementary coverage, improvement/enrichment/fusion, and the extra assumptions introduced by integration. These are measurement comparisons in this class, rather than optimization or causal-identification demonstrations.

## Daily workshop and short Ed records

The class rhythm is 10-minute warm-up, 50 Theory/Research, 10 individual Ed reflection, 10 break, 10 re-entry into Canvas-randomized daily groups, 50 Practice/Innovation, and 10 Human-Only handwritten exit/photo.

In the 50-minute Innovation studio: 0–5 author presents source/change, 5–15 peer critiques, 15–30 author improves/stages, 30–40 independent load/check, 40–48 author responds/revises, 48–50 append links. If the final hosted version changes later, repeat the peer's final load check on that exact commit.

Keep one **individual** Ed record and append:

- Theory: initial purpose A Measurement / B Explanation / C Prediction / D Causal inference / E Optimization / F Frontier method plus underlying aim; reason; required datasets/unit; rule/check; peer-triggered revision.
- Practice: topic/purpose, data/paper, unit/filter/settings, one Figure 1 screenshot and Colab Viewer link; interpretation and check.
- Research: a source passage/figure/row, one limitation, a revision and its validation, required datasets.
- Innovation: contribution, peer challenge, response, dataset URL/version/SHA, code/review link and final peer loading result.

## Organization workflow and release boundaries

The instructor confirms organizations and invites students using privately collected account identifiers. No organization identity is invented in the tutorial. Students retain personal accounts; no shared password. GitHub uses project branches/PRs and maintainer review. HF has native discussions/PRs without a personal fork. A contributor role can create/write its own organization repositories; write access covers others too and should be assigned deliberately. Do not assume paid fine-grained controls exist on a free organization.

For each release: issue/question → assigned author → small change → criterion-based peer review → response/revision → authorized maintainer release → independent loading at the stated version. Preserve source attribution and credit the actual contributors. If raw redistribution is not permitted, release a justified permitted derivative/annotation or reproducible access process; do not claim to host forbidden raw files. This week still requires an actual readable permitted dataset v0.1, so switch to a feasible starter if needed.

The notebook is a **first-release** helper with exists_ok=False. If creation succeeds but upload/tag fails, inspect the existing repository; do not blindly rerun creation. Correct files through the Hub UI or a reviewed PR, then validate and tag the intended commit. Record failures honestly. For subsequent versions, use a PR/review and a new immutable version. Never label an unperformed upload, peer check or online validation as complete.

Hugging Face's generated Croissant core may omit RAI. Inspect the saved file and retain all eight entries: rai:dataLimitations, rai:dataBiases, rai:personalSensitiveInformation, rai:dataUseCases, rai:dataSocialImpact, rai:hasSyntheticData, prov:wasDerivedFrom, prov:wasGeneratedBy. Compare platform distributions/fields with your released files. The official NeurIPS-linked validator remains the external check; course prototypes are not claims of complete conference readiness.

## AI and authorship

Initial judgments, ten-minute individual Ed reflection and handwritten exit are Human-Only. Coding/release work and the weekly reflection are AI-Assisted for explanations, debugging/test suggestions, language/translation and critique of your own draft. Produce the core reasoning, verify outputs, and disclose tool/model, date, purpose, significant prompts, incorporated/rejected outputs, human revisions and checks. State no use when applicable. Never put credentials into prompts, cells, saved outputs or commits.

## Full references

Public demonstration sources [33–34] identify the original collection and model. Frontier illustrations [35–36] are optional context. Dataset publications [1–10] are a choice menu. [11–12] are the shared Nature readings. [13–14] support teaching mechanisms. [15] attributes the older underlying Corelli source. [16–25] are authoritative standards/platform requirements. [28–32] illustrate modalities, rather than substitute for project eligibility.

[1] Hentschel, J.; Rammos, Y.; Neuwirth, M.; Rohrmeier, M. (2025). A corpus and a modular infrastructure for the empirical study of (an)notated music. Scientific Data, 12, 685. https://doi.org/10.1038/s41597-025-04976-z

[2] Long, S.; Li, W. (2026). A Chinese Named Entity Recognition Dataset for Intangible Cultural Heritage. Scientific Data, 13, 335. https://doi.org/10.1038/s41597-026-06700-x

[3] Bescoby, D.; Clarke, J.; Okolie, C.; Nicholls, R. J.; Simpson, N. P.; Vafeidis, A. T.; Vousdoukas, M. I. (2026). Mapping the global distribution of coastal World Heritage Sites: Glo-CoH, a new Global Coastal Heritage dataset. Scientific Data, 13, 1082. https://doi.org/10.1038/s41597-026-07758-3

[4] Qi, J.; Tang, Y.; Zhang, Z.; Wang, Y.; Qian, H. (2026). A Benchmark Dataset of Chinese Development Finance with Climate Relevance and SDG Annotations from 2000–2021. Scientific Data, 13, 277. https://doi.org/10.1038/s41597-026-06605-9

[5] Grossmann, M.; Lucas, C.; Yoel, B. (2025). Introducing CongressData and Correlates of State Policy. Scientific Data, 12, 1185. https://doi.org/10.1038/s41597-025-05495-7

[6] Han, Z.; Yang, Y.; Feng, Y.; Huang, W.; Ding, X.; Li, C.; Ge, J.; Ng, V. (2025). LawShift: Benchmarking Legal Judgment Prediction Under Statute Shifts. Advances in Neural Information Processing Systems 38, Datasets and Benchmarks Track. https://doi.org/10.52202/085713-4010

[7] Rodríguez-Paulino, E.; Stoffels, J.; Schlerf, M.; Röder, A.; Wagner, A.; Udelhoven, T. (2026). An AI-ready remote sensing dataset for high-resolution forest disturbance mapping. Scientific Data, 13, 490. https://doi.org/10.1038/s41597-026-07084-8

[8] Gou, X.; Li, Z.; Lan, T.; Lin, J.; Li, Z.; Zhao, B.; Zhang, C.; Wang, D.; Zhang, X. (2025). TraffiDent: A Dataset for Understanding the Interplay Between Traffic Dynamics and Incidents. Advances in Neural Information Processing Systems 38, Datasets and Benchmarks Track. https://doi.org/10.52202/085713-2796

[9] Gundawar, A.; Sagar, S.; Senanayake, R. (2025). PAC Bench: Do Foundation Models Understand Prerequisites for Executing Manipulation Policies? Advances in Neural Information Processing Systems 38, Datasets and Benchmarks Track. https://doi.org/10.52202/085713-3671

[10] Placidi, L.; Hataya, R.; Mori, T.; Aoyama, K.; Morisaki, H.; Mitarai, K.; Fujii, K. (2026). MNISQ: A Large-Scale Quantum Circuit Dataset for Machine Learning in the NISQ Era. Scientific Data, 13, 810. https://doi.org/10.1038/s41597-026-07493-9

[11] Bettencourt, L. M. A.; Marchio, N. (2025). Infrastructure deficits and informal settlements in sub-Saharan Africa. Nature, 645, 399-406. https://doi.org/10.1038/s41586-025-09465-2

[12] Sticher, V.; Bara, C.; Kotajoki, J.; Bromley, L.; Crevoisier, T.; Dalsasso, E.; Dietrich, O.; Donnay, K.; Dorm, F.; Fiol, M.; Li, X.; Naghizadeh, M. H.; Scher, C.; Van Den Hoek, J.; Schindler, K.; Tuia, D.; Wegner, J. D. (2026). Advancing conflict research and response through satellite-derived data. Nature, 657, 891-899. https://doi.org/10.1038/s41586-026-11004-6

[13] Patzak, A.; Zhang, X. (2025). Blending Teacher Autonomy Support and Provision of Structure in the Classroom for Optimal Motivation: A Systematic Review and Meta-Analysis. Educational Psychology Review, 37, 17. https://doi.org/10.1007/s10648-025-09994-2

[14] Hornstein, J.; Keller, M. V.; Greisel, M.; Dresel, M.; Kollar, I. (2025). Enhancing the Peer-Feedback Process Through Instructional Support: A Meta-Analysis. Educational Psychology Review, 37, 42. https://doi.org/10.1007/s10648-025-10017-3

[15] Hentschel, J.; Moss, F. C.; Neuwirth, M.; Rohrmeier, M. A. (2021). A semi-automated workflow paradigm for the distributed creation and curation of expert annotations. Proceedings of ISMIR 2021, 262–269. https://doi.org/10.5281/zenodo.5624417

[16] Wilkinson, M. D., et al. (2016). The FAIR Guiding Principles for scientific data management and stewardship. Scientific Data, 3, 160018. https://doi.org/10.1038/sdata.2016.18 ; GO FAIR implementation: https://www.go-fair.org/fair-principles/

[17] Carroll, S. R., et al. (2020). The CARE Principles for Indigenous Data Governance. Data Science Journal, 19, 43. https://doi.org/10.5334/dsj-2020-043 ; Global Indigenous Data Alliance: https://www.gida-global.org/careprinciples

[18] MLCommons (2026). Croissant Format Specification, version 1.1, January 29 2026. https://docs.mlcommons.org/croissant/docs/croissant-spec-1.1.html

[19] NeurIPS (2026). Evaluations and Datasets Hosting Guidelines. https://neurips.cc/Conferences/2026/EvaluationsDatasetsHosting

[20] MLCommons (2024). Croissant RAI Specification, version 1.0. Read with the current NeurIPS field requirements. https://docs.mlcommons.org/croissant/docs/croissant-rai-spec.html

[21] Hugging Face (n.d.). Dataset Cards. https://huggingface.co/docs/hub/datasets-cards

[22] Hugging Face (n.d.). Access control in organizations. https://huggingface.co/docs/hub/organizations-security

[23] Hugging Face (n.d.). Pull requests and Discussions. https://huggingface.co/docs/hub/repositories-pull-requests-discussions

[24] GitHub (n.d.). Best practices for organizations. https://docs.github.com/en/organizations/collaborating-with-groups-in-organizations/best-practices-for-organizations

[25] NeurIPS (2026). Call for Evaluations and Datasets; Reviewing Guidelines. https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets ; https://neurips.cc/Conferences/2026/EvaluationsDatasetsReviewerGuidelines

[26] Zhang, L. (2026). STATS 201 Introduction to Machine Learning for Social Science, Autumn 2026 syllabus. Duke Kunshan University. Current course copy distributed through Canvas and Ed Discussion.

[27] Zhang, L. (maintainer). ml4ss tutorial repository, main branch. Inspected November 1 2026. https://github.com/sunshineluyao/ml4ss ; Week 2 directory: https://github.com/sunshineluyao/ml4ss/tree/main/modules/01-measurement

[28] OpenSLR (n.d.). LibriSpeech ASR corpus, SLR12. Official data and documentation. https://www.openslr.org/12

[29] Ego-Exo4D Consortium (n.d.). Ego-Exo4D dataset and access conditions. https://ego-exo4d-data.org/

[30] ScanNet authors (n.d.). ScanNet RGB-D data, organization, and terms of use. https://github.com/ScanNet/ScanNet

[31] Reyes-Ortiz, J., et al. (2013). Human Activity Recognition Using Smartphones. UCI Machine Learning Repository. https://doi.org/10.24432/C54S4K

[32] Stanford SNAP (n.d.). Social circles Facebook, network data and documentation. https://snap.stanford.edu/data/ego-Facebook.html

[33] Cleveland Museum of Art. (n.d.). Open Access and Open Access API. Public-domain artwork images and catalogue data under CC0. Accessed November 1 2026. https://www.clevelandart.org/open-access ; https://openaccess-api.clevelandart.org/

[34] OpenAI. (n.d.). CLIP official implementation and original model release. Established model component, not a 2026 frontier-paper claim. Model openai/clip-vit-base-patch32, revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268. https://github.com/openai/CLIP ; https://huggingface.co/openai/clip-vit-base-patch32

[35] Black, K.; Brown, N.; Darpinian, J.; Dhabalia, K.; Driess, D.; Esmail, A.; Equi, M. R.; Finn, C.; Fusai, N.; Galliker, M. Y.; Ghosh, D.; Groom, L.; Hausman, K.; ichter, b.; Jakubczak, S.; Jones, T.; Ke, L.; LeBlanc, D.; Levine, S.; Li-Bell, A.; Mothukuri, M.; Nair, S.; Pertsch, K.; Ren, A. Z.; Shi, L. X.; Smith, L.; Springenberg, J. T.; Stachowicz, K.; Tanner, J.; Vuong, Q.; Walke, H.; Walling, A.; Wang, H.; Yu, L.; Zhilinsky, U. (2025). π0.5: a Vision-Language-Action Model with Open-World Generalization. Proceedings of the 9th Conference on Robot Learning, PMLR 305, 17–40. https://proceedings.mlr.press/v305/black25a.html

[36] Yang, J.; Shao, S.; Liu, D.; Shao, J. (2025). RiOSWorld: Benchmarking the Risk of Multimodal Computer-Use Agents. Advances in Neural Information Processing Systems, 38, Main Conference Track. https://doi.org/10.52202/085713-0286

## Execution record and reuse

Prepared November 1, 2026. The shared preview was recomputed from original public CLIP weights and tested for all modes and supplied queries. The project default local-preview path was executed in a fresh in-process IPython environment for each of the three live sources, including actual rows, measured data, Figure 1, saved card/Croissant, local parser and datasets reload. The default does not authenticate, stage a Hub repository or tag a release. The online validator, signed-in hosted Colab execution, live organization upload and final peer hosted reload must be performed on the actual student release; no success is claimed for those unperformed actions.

Original code is MIT. Original instructional prose and original conceptual visualizations are CC BY 4.0. External datasets retain their own licenses. Keep teaching release DOCX/PDF/PPTX, submissions, private usernames and grades outside this repository.

### Optional original-model recomputation

Use a fresh Python environment, install `requests pillow numpy torch transformers==4.57.6` with the appropriate CPU PyTorch wheel, then run `python prepare_multimodal_demo.py --output multimodal_demo.json`. It downloads public original weights and live museum source files; it does not publish data. The model revision and preprocessing are pinned; an API refresh can change the convenience sample. For exact replay use the committed fixture and its source hashes. The default student notebook needs neither this environment nor a model download.
