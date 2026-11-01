# Project-2_Deep-_Learning_-Applications

## Project 2, Week 5: Neural Networks, Convolutions, and Transfer Learning

**Author:** Raja Fahad Kiyani | **Course:** Introduction to Applied AI (BS 9th Semester, Department of Electronics) | **Platform:** Kaggle notebook

- **Dataset:** Intel Image Classification, 6 classes, 14,034 train / 3,000 test
- **Model:** ResNet-18 pretrained on ImageNet
- **Feature extraction (3,078 trainable parameters):** val accuracy 0.905
- **Fine-tuning layer4 (about 8.4M trainable):** val accuracy 0.933
- **Test accuracy (used once):** 0.933; weakest class: glacier (recall 0.881), confused with mountain
- **Most instructive error:** a dark city street scene with a skyline was labelled "buildings" but predicted "street" with probability 1.00, which shows that scene classes overlap and that a confident prediction is not always a wrong one.
- **Next: my own domain in Week 6:** [write your domain and candidate dataset here]

---

## 1. Dataset and splits

Six scene classes: buildings, forest, glacier, mountain, sea, street. Images are 150 x 150 x 3 (`uint8`, values 0 to 255).

| Class | Train folder | Test folder |
|---|---:|---:|
| buildings | 2,191 | 437 |
| forest | 2,271 | 474 |
| glacier | 2,404 | 553 |
| mountain | 2,512 | 525 |
| sea | 2,274 | 510 |
| street | 2,382 | 501 |
| **Total** | **14,034** | **3,000** |

The training folder is split 85% / 15% with a fixed seed (42). Augmentation is applied to the training view only; validation and test use the plain evaluation transform.

| Split | Images | Transform | Used for |
|---|---:|---|---|
| Train | 11,929 | random crop, horizontal flip, colour jitter, normalise | updating weights |
| Validation | 2,105 | resize 224, normalise | choosing the best epoch, comparing methods |
| Test | 3,000 | resize 224, normalise | final score, used once |

Split size: $n_{val} = \lfloor 0.15 \times 14{,}034 \rfloor = 2{,}105$ and $n_{train} = 14{,}034 - 2{,}105 = 11{,}929$.

Inputs are normalised with the ImageNet statistics the pretrained network expects:

$$x_{norm} = \frac{x - \mu}{\sigma}, \qquad \mu = [0.485,\ 0.456,\ 0.406], \quad \sigma = [0.229,\ 0.224,\ 0.225]$$

---

## 2. Method

**Feature extraction (epochs 1 to 3).** All ResNet-18 weights are frozen and the 1,000-class head is replaced by a new `Linear(512, 6)` layer trained with Adam (learning rate 1e-3).

**Fine-tuning (epochs 4 to 6).** The last residual block `layer4` is unfrozen with a gentle learning rate (1e-4); the new head keeps 1e-3. The best validation epoch is kept.

| Quantity | Formula | Value |
|---|---|---:|
| New head parameters | $512 \times 6 + 6$ | 3,078 |
| Total parameters | frozen body + head | 11,179,590 |
| Share trained (feature extraction) | $3{,}078 / 11{,}179{,}590$ | 0.0275% |
| Trainable after unfreezing `layer4` | $3{,}078 + 8{,}393{,}728$ | 8,396,806 |
| Share trained (fine-tuning) | $8{,}396{,}806 / 11{,}179{,}590$ | 75.1% |
| Conv 3x3, 3 to 64 channels | $(3 \cdot 3 \cdot 3 + 1) \cdot 64$ | 1,792 |
| Dense 67,500 to 512 | $67{,}500 \cdot 512 + 512$ | 34,560,512 |

A dense layer needs $34{,}560{,}512 / 1{,}792 \approx 19{,}286$ times more parameters than the convolution, which is the benefit of weight sharing.

Convolution formulas used to check every layer shape:

$$\text{Output size} = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1 \qquad \text{Parameters} = (K \cdot K \cdot C_{in} + 1)\cdot C_{out}$$

Training minimises the cross-entropy of the softmax probabilities:

$$p_k = \frac{e^{z_k}}{\sum_{j=1}^{6} e^{z_j}}, \qquad \mathcal{L} = -\log p_{\text{true class}}$$

---

## 3. Training results

| Epoch | Phase | Train loss | Train acc | Val loss | Val acc |
|---:|---|---:|---:|---:|---:|
| 1 | feature extraction | 0.601 | 0.807 | 0.331 | 0.895 |
| 2 | feature extraction | 0.339 | 0.885 | 0.309 | 0.895 |
| 3 | feature extraction | 0.308 | 0.892 | 0.274 | **0.905** |
| 4 | fine-tuning | 0.260 | 0.906 | 0.213 | 0.926 |
| 5 | fine-tuning | 0.166 | 0.940 | 0.214 | 0.932 |
| 6 | fine-tuning | 0.120 | 0.956 | 0.208 | **0.933** |

| Method | Trainable parameters | Best val accuracy |
|---|---:|---:|
| Feature extraction | 3,078 | 0.905 |
| Fine-tuning `layer4` | 8,396,806 | 0.933 |
| **Improvement** | | **+0.028 (2.8 points)** |

Fine-tuning gained about $0.028 \times 2{,}105 \approx 59$ more correct validation images. Training accuracy is lower than validation accuracy early on because training images are augmented (harder) and averaged while the weights are still changing. By epoch 6 the training loss (0.120) is well below the validation loss (0.208), a mild sign of overfitting, but the validation loss is not rising.

---

## 4. Test results (used once)

$$\text{accuracy} = \frac{\text{correct}}{\text{total}} = \frac{2{,}798}{3{,}000} = 0.933$$

$$\text{precision} = \frac{TP}{TP + FP}, \qquad \text{recall} = \frac{TP}{TP + FN}, \qquad F_1 = \frac{2PR}{P + R}$$

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| buildings | 0.937 | 0.918 | 0.927 | 437 |
| forest | 0.998 | 0.989 | 0.994 | 474 |
| glacier | 0.894 | 0.881 | 0.887 | 553 |
| mountain | 0.901 | 0.884 | 0.892 | 525 |
| sea | 0.947 | 0.982 | 0.964 | 510 |
| street | 0.928 | 0.950 | 0.939 | 501 |
| **Macro average** | 0.934 | 0.934 | 0.934 | 3,000 |

Confusion matrix (rows = true class, columns = predicted class):

| True \ Pred | buildings | forest | glacier | mountain | sea | street |
|---|---:|---:|---:|---:|---:|---:|
| **buildings** | **401** | 0 | 0 | 1 | 1 | 34 |
| **forest** | 0 | **469** | 4 | 1 | 0 | 0 |
| **glacier** | 1 | 1 | **487** | 47 | 16 | 1 |
| **mountain** | 1 | 0 | 49 | **464** | 10 | 1 |
| **sea** | 1 | 0 | 5 | 2 | **501** | 1 |
| **street** | 24 | 0 | 0 | 0 | 1 | **476** |

The diagonal sums to $401 + 469 + 487 + 464 + 501 + 476 = 2{,}798$, so there are 202 errors.

| Confused pair | Errors | Share of all errors |
|---|---:|---:|
| glacier and mountain (47 + 49) | 96 | 47.5% |
| buildings and street (34 + 24) | 58 | 28.7% |

The two weakest classes by recall are glacier (0.881) and mountain (0.884). Both pairs share visual content: snowy rocky slopes appear in mountain and glacier photos, and streets are made of buildings. The prediction made before training (glacier vs mountain) was correct.

---

## 5. Error analysis

In the first eight test errors, seven are true "buildings" predicted as "street" and one is predicted as "mountain". Several errors are confident (0.98, 0.90, 1.00).

| True | Predicted (confidence) | Verdict |
|---|---|---|
| buildings | mountain (0.69) | ambiguous: stone house in a rural landscape |
| buildings | street (0.98) | ambiguous: black-and-white shopfront on a road |
| buildings | street (0.66) | doubtful label: industrial hall with parallel rails |
| buildings | street (1.00) | arguable label: dark city street with a skyline |

Many errors are label overlap between classes rather than clear model failures, so the realistic ceiling is below 100%.

One prediction by hand (first test image, true class buildings): the logit for buildings is 7.21 and the closest competitor (street) is $-1.86$, so

$$p_{\text{buildings}} = \frac{e^{7.21}}{e^{7.21} + \dots} \approx \frac{1350}{1350 + 0.16 + \dots} \approx 1.000$$

---

## 6. Saved model

`resnet18_intel.pt` stores the weights only (`state_dict`), about 44.8 MB ($11.18\text{M} \times 4$ bytes in float32), under GitHub's 100 MB limit. To reload it, rebuild `resnet18` with a 6-class `fc` layer, then call `load_state_dict`.

---

## 7.Week 6 proposal
 
**Domain: solar panel condition (electronics and energy).** Photovoltaic (PV) panels lose output when they are dirty, covered or damaged, and checking them by eye is slow. This is a defect-classification task like industrial inspection, and the Week 5 pipeline (pretrained ResNet-18, feature extraction, then fine-tuning) applies directly, with colour photos.
 
**Problem.** Dust, bird droppings, snow, and electrical or physical damage reduce a panel's efficiency, and manual inspection is slow, costly and can be dangerous on large solar farms. I will build an image classifier that takes a photo of a solar panel and predicts its condition among 6 classes:
 
| Class | Meaning |
|---|---|
| Clean | panel in good condition, no obstruction |
| Dusty | dust that reduces light absorption |
| Bird-drop | bird droppings that cause partial shading |
| Snow-covered | panel partly or fully covered by snow |
| Physical-damage | external damage such as cracked glass or a broken frame |
| Electrical-damage | internal defects such as hot spots, delamination or bypass diode failure |
 
**Who benefits.** Operators and maintenance teams of solar plants, and owners of rooftop systems, who could use it to decide which panels need cleaning or repair first. It would support, not replace, a technician's inspection.
 
**Dataset.** The PV Panel Defect Dataset on Kaggle (alicjalena): **6 classes and about 14,999 images**. It combines the Kaggle dataset "Solar Panel Images: Clean and Faulty Images" with extra images collected from public web sources, and the rights to the individual images remain with their original owners, so I will use it for this course project only. Its description lists a ready-made split of 11,982 train, 1,501 validation and 1,516 test images, with between 2,169 (snow-covered) and 2,833 (bird-drop) images per class, so the largest class is only about 1.3 times the smallest. Links: <https://www.kaggle.com/datasets/alicjalena/pv-panel-defect-dataset> (main) and <https://www.kaggle.com/datasets/pythonafroz/solar-panel-images> (the smaller original source). I will confirm the exact counts and the licence on the Kaggle page and update this README.
 
**Plan.**
- Use colour images (`GRAYSCALE = False`), resize to $224 \times 224$, and normalise with the ImageNet mean and standard deviation.
- **Check for duplicates first.** The images come from several sources, and a class with over 2,000 images that grows from a smaller original set may contain repeated or augmented copies. I will search for exact and near-duplicate images across the train, validation and test folders. If copies are found, I will remove them or re-split by group before training, so the test score is not inflated.
- Use the dataset's own train, validation and test folders if the duplicate check passes. Otherwise, build my own grouped split.
- Augment the training set only (flips, mild brightness and colour jitter, and small crops). Strong colour changes could turn a dusty panel into a clean-looking one, so I will keep them mild.
- Train a new 6-class head first (`Linear(512, 6)`, $512 \times 6 + 6 = 3{,}078$ trainable parameters, the same as in Week 5), then fine-tune `layer4` with two learning rates ($10^{-3}$ head, $10^{-4}$ `layer4`), keep the best validation epoch by macro-F1, and evaluate on the test set once.
- Report accuracy, macro-F1, per-class precision and recall, a confusion matrix and an error gallery, and compare with the Week 5 scene classifier.
**Expected challenges.**
- **Electrical damage may not be visible.** Hot spots, delamination and diode failures are internal defects that normally show up on thermal or electroluminescence images, so ordinary photos may not show them. I expect electrical-damage and physical-damage to be the hardest classes and will check their recall in the confusion matrix.
- **Duplicates and leakage.** If the same image appears in both train and test, the score will be too high, which is why the duplicate check comes first.
- **Web images are not field images.** A high score on this dataset does not show the model would work on photos from a real plant with different cameras, angles and lighting, and I will say so when I report the results.
**Backup datasets I considered.** If this dataset turns out to be unsuitable (licence, duplicates or missing labels), I will switch to one of these. Counts are taken from the dataset descriptions and will be re-checked in my notebook.




  ## Week 6: Solar-panel condition classification with ResNet-18 (trained from scratch)

**Who benefits:** solar-farm and rooftop operators and maintenance crews who must decide *which panels to clean or repair first*, working from photographs taken on site (roof, ground or drone).
**Why it matters:** soiling (dust, bird droppings, snow) lowers energy output, while electrical or physical damage can mean permanent loss of capacity and a fire or shock risk. Inspecting thousands of panels by eye is slow, so an image classifier could help rank panels for a human inspector.

---

### Dataset

| Item | Value |
|---|---|
| Name | *Solar Panel Images – Clean and Faulty Images* (Afroz, 2023), Kaggle: `pythonafroz/solar-panel-images` |
| Link | https://www.kaggle.com/datasets/pythonafroz/solar-panel-images |
| License | **Not verified in this project – confirm on the Kaggle dataset page before publishing.** |
| Labels by | **Not documented in the notebook – confirm on the dataset page.** Labels are the folder names; no inter-annotator information was found. |
| Images / classes | **885** images, **K = 7** classes (folders under `Faulty_solar_panel/`) |
| Image properties | Colour photos (sample of 300: 291 RGB + 9 RGBA); width 149–5,472 px (median 720), height 110–5,376 px (median 600) |
| Collection | Mixed web/phone-style photos; stock-photo watermarks (e.g. *dreamstime*, *shutterstock*, *EasyEcoBlog*, *DIXTA*) are visible on some images |

| Class | Raw images | Share | After exact-duplicate removal | Train | Val (clean) | Test (clean) |
|---|---|---|---|---|---|---|
| Bird-drop | 191 | 21.6 % | 143 | 99 | 16 | 16 |
| Clean | 193 | 21.8 % | 186 | 130 | 30 | 23 |
| Dusty | 190 | 21.5 % | 182 | 127 | 30 | 25 |
| Electrical-damage | 103 | 11.6 % | 91 | 62 | 12 | 11 |
| New | 16 | 1.8 % | 16 | 10 | 3 | 3 |
| Physical-Damage | 69 | 7.8 % | 63 | 43 | 6 | 7 |
| Snow-Covered | 123 | 13.9 % | 113 | 80 | 16 | 13 |
| **Total** | **885** | 100 % | **794** | **551** | **113** | **98** |

---

### Data audit

**Imbalance ratio**

$$\text{IR}=\frac{n_{\max}}{n_{\min}}=\frac{193}{16}=12.1\quad(\text{raw}),\qquad \frac{186}{16}=11.6\quad(\text{after de-duplication})$$

**Groups.** File names carry no panel or site id, so every image is its own group:

$$\frac{N_{\text{images}}}{N_{\text{groups}}}=\frac{885}{885}=1.00$$

**Exact duplicates (MD5).** Only files with a repeated size were hashed (169 of 885 = 19.1 %).

$$\text{redundant copies}=\underbrace{165}_{\text{duplicate files}}-\underbrace{74}_{\text{identical sets}}=91\ (10.3\,\%),\qquad 885-91=794$$

Two identical-file sets carried **different labels** (label conflict). Bird-drop lost 48 of 191 images (25.1 %) – the most duplicated class.

**Near-duplicates (fingerprints).** Each image is reduced to a standardised 32 × 32 grayscale thumbnail $z$ and compared with

$$d(i,j)=\frac{1}{1024}\sum_{p=1}^{1024}\bigl|z_{i,p}-z_{j,p}\bigr|,\qquad d<0.10\Rightarrow\text{near-duplicate}$$

| Check (against train) | Flagged | Fraction | After removal |
|---|---|---|---|
| Test images with a near-duplicate in train | **12 of 110** | 10.9 % | 98 test images |
| Validation images with a near-duplicate in train | **20 of 133** | 15.0 % | 113 validation images |

Closest test distances: 0.005, 0.008, 0.008, 0.010, 0.014, 0.014, 0.018, 0.019, 0.023, 0.025 (a re-compressed copy scores about 0.007; different images ≥ 0.12). The six closest pairs were inspected side by side and were the same photograph.

**Split.** `StratifiedGroupKFold(n_splits=6)`, grouped by image path (no panel/site id exists): 794 → train 551 / val 133 / test 110, then 32 near-duplicates removed from val/test. **Groups shared by train and val: 0; by train and test: 0.**
Because each image is its own group, a group split only separates *files*; the MD5 and fingerprint checks above are what protect against the same photo appearing on both sides. Near-duplicates *inside* the training set were not removed.

---

### Imbalance handling

**Weighted cross-entropy** (`F.cross_entropy(..., weight=w)`), with class weights

$$w_c=\frac{N}{K\,n_c},\qquad L=-\frac{1}{\sum_i w_{y_i}}\sum_i w_{y_i}\log p_{i,y_i},\qquad n_c\,w_c=\frac{N}{K}=\frac{551}{7}=78.71$$

| Class | $n_c$ (train) | $w_c = 551/(7\,n_c)$ |
|---|---|---|
| Bird-drop | 99 | 551/693 = **0.795** |
| Clean | 130 | 551/910 = **0.605** |
| Dusty | 127 | 551/889 = **0.620** |
| Electrical-damage | 62 | 551/434 = **1.270** |
| New | 10 | 551/70 = **7.871** |
| Physical-Damage | 43 | 551/301 = **1.831** |
| Snow-Covered | 80 | 551/560 = **0.984** |

Sampler not used (`USE_SAMPLER = False`). Augmentation (train only): random resized crop (scale 0.8–1.0), horizontal and vertical flip, rotation ±20°, brightness/contrast jitter 0.15; hue and saturation are *not* changed because colour carries the label. Validation/test: resize to 224 × 224 and ImageNet normalisation.

---

### Model and training

| Item | Value |
|---|---|
| Backbone | ResNet-18, final layer replaced by `Linear(512, 7)` |
| Weights | **Random initialisation** – no ImageNet weights were available (no local file, Internet off) |
| Trainable parameters | $11{,}689{,}512-513{,}000+3{,}591 = 11{,}180{,}103$ |
| Optimiser / schedule | Adam, lr = 1e-3, 15 epochs, batch 32, input 224 × 224, CPU |
| Model selection | Best epoch by **validation macro-F1** (not accuracy) |
| Time | ≈ 76 s per epoch → 15 × 76 s ≈ 19 min |

$$F1_c=\frac{2P_cR_c}{P_c+R_c},\qquad \text{macro-F1}=\frac{1}{K}\sum_{c=1}^{K}F1_c$$

Validation macro-F1 per epoch: 0.067, 0.158, 0.241, 0.301, 0.439, 0.380, 0.310, 0.311, 0.416, **0.477 (epoch 10, selected)**, 0.468, 0.322, 0.452, 0.449, 0.335. The curve is noisy (113 validation images; 3 of them `New`).

---

### Results (test, used once; n = 98)

| Class | Precision | Recall (sensitivity) | F1 | Support |
|---|---|---|---|---|
| Bird-drop | 0.600 | 0.375 | 0.462 | 16 |
| Clean | 0.800 | 0.348 | 0.485 | 23 |
| Dusty | 0.654 | 0.680 | 0.667 | 25 |
| Electrical-damage | 0.538 | 0.636 | 0.583 | 11 |
| **New** | 0.000 | 0.000 | **0.000** | 3 |
| Physical-Damage | 0.250 | 0.143 | 0.182 | 7 |
| Snow-Covered | 0.524 | 0.846 | 0.647 | 13 |

$$\text{Accuracy}=\frac{6+8+17+7+0+1+11}{98}=\frac{50}{98}=0.510$$

$$\text{Macro-F1}=\frac{0.462+0.485+0.667+0.583+0+0.182+0.647}{7}=\frac{3.026}{7}=\mathbf{0.432}$$

$$\text{Weighted-F1}=\frac{\sum_c n_c F1_c}{N}=0.524$$

| Summary | Value |
|---|---|
| **Macro-F1** | **0.432** (validation best: 0.477) |
| Accuracy | 0.510 (random guess = 1/7 = 0.143; always “Dusty” = 25/98 = 0.255) |
| Macro precision / macro recall | 0.481 / 0.433 |
| Weakest class | **New** (0 of 3 found; 14 false `New` predictions), then Physical-Damage (1 of 7) |
| Strongest class | Snow-Covered (recall 0.846), Dusty (F1 0.667) |

**Confusion matrix** (rows = true class, columns = predicted):

| True \ Pred | Bird-drop | Clean | Dusty | Electrical | New | Physical | Snow |
|---|---|---|---|---|---|---|---|
| Bird-drop | **6** | 0 | 6 | 0 | 3 | 0 | 1 |
| Clean | 1 | **8** | 0 | 3 | 4 | 3 | 4 |
| Dusty | 1 | 1 | **17** | 1 | 4 | 0 | 1 |
| Electrical-damage | 0 | 0 | 0 | **7** | 2 | 0 | 2 |
| New | 0 | 1 | 1 | 1 | **0** | 0 | 0 |
| Physical-Damage | 2 | 0 | 1 | 0 | 1 | **1** | 2 |
| Snow-Covered | 0 | 0 | 1 | 1 | 0 | 0 | **11** |

**Binary sensitivity/specificity at a threshold:** *not applicable* – this is a 7-class task, so no recall-driven threshold was set. The equivalent one-vs-rest numbers are below.

$$\text{Sens}_c=\frac{TP_c}{TP_c+FN_c},\quad \text{Spec}_c=\frac{TN_c}{TN_c+FP_c},\quad \text{PPV}_c=\frac{\text{Sens}\cdot\text{prev}}{\text{Sens}\cdot\text{prev}+(1-\text{Spec})(1-\text{prev})}$$

| Class | Sensitivity | Specificity | Test prevalence | PPV at test prevalence | PPV at 5 % prevalence* |
|---|---|---|---|---|---|
| Bird-drop | 0.375 | 0.951 | 0.163 | 0.600 | 0.288 |
| Clean | 0.348 | 0.973 | 0.235 | 0.800 | 0.407 |
| Dusty | 0.680 | 0.877 | 0.255 | 0.654 | 0.225 |
| Electrical-damage | 0.636 | 0.931 | 0.112 | 0.538 | 0.327 |
| New | 0.000 | 0.853 | 0.031 | 0 (no true positives) | 0 |
| Physical-Damage | 0.143 | 0.967 | 0.071 | 0.250 | 0.186 |
| Snow-Covered | 0.846 | 0.882 | 0.133 | 0.524 | 0.275 |

\*5 % is an illustrative prevalence, not a measured fault rate. Worked example (Dusty, sens 0.680, spec 0.877):

$$\text{PPV}_{25.5\%}=\frac{0.680(0.255)}{0.680(0.255)+0.123(0.745)}=0.654,\qquad
\text{PPV}_{5\%}=\frac{0.680(0.05)}{0.680(0.05)+0.123(0.95)}=0.225$$

At a low prevalence most “Dusty” alerts would be false alarms, because the many non-dusty panels produce more false positives than the few dusty panels produce true positives.

---

### Grad-CAM

Grad-CAM on the last block (`layer4`, 512 maps of 7 × 7, up-sampled to 224 × 224):

$$\alpha_k^{c}=\frac{1}{Z}\sum_{i,j}\frac{\partial y^{c}}{\partial A^{k}_{ij}},\qquad L^{c}=\operatorname{ReLU}\Bigl(\sum_k \alpha_k^{c}A^{k}\Bigr),\qquad Z=7\times7=49$$

Sixteen heatmaps were read (6 + 10; the first six are all Bird-drop because the test table is sorted by class, so a second figure samples one correct image per class plus four errors).

| Group | Where the model looks | Judgement |
|---|---|---|
| Correct Bird-drop, Electrical-damage, Physical-Damage, Snow-Covered | On the soiled patch, damaged cells, cracked region, or the snow itself | Legitimate evidence |
| Correct Dusty | Mostly the panel surface, weak heat on horizon/sky | Mostly evidence |
| Correct **Clean** | **Hottest region is the sky above the panels** | **Shortcut suspected** |
| Wrong (Bird-drop → Dusty) | Whole soiled panel surface | Plausible confusion |
| Wrong (Bird-drop → New / Snow-Covered) | Frame edges, top-of-array band or sky/roof edge, not the bird or droppings | Background/edge cues, not evidence |

**Shortcut risk: suspected, not confirmed.** Stock-photo watermarks are present on some images and sit near hot regions in at least one heatmap. Planned test: mask or crop the sky and watermark areas of the test images and re-score; a large drop in `Clean` recall (currently 8/23 = 0.348) or macro-F1 would confirm the shortcut. Grad-CAM is coarse (each cell covers 32 × 32 px) and this network was trained from scratch on 551 images, so the maps are used to *detect* suspicious behaviour, not to localise faults.

---

### Limitations

* **Data source and size:** 885 web-style photos (794 unique); labels not documented; 2 duplicate sets with conflicting labels; `New` has only 16 images (3 in test) and several sampled `New` panels look dirty, so its label is questionable.
* **No panel/site ids:** group splitting cannot prevent the same panel appearing in train and test under different photos; near-duplicates inside train were not removed.
* **Model:** trained from scratch (no pretrained weights), CPU only, one run, no repeated seeds; validation (113) and test (98) are small, so one image moves accuracy by about 1 percentage point and F1 for small classes by tens of points.
* **Population and conditions:** unknown regions, climates, panel types and cameras; no thermal or night images; no external validation set.

**Not for:** safety or maintenance decisions made without a human, certifying panels as healthy, or any deployment on sites or imagery different from this dataset.

---

### Responsible-AI answers

| Question | Answer for this model |
|---|---|
| Who could be harmed, and how? | Operators and the public: a **damaged panel labelled Clean or Dusty** (recall 0.636 for electrical damage, 0.143 for physical damage) stays in service, risking lost energy and fire/shock hazards; false alarms waste crew time. |
| Where was it NOT validated? | Other countries and climates, other panel types, drone/thermal/night imagery, other cameras; there is no external test set. |
| What is the human role? | Triage only: rank panels or images for a technician to inspect; a person makes every maintenance and safety decision. |
| What would change my mind? | A pretrained (ImageNet) re-run with clearly higher macro-F1, a sky/watermark masking test showing the model does not depend on them, and external validation on images from a real site with PPV measured at that site's fault prevalence. |

---

### Reproducibility

Kaggle notebook, seed 42, `StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=42)`, PyTorch/torchvision ResNet-18, checkpoint `week6_model.pt` (epoch-10 weights, ≈ 44.7 MB: $11.18\text{ M}\times4$ bytes).

*Next steps:* turn Internet on (or add ResNet-18 weights) to run the planned two-phase fine-tuning, use a GPU, resolve the label-conflict duplicates, test `USE_SAMPLER = True`, and run the masking experiment.
