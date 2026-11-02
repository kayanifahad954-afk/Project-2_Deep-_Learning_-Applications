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


## Week 7: Text and Multimodal AI

Project 2 (Deep Learning Applications) · Introduction to Applied AI · Domain: **solar-panel condition monitoring** (Week 6 ResNet-18 image model + text notes)

- **Text classifier:** SMS Spam Collection, 5,572 texts, macro-F1 **0.974** (word) vs **0.982** (char)
- **Top positive / negative features:** `call, txt, free, text, reply` / `my, me, ok, that, it`
- **NER on my domain:** out-of-the-box held-out F1 **0.370** → with entity ruler **0.759** (provided set: 0.533 → 0.875)
- **Regex fields extracted:** `cnic, mobile, amount, date, email, invoice, panel_id, serial, power_wp, voltage`; false positives tested: 2 provided + 4 solar negative tests, all clean
- **OCR:** CER raw **0.000** → after correction rules **0.000** on clean synthetic images (unit tests: 0.143 → 0.000)
- **Pipeline:** image model + text evidence → JSON record (example below)
- **Privacy:** CNIC, phone, and email masked before logging

---

## Results at a glance

| Component | Metric | Before | After / result |
|---|---|---|---|
| TF-IDF by hand | d1 weights, hand vs library | n/a | identical: win 0.584, free 0.445, cash 0.584, now 0.345 |
| Text classifier (word 1-2 grams) | macro-F1 | baseline "always ham" accuracy 0.866 | **0.974** |
| Text classifier (char 2-5 grams) | macro-F1 | 0.974 (word) | **0.982** |
| NER, provided set (16 gold entities) | entity-level F1 | 0.533 | **0.875** (7 rules) |
| NER, solar domain, held-out (15 gold entities) | entity-level F1 | 0.370 | **0.759** (4 rules) |
| Regex | tests passed | n/a | 5/5 provided, 6/6 solar |
| OCR repair, unit tests | CER | 0.062 / 0.059 / 0.143 | 0.000 / 0.000 / 0.000 |
| OCR, EasyOCR on synthetic images | CER | 0.000 | 0.000 |
| Case record | decision | n/a | REFER, P(Electrical-damage) = 0.803, PII masked |

> Every component is reported as a number. The held-out NER score is the honest one: the rules were written from sentences 1-5 and scored on sentences 6-10.

---

## 1. TF-IDF by hand

scikit-learn uses smooth IDF followed by L2 normalisation:

$$\mathrm{idf}(t)=\ln\frac{1+N}{1+\mathrm{df}(t)}+1,\qquad \mathrm{tfidf}(t,d)=\mathrm{tf}(t,d)\cdot\mathrm{idf}(t),\qquad \hat v_d=\frac{v_d}{\lVert v_d\rVert_2}$$

Corpus: `d1 = "win free cash now"`, `d2 = "call me now"`, `d3 = "free call now"` ($N=3$). Hand calculation for d1 (each term occurs once, so tf = 1):

| term | df | idf | idf² | normalised value | library value |
|---|---|---|---|---|---|
| win | 1 | ln(4/2)+1 = 1.6931 | 2.8667 | 1.6931 / 2.8968 = **0.584** | 0.584 |
| free | 2 | ln(4/3)+1 = 1.2877 | 1.6581 | 1.2877 / 2.8968 = **0.445** | 0.445 |
| cash | 1 | ln(4/2)+1 = 1.6931 | 2.8667 | 1.6931 / 2.8968 = **0.584** | 0.584 |
| now | 3 | ln(4/4)+1 = 1.0000 | 1.0000 | 1.0000 / 2.8968 = **0.345** | 0.345 |

Vector length: $\lVert v\rVert=\sqrt{2.8667+1.6581+2.8667+1}=\sqrt{8.3916}=2.8968$.

"now" gets the lowest weight because it occurs in every message: $\ln(4/4)=0$, so only the smoothing "+1" is left.

Cosine similarity (vectors already have length 1, so it is the dot product), shared terms "call" and "now":

$$\cos(d_2,d_3)=0.548\times0.620+0.425\times0.481=0.340+0.204=\mathbf{0.544}\approx0.54\ \text{(slide 10)}$$

---

## 2. Text classification: word vs character n-grams

**Data.** SMS Spam Collection (Almeida et al.), 5,572 messages: 4,825 ham and 747 spam.

$$\text{spam share}=\frac{747}{5572}=0.134,\qquad \text{always-ham accuracy}=\frac{4825}{5572}=0.866$$

Accuracy alone is therefore misleading, so **macro-F1** is the headline metric. Split: 80/20 stratified, `random_state=42` → 4,457 train, 1,115 test (149 spam, 966 ham).

**Models.** TF-IDF + Logistic Regression with balanced class weights.

$$P(\text{spam}\mid x)=\sigma(w\cdot x+b)=\frac{1}{1+e^{-(w\cdot x+b)}},\qquad \mathrm{tf}_{\text{sublinear}}=1+\ln(\mathrm{tf}),\qquad w_c=\frac{n}{2\,n_c}$$

Class weights from the training set (ham 3,859, spam 598): $w_{\text{ham}}=\frac{4457}{2\cdot3859}=0.577$, $w_{\text{spam}}=\frac{4457}{2\cdot598}=3.727$.

**Metrics** (spam is the positive class):

$$P=\frac{TP}{TP+FP},\quad R=\frac{TP}{TP+FN},\quad F_1=\frac{2PR}{P+R},\quad \text{macro-}F_1=\frac{F_1^{\text{ham}}+F_1^{\text{spam}}}{2}$$

| Model | Features | Macro-F1 | Spam precision | Spam recall | False positives | Missed spam |
|---|---|---|---|---|---|---|
| Word 1-2 grams (`min_df=2`) | 12,128 | 0.974 | 0.979 | 0.933 | 3 | 10 |
| Char 2-5 grams (`char_wb`, `min_df=2`) | 36,449 | **0.982** | **0.986** | **0.953** | 2 | 7 |

Worked calculation, word model ($TP=149-10=139$, $TN=966-3=963$):

$$P_{\text{spam}}=\tfrac{139}{142}=0.979,\quad R_{\text{spam}}=\tfrac{139}{149}=0.933,\quad F_1^{\text{spam}}=0.955,\quad F_1^{\text{ham}}=0.993,\quad \text{macro-}F_1=\tfrac{0.993+0.955}{2}=0.974$$

Worked calculation, char model ($TP=142$, $TN=964$):

$$P_{\text{spam}}=\tfrac{142}{144}=0.986,\quad R_{\text{spam}}=\tfrac{142}{149}=0.953,\quad F_1^{\text{spam}}=0.969,\quad F_1^{\text{ham}}=0.995,\quad \text{macro-}F_1=\tfrac{0.995+0.969}{2}=0.982$$

The char model gains **+0.008** macro-F1 for about $36449/12128\approx3.0\times$ more features. A word gives $\sum_{n=2}^{5}(L-n+1)=5+4+3+2=14$ character n-grams when padded to length $L=6$ (e.g. `" free "`), so misspellings such as `FR33` still share pieces with `free`.

### What the word model learned

A coefficient is a change in log-odds, so a present feature multiplies the odds of spam by $e^{w_j}$.

| Direction | Top n-grams | Reading |
|---|---|---|
| Spam-like | `call, txt, free, text, reply, uk, to, www, mobile, claim, stop, your` | mostly sensible (calls to action, premium-rate vocabulary); `to` and `your` are likely shortcuts (common function words in long promotional texts) |
| Ham-like | `my, me, ok, that, it, but, ll, gt, lt, can, what, da` | personal chat words; `gt` / `lt` are leftovers of HTML escapes (`&gt;`, `&lt;`) in the data, not meaning |

### Error analysis (word model: 13 errors = 10 missed spam + 3 false alarms; error rate 13/1115 = 1.17%)

| Verdict | Count | Examples |
|---|---|---|
| Model error | 7 | "ringtoneking 84484", "FreeMsg Hey there darling...", "Download as many ringtones...", "Waiting for your call." (the word "call" is a shortcut), "Ok... Ur typical reply..." |
| Label error | 1 | "Do you realize that in about 40 years... tattoos?" labelled spam |
| Genuinely ambiguous | 5 | warning about scam texts, "arsenal dartboard" ad, callback "I'm on 07090201529", joke-news subscription, recruiter-style text labelled ham |

### Tricky messages (P(spam); threshold 0.5)

| Message | Word | Char | Outcome |
|---|---|---|---|
| `W1N a FR33 pr1ze!! txt CLAIM now` (lab example) | n/a | 0.85 | caught, log-odds $\ln\frac{0.85}{0.15}=+1.73$ |
| `kal milte hain, 5 baje` (Roman Urdu ham, lab example) | n/a | 0.12 | correct, log-odds $-1.99$ |
| Roman Urdu spam: "Mubarak ho! Aap ne 50000 ka inam jeeta hai..." | 0.63 | 0.49 | word barely catches it, char barely misses it |
| Leetspeak spam: `Y0u h4ve w0n a fr33 g1ft c@rd!!...` | 0.17 | 0.42 | both miss, char is closer |
| Roman Urdu ham: "Kal office aa jana, meeting 5 baje hai" | 0.13 | 0.09 | both correct |

Character features help with disguised spelling, but neither model reliably catches Roman Urdu spam: the dataset is English and has almost no Roman Urdu. The fix is labelled Roman Urdu and obfuscated examples, not just different features.

---

## 3. Named entity recognition, measured

An entity counts as correct only if its **exact text and label** both match the gold annotation. With predicted set $\hat E$ and gold set $G$:

$$TP=|\hat E\cap G|,\quad FP=|\hat E\setminus G|,\quad FN=|G\setminus\hat E|,\quad F_1=\frac{2TP}{2TP+FP+FN}$$

### Provided Pakistani sentences (6 sentences, 16 gold entities)

| Pipeline | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| spaCy `en_core_web_sm` out of the box | 8 | 6 | 8 | 0.571 | 0.500 | **0.533** |
| + entity ruler (7 patterns) | 14 | 2 | 2 | 0.875 | 0.875 | **0.875** |

$$F_1^{\text{before}}=\frac{2\cdot8}{2\cdot8+6+8}=\frac{16}{30}=0.533,\qquad F_1^{\text{after}}=\frac{2\cdot14}{2\cdot14+2+2}=\frac{28}{32}=0.875,\qquad \Delta F_1=+0.342$$

Typical out-of-the-box failures: `Engro Fertilizers` → PERSON (should be ORG), `30/09/2026` → GPE (should be DATE), `0300-1234567` → DATE (a phone number, a job for regex), `250,000` → CARDINAL (should be MONEY including "Rs."), `Farmer Muhammad Aslam` → PERSON including the title. The model was trained mostly on US/European news, so local names, formats and currency are out of distribution.

### My domain: solar-panel sentences (10 sentences)

Rules (`Jinko Solar`, `K-Electric`, `Huawei`, `Quaid-e-Azam Solar Park`, plus the MONEY and dd/mm/yyyy rules) were written from sentences 1-5 only; sentences 6-10 are held out.

| Pipeline | Dev (sentences 1-5, 16 gold) | Held-out (sentences 6-10, 15 gold) |
|---|---|---|
| spaCy out of the box | 0.485 | 0.370 |
| + entity ruler | 0.848 | **0.759** |

Held-out detail after rules: 4 gold entities missed and 3 spurious, so $TP=15-4=11$, $FP=3$, $FN=4$:

$$P=\frac{11}{14}=0.786,\quad R=\frac{11}{15}=0.733,\quad F_1=\frac{22}{22+3+4}=0.759,\quad \Delta F_1=0.759-0.370=+0.389$$

The dev score is higher than the held-out score ($0.848>0.759$) because the rules were written from the dev sentences, which is why only the held-out number is reported as the result.

| Remaining error after rules | What happened | Best fix |
|---|---|---|
| `Multan Solar Farm` | labelled ORG instead of FAC | add to the gazetteer / entity ruler |
| `Technician Sana Iqbal` | PERSON span includes the title, so the exact text mismatches | rule that strips titles, or a model trained on my data |
| `Bilal Raza` | not found at all | a model trained on local names (rules never cover every new name) |
| `Peshawar` | labelled ORG instead of GPE | gazetteer of cities |

Choice of fix: **names and facilities** → entity ruler / gazetteer (cheap and exact, but only covers listed names); **fixed formats** (phone, CNIC, dates) → regex; **open-ended person and place names** → a model trained on my own annotated data once I have a few hundred sentences. Ten sentences are far too few to train a model, so rules and regex are the right tools today.

---

## 4. Regex fields with negative tests

| Field | Pattern | Idea |
|---|---|---|
| `cnic` | `\b\d{5}-\d{7}-\d\b` | exactly 5-7-1 digits with dashes |
| `mobile` | `(?:\+92[- ]?\|\b0)3\d{2}[- ]?\d{7}\b` | `+92` or leading `0`, then 3 + 2 digits, then exactly 7 digits |
| `amount` | `(?:PKR\|Rs\.?)\s?([\d,]+(?:\.\d{1,2})?)` | currency word, digits and commas, optional 1-2 decimals |
| `date` | `\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b` | d/m/yyyy or d-m-yyyy |
| `email` | `\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b` | local part, `@`, dotted domain |
| `invoice` | `\b(?:Invoice\s+)?No\.?\s*(\d{3,})\b` | "No." followed by 3 or more digits |
| `panel_id` (mine) | `\bSP-\d{4}\b` | `SP-` plus exactly 4 digits |
| `serial` (mine) | `\b[A-Z]{3}\d{7}\b` | 3 letters + 7 digits |
| `power_wp` (mine) | `\b\d{3,4}\s?Wp?\b` | 3-4 digits, optional space, `W` or `Wp` |
| `voltage` (mine) | `\b\d{2,3}(?:\.\d{1,2})?\s?V\b` | 2-3 digits, optional decimals, `V` |

| Test | Must find | Result |
|---|---|---|
| `CNIC 35202-1234567-1, mobile +92 300 1234567` | CNIC, mobile | PASS |
| `Total Rs. 12,500.50 paid on 05/10/2026` | amount 12,500.50, date | PASS |
| `Invoice No. 4417 from info@engro.com.pk` | email, invoice 4417 | PASS |
| `Order 12345678901234 shipped` | nothing (not a CNIC) | PASS (`{}`) |
| `Room 0300 on floor 3` | nothing (not a mobile) | PASS (`{}`) |
| `Panel SP-1042 rated 550 Wp at 49.5 V, serial ABC1234567` | panel_id, power_wp, voltage, serial | PASS |
| `Inverter output fell to 220 V on SP-2201` | voltage, panel_id | PASS |
| `Room SP-12 on floor 3` | nothing (ID needs 4 digits) | PASS (`[]`) |
| `He ordered 500 Watts of 48 Volts power` | nothing (unit needs a word boundary) | PASS (`[]`) |
| `Call 2024 West Road, VIP gate 12` | nothing | PASS (`[]`) |
| `Order 12345678 shipped` | nothing (no 3-letter prefix) | PASS (`[]`) |

**Total: 5/5 provided tests and 6/6 solar tests pass (2 + 4 = 6 negative tests, all clean).**

**A pattern that matched too much.** My first panel-ID pattern `SP-\d+` returned `['SP-1042', 'SP-99', 'SP-12345678']`: it matched inside `ASP-99` (no left boundary) and swallowed an 8-digit number (no exact count). The fixed `\bSP-\d{4}\b` returns only `['SP-1042']`. The three tools were word boundaries, an exact digit count, and a fixed prefix.

---

## 5. Character error rate and OCR repair

Edit distance by dynamic programming, and CER as edits over reference length:

$$d_{i,j}=\min\big(d_{i-1,j}+1,\ d_{i,j-1}+1,\ d_{i-1,j-1}+\mathbb{1}[a_i\ne b_j]\big),\qquad \mathrm{CER}=\frac{S+D+I}{N}=\frac{\mathrm{lev}(\text{ref},\text{hyp})}{N}$$

`fix_ocr()` repairs look-alike characters only **inside numbers** (`O`→`0`, `l`/`I`→`1`) and `N0.` → `No.`

| Reference | OCR output | Edits | CER before | CER after fix |
|---|---|---|---|---|
| `Invoice No. 4417` | `Invoice N0. 4417` | 1 | 1/16 = **0.062** | 0.000 |
| `Total: PKR 12,500` | `Total: PKR 12,5O0` | 1 | 1/17 = **0.059** | 0.000 |
| `Due 30/09/2026` | `Due 3O/09/2O26` | 2 | 2/14 = **0.143** | 0.000 |

EasyOCR on two synthetic images (whitespace normalised so line breaks do not count):

| Document | Characters wrong | CER raw | CER after fixes | Field-level accuracy |
|---|---|---|---|---|
| Shared invoice (Engro Fertilizers Ltd) | 0 | 0.000 | 0.000 | 4/4 = 1.00 |
| My solar inspection label | 0 | 0.000 | 0.000 | 5/5 = 1.00 |

$$\text{field accuracy}=\frac{\#\{\text{fields whose extracted value equals the truth}\}}{\#\{\text{fields in the truth}\}}$$

**Interpretation.** The raw CER was already 0.000, so the repair rules had nothing to fix. A synthetic image is the easiest possible case: black text on white, one regular font, no blur, perspective, glare, shadows or JPEG noise. On a phone photo of a real invoice or panel nameplate I expect a much higher raw CER and a smaller gain from the simple "O → 0" rules. CER also understates the damage: one wrong digit makes a whole amount or ID wrong, which is why field-level accuracy is reported next to it.

---

## 6. Pipeline: image model + text evidence → JSON record

```
Image ─── Week 6 ResNet-18 (7 classes) ─────────────────┐
                                                         ├─► Fuse (transparent rules) ─► JSON record + reasons
Note ──► Mask PII FIRST ──► NER + regex + keywords ──────┘
```

**Image model.** ResNet-18, input 224×224, 7 classes: Bird-drop, Clean, Dusty, Electrical-damage, New, Physical-Damage, Snow-Covered. Softmax turns the 7 logits into probabilities:

$$p_k=\frac{e^{z_k}}{\sum_{j=1}^{7}e^{z_j}},\qquad \sum_k p_k=1$$

**Decision rule (late fusion by rules, threshold $\tau=0.5$, positive class = Electrical-damage, the class that must not be missed):**

$$\text{image flag}=\mathbb{1}[p\ge\tau],\qquad \text{text flag}=\mathbb{1}[\text{urgent terms}\ne\varnothing],\qquad \text{REFER if any flag is set; flag for human review if the flags differ}$$

Urgent solar terms: `hot spot, burn, spark, smoke, melt, short circuit, spreading, crack, no output, shock`. When labelled image-text pairs are available, a weighted fusion $p=w\,p_{\text{img}}+(1-w)\,p_{\text{txt}}$ can replace the rules, with $w$ chosen on validation data (e.g. $w=0.5$, $p_{\text{img}}=0.30$, $p_{\text{txt}}=0.80$ gives $p=0.55$).

**Privacy.** The note is masked **before** any extraction, so nothing downstream ever holds the identifier:

| Input | Output |
|---|---|
| `CNIC 35202-1234567-1, call 0300-1234567, a@b.pk` | `CNIC [CNIC], call [PHONE], [EMAIL]` |
| `Room 0300 on floor 3, panel SP-1042, order 12345678901234` | unchanged (0 false masks) |

An assertion checks that no phone or CNIC digit appears anywhere in the final JSON (`Privacy check passed`).

### Example record

```json
{
  "image": "Electrical (1).jpg",
  "image_probabilities": {
    "Bird-drop": 0.04, "Clean": 0.009, "Dusty": 0.021,
    "Electrical-damage": 0.803, "New": 0.049,
    "Physical-Damage": 0.071, "Snow-Covered": 0.007
  },
  "note_masked": "Technician Hamza Malik inspected the rooftop array at Gulberg Lahore on 12/09/2026. Burn marks and a hot spot near panel SP-1042 (550 Wp) are spreading. Owner CNIC [CNIC], mobile [PHONE].",
  "text_evidence": {
    "entities": [["Technician Hamza Malik", "PERSON"], ["Gulberg Lahore", "FAC"],
                 ["12/09/2026", "DATE"], ["550", "CARDINAL"],
                 ["CNIC", "ORG"], ["CNIC", "ORG"]],
    "fields": {"date": ["12/09/2026"], "panel_id": ["SP-1042"], "power_wp": ["550 Wp"]},
    "urgent_terms": ["hot spot", "burn", "spreading"]
  },
  "modalities_agree": true,
  "decision": "REFER",
  "reasons": [
    "image model: P(Electrical-damage) = 0.80",
    "note mentions: hot spot, burn, spreading"
  ],
  "disclaimer": "Decision support only; a qualified human decides."
}
```

**Calculation check.** Probabilities sum to $0.040+0.009+0.021+0.803+0.049+0.071+0.007=1.000$. Since $0.803\ge0.5$ the image flag is set, and three urgent terms set the text flag. Both agree, so the decision is REFER with two reasons.

**Wrong entities.** `CNIC` tagged ORG (twice, from the `[CNIC]` placeholder), `Technician Hamza Malik` (title inside PERSON), `Gulberg Lahore` as one FAC, `550` as CARDINAL. These are harmless here because the decision uses only the image probability and the urgent-term list; they would be harmful if a later step used ORG entities to look up or notify an organisation.

### Image and note disagree

Using a `Clean`-looking panel image:

| Note | Image flag | Text flag | Agree | Decision |
|---|---|---|---|---|
| "Routine quarterly check, nothing unusual..." | 0 | 0 | True | ROUTINE |
| "Smoke smell and a burn mark near the junction box..." | 0 | 1 (`burn, smoke`) | **False** | **REFER** (reason: "image and note DISAGREE: human review required") |

The system does not average the disagreement away: either the photo (bad angle, wrong panel) or the note (copy-paste, wrong site) is faulty, so a human checks both.

---

## Limitations and next steps

- The solar gold set has only 10 sentences (15 held-out entities), so one entity changes the held-out F1 by several points; a larger annotated set is needed before trusting the 0.759.
- The SMS data is English; Roman Urdu spam and leetspeak are not reliably caught (0.49 and 0.42 for the char model).
- OCR was tested only on clean synthetic images (CER 0.000); real phone photos and Urdu (Nastaliq) text still need measuring.
- Image/text fusion uses transparent rules, not learned weights, because there are no labelled image-text pairs to choose $w$ on.
- Decision support only: a qualified human makes the final call.

## How to run

1. Kaggle notebook, Internet **On** (EasyOCR downloads about 100 MB on first use).
2. Add the *SMS Spam Collection Dataset* (uciml), the private `week6-model` dataset containing `week6_model.pt`, and the Week 6 image dataset.
3. Run all cells; each part ends with a "Write it" cell and the notebook finishes with the Check Your Understanding answers.
