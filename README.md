# Project-2_Deep-_Learning_-Applications

## Project 2, Week 5: Scene Classification with Transfer Learning

**Author:** Raja Fahad Kiyani | **Course:** Introduction to Applied AI (BS 8th Semester, Department of Electronics) | **Platform:** Kaggle notebook

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

## 7. Week 6 proposal

[One paragraph: the problem, who benefits, the dataset link, and the number of classes and images.]
