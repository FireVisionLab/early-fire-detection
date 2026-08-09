# Teslim Notu — Yağız'dan Ömer'e

Bu doküman Yöntem 2 (YOLO26) ve ortak altyapı tarafında yapılan her şeyi, mevcut kod tabanının tam durumunu, karşılaşılan hataları ve düzeltmelerini, ve birlikte yapmamız gerekenleri anlatır. Amaç: bu dosyayı okuyan biri (insan ya da AI), repoyu baştan taramadan projenin tam durumunu anlayıp EYT-Net tarafında devam edebilsin.

Proje: TOBB ETÜ BİL 468/587 yaz dönemi projesi, "CNN Tabanlı Erken Aşama İç Mekân Yangın ve Duman Tespiti". İki sınıf: `fire` (id=0), `smoke` (id=1). İki yöntem karşılaştırılıyor: Yöntem 1 = EYT-Net (Ömer, sıfırdan PyTorch CNN detector), Yöntem 2 = YOLO26 fine-tuning (Yağız).

## 1. Görev bölümü (referans için)

**Ömer:** EYT-Net mimarisi (backbone, feature fusion, detection head), CIoU/DIoU box loss + BCE loss, PyTorch eğitim döngüsü, confidence/NMS filtreleme, ortak değerlendirme modülü (P/R/F1/AP/mAP/PR eğrileri), hata analizi (false positive/negative).

**Yağız (ben):** Veri hazırlama/EDA, ortak ön işleme, augmentation, anchor k-means, eğitim altyapısı (logging/checkpoint), YOLO26 fine-tuning + hiperparametre deneyleri, demo/görselleştirme.

**Ortak:** Literatür taraması, karşılaştırma deneyi tasarımı, inference time ölçümü, nicel karşılaştırma, final rapor + sunum.

## 2. Veri seti

- Kaynak: Home Fire Dataset (Peng & Kim), `notebooks/01_setup.ipynb` içinde `kagglehub` ile `../data`'ya indiriliyor.
- Toplam 6500 görüntü: train 3900, val 1300, test 1300 (hazır ayrım, biz değiştirmedik).
- Format: YOLO `.txt` etiketleri, satır başına `class_id x_center y_center width height` (hepsi normalize `[0,1]`). `class_id`: `0=fire`, `1=smoke`.
- Görüntü/etiket bütünlüğü kontrol edildi: bozuk görüntü yok, geçersiz etiket satırı yok (`02_data_analysis.ipynb`).
- Sınıf dengesi: fire kutuları smoke'dan daha fazla (her split'te tutarlı şekilde).
- Kutu boyutu: kutuların ~%40'ı görüntü alanının <%1'i (küçük nesne — projenin ana zorluğu).
- 135 görüntüde boş etiket dosyası var (negatif örnek, ne fire ne smoke).
- Çözünürlükler değişken (tek bir sabit boyut yok) — bu yüzden ortak letterbox ön işleme gerekli.
- `data/dataset.yaml`:

```yaml
path: <repo>/data
train: train/images
val: val/images
test: test/images
nc: 2
names:
  0: fire
  1: smoke
```

Bu dosya hem YOLO26 (Ultralytics otomatik okuyor) hem de EYT-Net için ortak referans olmalı — split yolları ve sınıf isimleri buradan gelsin.

## 3. Ortak helper modülleri — tam API

### `helpers/preprocess.py`
```python
IMG_SIZE = 640

def resize_image(img: Image.Image, target_size: int = IMG_SIZE) -> tuple[Image.Image, float, int, int]:
    """Letterbox resize: aspect ratio korunur, gri (114,114,114) padding ile kareye tamamlanır.
    Döner: (canvas, scale, pad_x, pad_y)."""

def remap_bbox(xc, yc, w, h, orig_width, orig_height, scale, pad_x, pad_y, target_size=IMG_SIZE) -> tuple[float, float, float, float]:
    """Normalize YOLO kutusunu orijinal görüntüden letterbox canvas'a taşır. Girdi/çıktı hep normalize [0,1]."""

def normalize_image(img: Image.Image) -> np.ndarray:
    """PIL -> float32 tensor, (C,H,W), [0,1] aralığında."""

def preprocess(img: Image.Image, target_size: int = IMG_SIZE) -> tuple[np.ndarray, float, int, int]:
    """resize_image + normalize_image zinciri. Döner: (tensor, scale, pad_x, pad_y)."""

def load_yolo_boxes(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """.txt etiketini [(cls_id, xc, yc, w, h), ...] listesine çevirir."""
```
Doğrulanmış davranış: `train_1.jpg` (orijinal boyut farklı, letterbox sonrası) için `tensor.shape == (3,640,640)`, `dtype=float32`, `min/max=0.0/1.0`, örnek `scale=0.5, pad_x=0, pad_y=140`.

### `helpers/augmentations.py`
```python
def horizontal_flip(img, boxes, p=0.5) -> tuple[Image.Image, list]:
    """p olasılıkla sol-sağ flip, x_center'ı 1-xc yapar. boxes formatı [(cls_id,xc,yc,w,h)]."""

def limited_color_jitter(img, brightness=0.15, contrast=0.15, saturation=0.15) -> Image.Image:
    """Brightness/contrast/saturation'ı ±oran kadar rastgele değiştirir. Hue'ya dokunmaz. Kutuları etkilemez."""

def random_scale(img, boxes, scale_range=(0.8, 1.25), min_visibility=0.2) -> tuple[Image.Image, list]:
    """Görüntüyü scale_range'den rastgele bir s ile büyütüp/küçültür, SONRA orijinal (w,h) boyutuna
    gri padding/crop ile geri döndürür (letterbox ile zincirlendiğinde etkisi kaybolmasın diye).
    Kutular buna göre yeniden hesaplanır; görünür alanı orijinalin %20'sinden azına düşen kutular atılır."""

def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8,8)) -> Image.Image:
    """LAB uzayında L kanalına CLAHE uygular (düşük kontrastlı duman için). Opsiyonel, varsayılan kapalı."""

def augment_sample(img, boxes, use_clahe=False) -> tuple[Image.Image, list]:
    """horizontal_flip -> limited_color_jitter -> random_scale -> (opsiyonel) apply_clahe zinciri.
    Yalnızca train split'te çağrılmalı; val/test'te augmentation yok."""
```

**Önemli — bug geçmişi:** `random_scale`'in ilk hali sadece `img.resize(...)` yapıp orijinal boyutu değiştiriyordu; hemen ardından gelen letterbox (`resize_image`) bu ölçek farkını iptal ediyordu (çünkü letterbox scale = `min(640/w, 640/h)`, sonuçta içerik her zaman aynı orana oturuyordu). Düzeltme: `random_scale` artık scale uyguladıktan sonra görüntüyü **orijinal `(w,h)` boyutuna** crop/pad ile geri getiriyor, böylece letterbox'tan önce gerçek bir boyut/konum farkı kalıyor. Ayrıca kırpma sonucu orijinal kutu alanının %80'inden fazlasını kaybeden kutular (`min_visibility=0.2` eşiği) otomatik atılıyor — aksi halde neredeyse görünmez, gürültülü kutular etikette kalırdı.

**Kullanım sırası (önemli):** `augment_sample` çıktısı hâlâ orijinal görüntü boyutunda, normalize `[0,1]` kutular içerir — letterbox uygulanmamıştır. Doğru sıra:
```python
img, boxes = augment_sample(img, boxes)   # sadece train split
tensor, scale, pad_x, pad_y = preprocess(img)  # her split
boxes = [(c, *remap_bbox(xc, yc, w, h, img.width, img.height, scale, pad_x, pad_y)) for c, xc, yc, w, h in boxes]
```

### `helpers/viz.py`
```python
CLASS_IDS = {0: {"name": "fire", "color": "red"}, 1: {"name": "smoke", "color": "blue"}}

def draw_bbox(ax, label_path: Path, img_path: Path):
    """Etiket dosyasından okuyup matplotlib ax'ine kutu çizer."""

def draw_yolo_boxes(ax, boxes, img_w, img_h):
    """Bellekteki [(cls_id,xc,yc,w,h)] listesinden kutu çizer (augment sonrası görselleştirme için)."""
```

### `helpers/utils.py`
```python
def set_seed(seed=42):
    """random/numpy/torch (+cuda) seed'lerini sabitler."""

def make_run_dir(base_dir, run_name) -> Path:
    """<base_dir>/<run_name>/weights/ klasörünü oluşturur, run_dir döner."""

def save_checkpoint(path, model, optimizer, epoch, metric):
    """torch.save ile {epoch, metric, model.state_dict(), optimizer.state_dict()} kaydeder."""

def load_checkpoint(path, model, optimizer=None) -> dict:
    """Checkpoint'i yükler, model/optimizer state_dict'lerini set eder."""

class BestCheckpoint:
    """__init__(run_dir); .update(model, optimizer, epoch, metric) her çağrıda last.pt yazar,
    metric iyileşirse best.pt de yazar. Generic PyTorch training loop için (YOLO'ya bağımlı değil,
    EYT-Net eğitim döngün için hazır)."""

def append_metrics(csv_path, row: dict):
    """CSV'ye epoch başına satır ekler (header otomatik)."""

def yolo_log_kwargs(project, name, save_period=5) -> dict:
    """model.train(**yolo_log_kwargs(...)) şeklinde kullanılır; Ultralytics'in project/name/save_period
    ayarlarını standardize eder. YOLO26 tüm run'larımızda bunu kullandık."""

def read_yolo_metrics(run_dir) -> pd.DataFrame:
    """<run_dir>/results.csv'yi okur (Ultralytics'e özel, epoch bazlı loss/metric geçmişi)."""

def f1_score(p, r) -> float
def f2_score(p, r) -> float
    """Bu projede recall'a f1'den daha çok ağırlık veriyoruz (erken yangın tespitinde recall kritik),
    bu yüzden model seçimi F2'ye göre yapıldı."""
```
Not: `save_checkpoint`/`load_checkpoint`/`BestCheckpoint`/`append_metrics`/`read_yolo_metrics` şu an hiçbir notebook'ta çağrılmıyor — bunlar YOLO'ya değil, **senin custom PyTorch eğitim döngüne** hazırlanmış generic altyapı. `yolo_log_kwargs`, `set_seed`, `f1_score`, `f2_score` YOLO notebook'larında aktif kullanılıyor.

## 4. Anchor analizi (`configs/anchors.json`)

`notebooks/05_anchor_kmeans.ipynb`'de üretildi. Metodoloji:
- Sadece **train** split kutuları kullanıldı, 640x640 letterbox ölçeğinde piksele çevrildi (4812 kutu; genişlik 3.5–640px, yükseklik 5–360px).
- Klasik Öklid k-means yerine **IoU tabanlı k-means** (`1 - IoU` mesafesi) kullanıldı — küçük kutuların büyük kutular tarafından ezilmemesi için (YOLO/SSD literatüründeki standart yaklaşım).
- Küme merkezleri medyan ile güncellendi (ortalama değil — boyut dağılımı çarpık/skewed).
- `k=1..9` denendi, avg IoU sürekli arttı (elbow yok). EYT-Net'in **2 detection scale**'i olacağı için ölçek başına 3 anchor = **k=6** seçildi. `k=6`'da avg IoU ≈ 0.638.
- Sonuç, alana göre küçükten büyüğe sıralı, seed=42 ile tekrarlanabilir şekilde kaydedildi:

```json
{
  "img_size": 640,
  "seed": 42,
  "k": 6,
  "avg_iou": 0.6379215170496142,
  "anchors_px": [[9.5, 11.0], [21.0, 22.5], [36.0, 48.0], [69.5, 80.0], [110.5, 148.5], [241.5, 197.0]]
}
```

`anchors_px`, 640x640 letterbox sonrası **piksel** cinsinden. EYT-Net iki ölçekli olacağı için ilk 3'ü küçük/orta nesne başlığına, son 3'ü büyük nesne başlığına verilebilir — kesin stride ataması senin backbone tasarımına bağlı (`anchor_w / stride`, `anchor_h / stride` ile grid birimine çevir).

## 5. Yöntem 2 — YOLO26 fine-tuning (tam geçmiş)

Sırasıyla notebooklar ve neden var oldukları:

### `03_yolo_test_basic.ipynb` — ilk duman testi
10 epoch, sadece pipeline'ın çalıştığını doğrulamak için. `models/yolo26n/` altına kaydedildi (sonraki run'lardan izole).

### `06_yolo_finetune.ipynb` — asıl deneyler
- **`yolo26n_baseline`**: 50 epoch, `imgsz=640, batch=16, seed=42, close_mosaic=10`. Sonuç (val): P=0.942, R=0.898, mAP50=0.945, mAP50-95=0.645, F1=0.920, F2=0.906. Fire (R=0.912) smoke'dan (R=0.884) daha kolay.
- **`yolo26n_freeze8`**: backbone dondurulup (`freeze=10`) 8 epoch, yeni head'in adapte olması için (kurs notu / literatür önerisi).
- **`yolo26n_unfreeze50`**: `freeze8`'in best ağırlığından başlayıp `freeze=None, patience=15` ile 50 epoch tam eğitim. Sonuç baseline'a çok yakın, recall biraz daha iyi (F2=0.909 vs 0.906) → **hiperparametre tuning için başlangıç noktası olarak seçildi.**

### `07_hyperparameter_for_yolo.ipynb` — lokal tune denemesi (yarım kaldı)
`model.tune(epochs=25, iterations=30, optimizer="AdamW")` lokal GPU'da çok yavaş kaldığı için Colab'a taşındı.

### `07_hyperparameter__for_yolo_colab.ipynb` — genetik tune + ilk final run
- `unfreeze50` best ağırlığından, `model.tune(epochs=25, iterations=30, optimizer="AdamW")` → `models/runs/yolo26n_tune/best_hyperparameters.yaml` üretildi.
- Bulunan en iyi ağırlıklarla `yolo26n_tuned50` run'ı eğitildi (`epochs=50, cfg=best_hyperparameters.yaml`).
- **Bug tespit edildi:** Log'da `optimizer='auto' found, ignoring 'lr0=0.00295' and 'momentum=0.84846'` — yani tune edilen `lr0`/`momentum` fiilen uygulanmamış, sadece augmentation/loss ağırlıkları (`cfg=` üzerinden gelenler) kullanılmış.

### `07_hyperparameter_full_run.ipynb` — düzeltilmiş run
- Aynı başlangıç ağırlığından, bu sefer `optimizer="AdamW"` + `lr0/lrf/momentum/weight_decay/warmup_epochs/warmup_momentum` **açıkça** `model.train()`'e geçirilerek `yolo26n_tuned50_fixed` eğitildi.
- Sonuç: val F2 **0.904** (fixed) vs **0.921** (eski, "hatalı" tuned50). Recall düştü, mAP50 hemen hemen aynı kaldı.
- **Karar:** Düzeltme metodolojik olarak doğru uygulandı ("tuned hyp'leri gerçekten denedik mi?" sorusunun cevabı evet), ama F2 iyileşmediği için **final model olarak eski `yolo26n_tuned50` tutuldu** (val F2'ye göre model seçimi yapıldığı rapor edilecek).

### `08_demos.ipynb` — final demo ve test sonuçları
- Final model: `models/runs/yolo26n_tuned50/weights/best.pt`.
- Test seti (hiç dokunulmamış, sadece final ölçüm): **P=0.906, R=0.830, F2=0.845, mAP@0.5=0.897**, mAP@0.5:0.95 raporlanıyor.
- Confidence sweep **val** üzerinde yapıldı (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50) — test setine bakmadan eşik seçmek için. 0.10–0.30 arası F2 sabit (0.921), 0.40+ recall'ı düşürüyor. Varsayılan confidence korundu (sweep'ten kazanç yok).
- İnference hızı: 50 test görüntüsünde ortalama ≈1.7ms/görüntü (laptop RTX 4060 GPU) → gerçek zamanlı çalışabilirlik doğrulandı.
- Altı sabit test görüntüsünde pred-vs-GT görselleştirmesi var (`result.plot()` + `draw_bbox`).

## 6. Yöntem 1 — EYT-Net (senin tarafın, şu ana kadar var olan)

`notebooks/model_arch.ipynb` içinde şu an sadece başlangıç iskeleti var (muhtemelen senin attığın ilk taslak):

```python
class FireFeatureBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
    def forward(self, x):
        return self.block(x)
```
Test edildi: `input (1,3,416,416) -> output (1,32,208,208)` (stride=2 ile). Bunun ötesinde backbone/neck/head/loss/training loop henüz yok — proposal'daki Adım 2-7 hâlâ senin önünde.

## 7. Ortam / kurulum

- Python ≥3.11, `pyproject.toml` (uv ile yönetiliyor): `torch>=2.12`, `torchvision>=0.27`, `ultralytics>=8.4.56`, `pillow`, `matplotlib`, `pandas`, `kagglehub`.
- CUDA: `cu130` index tanımlı (`pyproject.toml`), GPU: RTX 4060 Laptop (8GB) lokal, Colab GPU tune için kullanıldı.
- Kurulum: `uv sync` → `.venv\Scripts\activate`. GPU doğrulama: `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.

## 8. Ömer'in doğrudan kullanabileceği hazır parçalar

| İhtiyaç | Nereden al |
|---|---|
| Ortak ön işleme | `from helpers.preprocess import preprocess, resize_image, remap_bbox, normalize_image, load_yolo_boxes` |
| Augmentation | `from helpers.augmentations import augment_sample, horizontal_flip, limited_color_jitter, random_scale, apply_clahe` |
| Görselleştirme | `from helpers.viz import draw_bbox, draw_yolo_boxes` |
| Checkpoint/logging altyapısı | `from helpers.utils import set_seed, save_checkpoint, load_checkpoint, BestCheckpoint, make_run_dir, append_metrics, f1_score, f2_score` |
| Anchor boyutları | `configs/anchors.json` → `anchors_px` (640 letterbox ölçeğinde piksel, alana göre sıralı) |
| Sınıf/split tanımı | `data/dataset.yaml` |

`helpers/dataset.py` **bilerek yazılmadı** — PyTorch `Dataset`/`DataLoader`, anchor-to-grid atama mantığına (senin detection head tasarımına) bağlı olduğu için bunu kendi eğitim döngünle birlikte tasarlaman daha doğru.

## 9. Birlikte yapacaklarımız (ortak görevler)

- [ ] **Tahmin çıktı formatı**: Ortak evaluator'ın her iki modelden de aynı formatta beklediği çıktıyı netleştirelim — öneri: `(image_id, class_id, confidence, x1, y1, x2, y2)`, piksel cinsinden, letterbox öncesi orijinal görüntü koordinatlarına geri map'lenmiş.
- [ ] **Confidence/NMS eşik seçim metodolojisi**: Ben YOLO için val üzerinde sweep yaptım (bkz. `08_demos.ipynb`); aynı yaklaşımı EYT-Net çıkarımında da val üzerinde uygula, test setine bakmadan.
- [ ] **Inference time ölçümü**: Aynı 50 test görüntüsü, aynı cihaz, aynı warm-up protokolüyle EYT-Net için de ölçüm yapalım.
- [ ] **Nicel karşılaştırma tablosu**: P/R/F1/mAP@0.5/mAP@0.5:0.95/inference time satırları; YOLO26 sütunu hazır (yukarıdaki sayılar), EYT-Net sütununu sen eğitimi bitirince dolduracağız.
- [ ] **Hata ve yanlış alarm analizi**: Lamba/ekran/yansıma/buhar gibi false positive kaynaklarına birlikte bakalım.
- [ ] **Final rapor ve sunum**: Ben veri hazırlığı, ön işleme/augmentation, anchor analizi ve YOLO26 bölümlerini yazıyorum; EYT-Net mimarisi/sonuçları bölümünü sen yazarsın, karşılaştırma ve sonuç bölümünü birlikte tamamlarız.

## 10. Dosya haritası

```
helpers/
  preprocess.py      -> ortak ön işleme (letterbox, normalize, bbox remap, label loader)
  augmentations.py   -> ortak augmentation (flip, jitter, scale, CLAHE)
  utils.py           -> seed, checkpoint/logging altyapısı, F1/F2
  viz.py             -> draw_bbox / draw_yolo_boxes
configs/
  anchors.json       -> k-means anchor sonuçları (EYT-Net detection head için)
data/
  dataset.yaml       -> split yolları, nc=2, class isimleri
notebooks/
  01_setup.ipynb                        -> veri indirme
  02_data_analysis.ipynb                -> EDA, bütünlük kontrolü
  03_yolo_test_basic.ipynb              -> ilk 10 epoch duman testi
  04_augmentation.ipynb                 -> augmentation geliştirme + görsel test
  05_anchor_kmeans.ipynb                -> anchor hesaplama
  06_yolo_finetune.ipynb                -> baseline + freeze8 + unfreeze50
  07_hyperparameter_for_yolo.ipynb      -> lokal tune denemesi (yarım, Colab'a taşındı)
  07_hyperparameter__for_yolo_colab.ipynb -> genetik tune + ilk (hatalı) final run
  07_hyperparameter_full_run.ipynb      -> düzeltilmiş final run + karar
  08_demos.ipynb                        -> final demo, test metrikleri, inference hızı
  model_arch.ipynb                      -> EYT-Net başlangıç iskeleti (FireFeatureBlock)
models/runs/
  yolo26n_baseline/, yolo26n_freeze8/, yolo26n_unfreeze50/,
  yolo26n_tune/, yolo26n_tuned50/ (final), yolo26n_tuned50_fixed/
```
