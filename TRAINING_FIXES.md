# Telescope — correcciones para entrenar acorde al paper

Revisión de consistencia con el paper (arXiv:2604.06332) y correcciones para que
el entrenamiento funcione correctamente. Estado vivo: marca lo hecho / pendiente.

## Contexto del problema original
El entrenamiento mostraba `matched=0` en todos los steps y `loss` plano (~1.9):
el modelo no veía ninguna caja GT. Causa raíz: la carga de anotaciones de
Argoverse2 fallaba en silencio (`except Exception` se tragaba el error).

---

## ✅ COMPLETADO

### 1. Carga de anotaciones (bug `matched=0`) — `telescope/data.py`
Tres bugs contra la API de `av2` 0.3.6, todos silenciados por `except Exception`:
- `get_camera_params` no existe → es `get_log_pinhole_camera(log_id, cam_name)`.
- `project_ego_to_img` devuelve **tupla `(uv, points_cam, is_valid)`**, no un array
  `(N,3)`; el check "delante de cámara" ahora usa `points_cam[:, 2] > 0`.
- `get_labels_at_lidar_timestamp` devuelve un `CuboidList`; hay que iterar
  `cuboid_list.cuboids`.
- Extra: `dst_SE3_object.translation` → `xyz_center_m`; el `except` ahora avisa
  con `warnings.warn` en vez de descartar en silencio.

### 2. Calentamiento de LR desperdiciaba la época 0 — `train.py`
El `LambdaLR` por época evaluaba la rampa en el índice 0 → factor `1e-8` (LR≈0
toda la primera época). Sustituido por **rampa lineal por iteración** sobre la
primera época (`set_lr(global_step)`), luego constante. Se eliminó el
`scheduler.step()` por época.

### 3. Ablación `--no_foveation` rota — `telescope/pipeline.py`, `train.py`
`forward()` nunca leía `_no_foveation`: la imagen se deformaba con la `R` del
estimador aleatorio congelado mientras la pérdida decodificaba con `R=0.001`
(warp/decode inconsistentes). Ahora `forward()` fuerza `R≈1e-3` → Φ=identidad
consistente en warp, embedding y decode. Verificado: `|c_eu − c_ri| = 0`.

### 4. Backbone real SAM 3.1 — `telescope/backbone_sam3.py` (NUEVO), `train.py`
`_load_sam3_backbone` era un `TODO` no-op; `--backbone_ckpt` se ignoraba y se
entrenaba con `SAM3EncoderStub` (convs 1×1 aleatorias congeladas).
- Nuevo `SAM3Backbone`: ViT (depth 32) + SimpleFPN neck reales, carga **solo** los
  pesos `detector.backbone.vision_backbone.*` del checkpoint multiplex (438 pesos,
  ignora 36 claves del neck interactivo SAM2).
- Neck con `scale_factors=(4.0,2.0,1.0)` → 3 niveles de 256 canales a 288²/144²/72²
  (img 1008); devueltos coarse→fine para casar con el contrato del stub.
- Normalización SAM3 `(x·2−1)` y resize a 1008 dentro del wrapper.
- Corre **siempre en eval** (override de `.train()`) para desactivar activation
  checkpointing y drop-path del ViT, y en **bf16 nativo** (el ViT da NaN en fp16),
  bajo `no_grad` cuando está congelado → no guarda activaciones del ViT.
- Verificado en GPU: carga limpia, **VRAM 9.1 GB** con batch 2 @ 1024 (GPU de 12 GB),
  loss baja, gradientes finitos a fov/emb/detr/box_head, backbone congelado.

### 5. Geometría inestable en fp16 (bug crítico latente) — `telescope/box.py`, `telescope/warp.py`
`EPS=1e-8` es subnormal en fp16 (**= 0.0**), así que `clamp(min=EPS)` no protegía y
`1/r`, `1/‖col‖` → inf → **grads NaN**. Estaba oculto porque con `matched=0` nunca
se ejercitaba el decode de cajas. Ahora `riemannian_to_euclidean_box`,
`euclidean_to_riemannian_box` y el grid del warp corren en **fp32 con autocast
desactivado**. También se bajó `init_scale` del GradScaler a `2**13`.

### 6. Denoising DINO-style — `telescope/pipeline.py`, `telescope/matcher.py`, `train.py`
`denoise_boxes` se importaba pero nunca se usaba. Implementado:
- `RealDeformableDetr`/`DeformableDetrStub` refactorizados en `encode`/`decode`
  para **reusar el encoder** (lo caro) entre el pase de matching y el de denoising.
- `TelescopeModel._denoising_pass`: genera queries de GT con ruido (label embedding
  + foveación, ref-points = centros ruidosos) y corre un **pase separado** del
  decoder (el decoder de HF no tiene máscara de self-attention → la concatenación
  filtraría las GT a las matching queries).
- `compute_denoising_loss` (matcher.py): pérdida L1+gIoU+cls directa 1-a-1, sin
  Hungarian.
- Flags en train.py: `--denoising` (ON por defecto), `--no_denoising`,
  `--dn_noise_scale 0.4`, `--dn_weight 1.0`.
- Verificado end-to-end: dn_loss baja, grad llega a `dn_label_emb`, path sin
  denoising sigue devolviendo la 5-tupla (retrocompatible).

---

## ✅ COMPLETADO — revisión pipeline vs paper (2026-06-02)

Verificado contra las ecuaciones del paper (versión HTML de arXiv) y con tests
numéricos (round-trip, `gradcheck`, forward/backward end-to-end en GPU fp16).
El núcleo geométrico (h, w, Φ, iteración inversa, α=p=2, caja Riemanniana, L1+gIoU,
denoising DINO) coincide con el paper.

### #3 — mAP por distancia — `telescope/eval.py`, `telescope/data.py`  ✅ (ya estaba)
`data.py` propaga `distances` por caja al `target`; `eval.py` tiene `DISTANCE_BINS`
(0–50/50–150/150–250/250 m+) y `_eval_distance_bin` (marca `ignore` las GT fuera del
bin y reusa COCOeval). `train.py` imprime el desglose.

### #4 — Gradiente de Φ⁻¹ a `o, R` — `telescope/geometry.py`
`HyperbolicInverseNR.backward` ahora propaga a `o,R` por el teorema de la función
implícita: con `g = J_x⁻ᵀ·dL/dx*`, devuelve `dL/do = −Φ_oᵀ·g`, `dL/dR = −Φ_Rᵀ·g`
(VJP por autograd a través de Φ en el punto fijo). Verificado con `gradcheck` (doble
precisión); el warp ya pasa gradiente a `o,R` (antes `None`). Nota: con el backbone
real bajo `no_grad` (memoria), el camino warp→features sigue cortado; la señal a
`o,R` llega por el decode (ahora vía centre-inversion además del Jacobiano).

### #5 (resolución/features) — estimador de foveación — `telescope/pipeline.py`
Ya no usa un `SAM3EncoderStub` aleatorio separado a 64²: usa el **backbone
compartido** (SAM3 real cuando se carga) sobre la imagen sin deformar a **512²**
(paper: 256/512). Eliminado `param_encoder`; `fov_estimator` toma `query_dim*3`.
Notebook 05 actualizado. (Coste: el backbone corre 2× por step; bajo `no_grad` el
pico de memoria no se acumula.)

### #6 — `RealDeformableDetr` incompatible con `transformers` 5.x — `telescope/pipeline.py`
En `transformers≥5`, `DeformableDetrSinePositionEmbedding` ya devuelve `(B,H*W,C)`;
el código lo re-aplanaba y reventaba en el primer forward (`__init__` solo captura
`ImportError`, así que no caía al stub → el entrenamiento abortaba). Ahora `encode`
detecta `pos.dim()==4` y solo aplana en la API vieja. Verificado con 5.6.2.

### #7 — clasificación sin ponderar fondo — `telescope/matcher.py`
`match_and_compute_loss` usa `eos_coef=0.1` (estilo DETR) para que el fondo no
domine la cross-entropy de las queries.

### #8 — `eval.py`/`compare.py` — `eval.py`, `compare.py`
`eval.py --backbone_ckpt`: reconstruye `SAM3Backbone` antes de `load_state_dict`
(si no, las claves no casan al evaluar un modelo entrenado con el backbone real).
`compare.py` usa las claves de bins de distancia reales (`mAP_0_50`, …).

---

## ⏳ PENDIENTE

### Thresholds de tamaño COCO en `[-1,1]` — `telescope/eval.py`
Los splits small/medium/large de COCO (32²/96² px) no tienen sentido con cajas en
`[-1,1]` (todo cae en "small"). Los bins por distancia son la métrica del paper y
sí funcionan; los de tamaño son ruido informativo.

### Warp→features con backbone real congelado — `telescope/backbone_sam3.py`
Para que la foveación aprenda por el re-muestreo (señal STN completa) haría falta no
envolver el backbone congelado en `no_grad` — cuesta memoria (activaciones del ViT).
Tradeoff abierto: memoria vs señal del warp a `o,R`.

---

## Verificación end-to-end con datos reales (2026-06-02)

Tras descargar Argoverse 2 (subset 5 train + 5 val, ~10 GB) e instalar los extras:

- **Anotaciones reales cargan → `matched>0`.** 783 frames en val; las cajas se
  proyectan con distancias (16–126 m) y clases correctas. El bug original
  `matched=0` está confirmado muerto sobre datos reales (no solo sintéticos).
- **Bucle de entrenamiento completo OK sobre datos reales:** loss baja
  (12.6→10.6), `matched` = nº de GT cada paso, `fov_estimator` recibe gradiente.
  Verificado con el **stub** a 256² batch 1 (pico 3.0 GB).
- **Backbone SAM 3.1 real carga** (438 pesos vision, ignora 36 del neck SAM2) y el
  Deformable DETR real se construye con **`transformers` 5.9** (el fix del
  pos-embed #6 funciona). Pero **da OOM a 1024/640/512** en esta GPU (ver abajo).
- **`einops` faltaba en el venv** (lo importa el repo `sam3`): añadido a
  `requirements-train.txt`.
- **Prueba de corrección por overfitting:** con el backbone *entrenable* y 2 frames de
  objetos grandes/cercanos, el modelo sobreajusta a **`mAP50 = 1.000`** (loss 2.6→0.43).
  Como `mAP50` pasa de 0 → 1.0, queda probado que el camino warp→DETR→Riemann→Φ⁻¹→
  matching→pérdida **localiza correctamente, sin bug**. El `mAP≈0` del entrenamiento real
  es solo el setup débil (stub congelado + 5 logs + pocas épocas + 256 px), no un error de
  código. (Un primer overfit dio mAP50 bajo por dos *confounds*: 28 objetos/frame muchos a
  100 m+ = 2-3 px a 256 px, y el stub **congelado**; al quitarlos → 1.0.)

## Notas de verificación / entorno
- **Máquina actual: GPU de 8 GB** (RTX 3070 Ti Laptop), con DaVinci Resolve
  ocupando ~3.4 GB → el backbone SAM 3.1 real **no cabe** (OOM a cualquier
  resolución; el ViT corre a 1008² fijo). Entrenar fiel al paper requiere
  **12 GB+** (la nota previa de "12 GB, batch 2" era de otra máquina).
- `flash_attn` NO instalado; SAM3 corre con SDPA (`use_fa3=False`).
- El backbone real requiere `query_dim=256` (d_model del neck SAM3).
- En esta máquina solo caben smoke tests (stub, ≤256², batch 1); las cifras del
  paper requieren GPU 12 GB+ y el split completo (Argoverse2 ≠ TruckDrive).
